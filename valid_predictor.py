import argparse
import torch
from torch.utils.data import DataLoader
from timm.utils import ModelEma

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.train_utils import set_seed, resume_model
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils import ddp

from torch import optim
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def validate_model(model, val_loader, args, device) -> tuple[float, float]:
    """return: ave_loss_ego, ave_loss_neighbor"""
    model.eval()
    total_loss_ego = 0.0
    total_loss_neighbor = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            # データの準備
            inputs = {
                "ego_current_state": batch[0].to(device),
                "neighbor_agents_past": batch[2].to(device),
                "lanes": batch[4].to(device),
                "lanes_speed_limit": batch[5].to(device),
                "lanes_has_speed_limit": batch[6].to(device),
                "route_lanes": batch[7].to(device),
                "route_lanes_speed_limit": batch[8].to(device),
                "route_lanes_has_speed_limit": batch[9].to(device),
                "static_objects": batch[10].to(device),
            }

            B = inputs["ego_current_state"].shape[0]

            ego_future = batch[1].to(device)
            ego_future = torch.cat(
                [
                    ego_future[..., :2],
                    ego_future[..., 2:3].cos(),
                    ego_future[..., 2:3].sin(),
                ],
                dim=-1,
            )  # (B, T, 4)
            print(f"{ego_future.shape=}")
            neighbors_future = batch[3].to(args.device)
            neighbor_future_mask = (
                torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            )  # (B, Pn, T)
            neighbors_future = torch.cat(
                [
                    neighbors_future[..., :2],
                    neighbors_future[..., 2:3].cos(),
                    neighbors_future[..., 2:3].sin(),
                ],
                dim=-1,
            )  # (B, Pn, T, 4)
            neighbors_future[neighbor_future_mask] = 0.0

            B, Pn, T, _ = neighbors_future.shape
            ego_current, neighbors_current = (
                inputs["ego_current_state"][:, :4],
                inputs["neighbor_agents_past"][:, :Pn, -1, :4],
            )
            inputs = args.observation_normalizer(inputs)

            _, outputs = model(inputs)

            neighbor_current_mask = (
                torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
            )  # (B, Pn)
            neighbor_mask = torch.concat(
                (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
            )  # (B, Pn, T + 1)

            gt_future = torch.cat(
                [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
            )  # (B, Pn + 1, T, 4)
            current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
            # (B, Pn + 1, 4)

            all_gt = torch.cat(
                [current_states[:, :, None, :], gt_future], dim=2
            )  # (B, Pn + 1, T + 1, 4)
            all_gt[:, 1:][neighbor_mask] = 0.0

            prediction = outputs["prediction"]

            neighbors_future_valid = ~neighbor_future_mask
            all_gt = all_gt[:, :, 1:, :] # (B, Pn + 1, T, 4)
            loss_tensor = (prediction - all_gt) ** 2
            loss_ego = loss_tensor[:, 0, :]
            loss_nei = loss_tensor[:, 1:, :]
            loss_nei = loss_nei[neighbors_future_valid]
            total_loss_ego += loss_ego.mean().item() * B
            total_loss_neighbor += loss_nei.mean().item() * B
            total_samples += B

    avg_loss_ego = total_loss_ego / total_samples
    avg_loss_neighbor = total_loss_neighbor / total_samples
    return avg_loss_ego, avg_loss_neighbor


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    # Arguments
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--train_set", type=str, help="path to train data", default=None
    )
    parser.add_argument(
        "--train_set_list", type=str, help="data list of train data", default=None
    )

    parser.add_argument(
        "--future_len", type=int, help="number of time point", default=80
    )
    parser.add_argument("--time_len", type=int, help="number of time point", default=21)

    parser.add_argument(
        "--agent_state_dim", type=int, help="past state dim for agents", default=11
    )
    parser.add_argument("--agent_num", type=int, help="number of agents", default=32)

    parser.add_argument(
        "--static_objects_state_dim",
        type=int,
        help="state dim for static objects",
        default=10,
    )
    parser.add_argument(
        "--static_objects_num", type=int, help="number of static objects", default=5
    )

    parser.add_argument("--lane_len", type=int, help="number of lane point", default=20)
    parser.add_argument(
        "--lane_state_dim", type=int, help="state dim for lane point", default=12
    )
    parser.add_argument("--lane_num", type=int, help="number of lanes", default=70)

    parser.add_argument(
        "--route_len", type=int, help="number of route lane point", default=20
    )
    parser.add_argument(
        "--route_state_dim", type=int, help="state dim for route lane point", default=12
    )
    parser.add_argument(
        "--route_num", type=int, help="number of route lanes", default=25
    )

    # DataLoader parameters
    parser.add_argument(
        "--augment_prob", type=float, help="augmentation probability", default=0.5
    )
    parser.add_argument(
        "--normalization_file_path",
        default="normalization.json",
        help="filepath of normalizaiton.json",
        type=str,
    )
    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, help="fix random seed", default=3407)
    parser.add_argument(
        "--train_epochs", type=int, help="epochs of training", default=500
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size (default: 2048)", default=1024
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate (default: 5e-4)",
        default=5e-4,
    )
    parser.add_argument(
        "--warm_up_epoch", type=int, help="number of warm up", default=5
    )
    parser.add_argument(
        "--encoder_drop_path_rate",
        type=float,
        help="encoder drop out rate",
        default=0.1,
    )
    parser.add_argument(
        "--decoder_drop_path_rate",
        type=float,
        help="decoder drop out rate",
        default=0.1,
    )

    parser.add_argument(
        "--alpha_planning_loss",
        type=float,
        help="coefficient of planning loss (default: 1.0)",
        default=1.0,
    )

    parser.add_argument(
        "--device", type=str, help="run on which device (default: cuda)", default="cuda"
    )

    parser.add_argument("--use_ema", default=True, type=boolean)

    # Model
    parser.add_argument(
        "--encoder_depth", type=int, help="number of encoding layers", default=3
    )
    parser.add_argument(
        "--decoder_depth", type=int, help="number of decoding layers", default=3
    )
    parser.add_argument("--num_heads", type=int, help="number of multi-head", default=6)
    parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=192)
    parser.add_argument(
        "--diffusion_model_type",
        type=str,
        help="type of diffusion model [x_start, score]",
        choices=["score", "x_start"],
        default="x_start",
    )

    # decoder
    parser.add_argument(
        "--predicted_neighbor_num",
        type=int,
        help="number of neighbor agents to predict",
        default=10,
    )
    parser.add_argument(
        "--resume_model_path", type=str, help="path to resume model", required=True
    )

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument("--notes", default="", type=str)

    # distributed training parameters
    parser.add_argument("--ddp", default=True, type=boolean, help="use ddp or not")
    parser.add_argument("--port", default="22323", type=str, help="port")

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args


if __name__ == "__main__":
    args = get_args()

    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        # Logging
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    # set up data loaders
    aug = (
        StatePerturbation(augment_prob=args.augment_prob, device=args.device)
        if args.use_data_augment
        else None
    )
    train_set = DiffusionPlannerData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
    )
    train_sampler = DistributedSampler(
        train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(
        rank if args.device == "cuda" else args.device
    )

    if args.ddp:
        diffusion_planner = DDP(
            diffusion_planner, device_ids=[rank], find_unused_parameters=False
        )

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(
                    p.numel()
                    for p in ddp.get_model(diffusion_planner, args.ddp).parameters()
                )
            )
        )

    # optimizer
    params = [
        {
            "params": ddp.get_model(diffusion_planner, args.ddp).parameters(),
            "lr": args.learning_rate,
        }
    ]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, train_epochs, args.warm_up_epoch
    )

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = (
            resume_model(
                args.resume_model_path,
                diffusion_planner,
                optimizer,
                scheduler,
                model_ema,
                args.device,
            )
        )
    else:
        init_epoch = 0
        wandb_id = None

    if args.ddp:
        torch.distributed.barrier()

    avg_loss_ego, ave_loss_neighbor = validate_model(
        diffusion_planner, train_loader, args, args.device
    )
    print(f"{avg_loss_ego=:.4f} {ave_loss_neighbor=:.4f}")
