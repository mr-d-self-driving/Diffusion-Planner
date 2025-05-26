import torch
import torch.nn.functional as F


def route_following_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """
    x: [B * Pn+1, T + 1, 4]
    t: [B, 1],
    inputs: Dict[str, torch.Tensor]
    """
    B, P, T, _ = x.shape
    route_lanes = inputs["route_lanes"]
    # print(route_lanes.shape)  # torch.Size([B=1024, SegNum=25, PointNum=20, FeatureDim=12])
    route_lanes = route_lanes.reshape(B, 25 * 20, 12)  # [B, SegNum * PointNum, FeatureDim]
    route_lanes = route_lanes[:, :, :2]  # [B, SegNum * PointNum, 2]

    x: torch.Tensor = x.reshape(B, P, -1, 4)
    mask_diffusion_time = (t < 0.1) * (t > 0.005)
    mask_diffusion_time = mask_diffusion_time.view(B, 1, 1, 1)
    x = torch.where(mask_diffusion_time, x, x.detach())

    predictions = x[:, 0, :, :2]  # [B, T + 1, 2]

    pred_points = predictions[:, :T]
    # route_lanesを拡張 [B, 1, SegNum*PointNum, 2]
    expanded_routes = route_lanes.unsqueeze(1)
    # 予測点を拡張 [B, T, 1, 2]
    expanded_preds = pred_points.unsqueeze(2)
    # 全ての距離を一度に計算 [B, T, SegNum*PointNum]
    distances = torch.norm(expanded_preds - expanded_routes, dim=-1)
    # 各バッチ、各タイムステップでの最小距離を取得 [B, T]
    min_distances = torch.min(distances, dim=2)[0]
    # 時間軸に沿って合計し、符号を反転 [B]
    reward = -torch.sum(min_distances, dim=1)

    return reward
