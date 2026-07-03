from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterator

import torch

__all__ = [
    "compute_curvature_rate_batch",
    "compute_mean_abs_jerk_batch",
    "compute_replan_consistency_batch",
    "compute_trajectory_consistency_batch",
    "consecutive_frame_pairs",
    "group_frames_by_scenario",
    "inter_frame_transform",
    "parse_frame_key",
]

TRAJECTORY_CONSISTENCY_DT = 0.1
TRAJECTORY_CONSISTENCY_VEL_TOL = 1.0
TRAJECTORY_CONSISTENCY_ACC_TOL = 2.0
TRAJECTORY_CONSISTENCY_JERK_TOL = 5.0
TRAJECTORY_CONSISTENCY_HEADING_TOL = 0.2
TRAJECTORY_CONSISTENCY_HEADING_RATE_TOL = 0.5


def _angle_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    diff = lhs - rhs
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _build_sg_diff_kernel(window: int, poly: int, deriv: int, delta: float, device) -> torch.Tensor:
    half = window // 2
    x = torch.arange(-half, half + 1, dtype=torch.float64, device=device)
    powers = torch.arange(poly + 1, dtype=torch.float64, device=device)
    design = x[:, None].pow(powers[None])
    coeff = torch.linalg.pinv(design)[deriv] * (math.factorial(deriv) / (delta**deriv))
    return coeff.to(dtype=torch.float32)


def _odd_window(length: int, max_window: int = 11) -> int:
    window = min(max_window, length)
    if window % 2 == 0:
        window -= 1
    return window


@torch.no_grad()
def compute_mean_abs_jerk_batch(ego_trajs: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Mean absolute XY jerk in m/s^3; lower is smoother."""
    n, t, _ = ego_trajs.shape
    window = _odd_window(t)
    if window < 5:
        return torch.zeros(n, device=ego_trajs.device, dtype=ego_trajs.dtype)

    kernel = _build_sg_diff_kernel(window, poly=3, deriv=3, delta=dt, device=ego_trajs.device).to(
        dtype=ego_trajs.dtype
    )
    pad = window // 2
    xy = ego_trajs[..., :2].detach().permute(0, 2, 1)
    xy = torch.nn.functional.pad(xy, (pad, pad), mode="replicate")
    jerk = torch.nn.functional.conv1d(xy, kernel.view(1, 1, -1).expand(2, 1, -1), groups=2)
    jerk_mag = jerk.norm(dim=1)
    if jerk_mag.shape[1] > 2 * pad + 1:
        jerk_mag = jerk_mag[:, pad:-pad]
    return jerk_mag.mean(dim=1)


@torch.no_grad()
def compute_curvature_rate_batch(ego_trajs: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Mean absolute curvature-rate proxy; lower means less steering oscillation."""
    n, t, _ = ego_trajs.shape
    window = _odd_window(t)
    if window < 5:
        return torch.zeros(n, device=ego_trajs.device, dtype=ego_trajs.dtype)

    device = ego_trajs.device
    dtype = ego_trajs.dtype
    vel_kernel = _build_sg_diff_kernel(window, poly=3, deriv=1, delta=dt, device=device).to(dtype)
    acc_kernel = _build_sg_diff_kernel(window, poly=3, deriv=2, delta=dt, device=device).to(dtype)
    pad = window // 2
    xy = ego_trajs[..., :2].detach().permute(0, 2, 1)
    xy = torch.nn.functional.pad(xy, (pad, pad), mode="replicate")
    vel = torch.nn.functional.conv1d(xy, vel_kernel.view(1, 1, -1).expand(2, 1, -1), groups=2)
    acc = torch.nn.functional.conv1d(xy, acc_kernel.view(1, 1, -1).expand(2, 1, -1), groups=2)

    vx, vy = vel[:, 0], vel[:, 1]
    ax, ay = acc[:, 0], acc[:, 1]
    speed = torch.sqrt(torch.clamp(vx * vx + vy * vy, min=1.0e-4))
    curvature = (vx * ay - vy * ax) / torch.clamp(speed.pow(3), min=1.0e-3)
    curvature_rate = torch.diff(curvature, dim=1).abs() / dt
    if curvature_rate.shape[1] > 2 * pad + 1:
        curvature_rate = curvature_rate[:, pad:-pad]
    return curvature_rate.mean(dim=1)


@torch.no_grad()
def compute_trajectory_consistency_batch(
    ego_pred: torch.Tensor,
    ego_past: torch.Tensor,
    ego_current: torch.Tensor,
    dt: float = TRAJECTORY_CONSISTENCY_DT,
) -> dict[str, torch.Tensor]:
    pred_xy = ego_pred[..., :2]
    current_xy = ego_current[:, :2]
    batch_size = pred_xy.shape[0]
    device = pred_xy.device
    dtype = pred_xy.dtype

    points = torch.cat([current_xy[:, None], pred_xy], dim=1)
    velocity = (points[:, 1:] - points[:, :-1]) / dt
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt if velocity.shape[1] > 1 else None
    jerk = (
        (acceleration[:, 1:] - acceleration[:, :-1]) / dt
        if acceleration is not None and acceleration.shape[1] > 1
        else None
    )

    if ego_past.shape[1] >= 2:
        past_velocity = (current_xy - ego_past[:, -2, :2]) / dt
        velocity_boundary = torch.linalg.vector_norm(velocity[:, 0] - past_velocity, dim=-1)
    else:
        past_velocity = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        velocity_boundary = torch.zeros(batch_size, device=device, dtype=dtype)

    if ego_past.shape[1] >= 3 and acceleration is not None and acceleration.shape[1] > 0:
        previous_past_velocity = (ego_past[:, -2, :2] - ego_past[:, -3, :2]) / dt
        past_acceleration = (past_velocity - previous_past_velocity) / dt
        acceleration_boundary = torch.linalg.vector_norm(
            acceleration[:, 0] - past_acceleration, dim=-1
        )
    else:
        acceleration_boundary = torch.zeros(batch_size, device=device, dtype=dtype)

    if jerk is not None and jerk.numel() > 0:
        jerk_rms = torch.sqrt(torch.clamp(torch.sum(jerk**2, dim=-1).mean(dim=1), min=0.0))
    else:
        jerk_rms = torch.zeros(batch_size, device=device, dtype=dtype)

    if ego_pred.shape[-1] >= 4 and ego_current.shape[-1] >= 4:
        pred_heading = torch.atan2(ego_pred[..., 3], ego_pred[..., 2])
        current_heading = torch.atan2(ego_current[:, 3], ego_current[:, 2])
        heading = torch.cat([current_heading[:, None], pred_heading], dim=1)
        heading_step = _angle_diff(heading[:, 1:], heading[:, :-1])
        heading_boundary = heading_step[:, 0].abs()
        heading_rate = heading_step / dt
        if heading_rate.shape[1] > 1:
            heading_rate_change = (heading_rate[:, 1:] - heading_rate[:, :-1]).abs().mean(dim=1)
        else:
            heading_rate_change = torch.zeros(batch_size, device=device, dtype=dtype)
    else:
        heading_boundary = torch.zeros(batch_size, device=device, dtype=dtype)
        heading_rate_change = torch.zeros(batch_size, device=device, dtype=dtype)

    consistency_error = torch.sqrt(
        (
            (velocity_boundary / TRAJECTORY_CONSISTENCY_VEL_TOL) ** 2
            + (acceleration_boundary / TRAJECTORY_CONSISTENCY_ACC_TOL) ** 2
            + (jerk_rms / TRAJECTORY_CONSISTENCY_JERK_TOL) ** 2
            + (heading_boundary / TRAJECTORY_CONSISTENCY_HEADING_TOL) ** 2
            + (heading_rate_change / TRAJECTORY_CONSISTENCY_HEADING_RATE_TOL) ** 2
        )
        / 5.0
    )

    return {
        "ego_trajectory_consistency_error": consistency_error,
        "ego_trajectory_consistency_velocity_boundary": velocity_boundary,
        "ego_trajectory_consistency_acceleration_boundary": acceleration_boundary,
        "ego_trajectory_consistency_jerk": jerk_rms,
        "ego_trajectory_consistency_heading_boundary": heading_boundary,
        "ego_trajectory_consistency_heading_rate_change": heading_rate_change,
    }


def parse_frame_key(path: str) -> tuple[str, int]:
    base = os.path.basename(path)
    name = base[:-4] if base.endswith(".npz") else base
    int_tokens: list[str] = []
    for token in reversed(name.split("_")):
        if token.isdigit():
            int_tokens.append(token)
        else:
            break
    int_tokens.reverse()
    if not int_tokens:
        raise ValueError(f"cannot parse trailing frame index from {path!r}")
    frame = int(int_tokens[-1])
    scene = int_tokens[-2] if len(int_tokens) >= 2 else None
    group = os.path.dirname(path) if scene is None else f"{os.path.dirname(path)}#{scene}"
    return group, frame


def group_frames_by_scenario(paths: list[str]) -> dict[str, list[tuple[int, str]]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for path in paths:
        group, frame = parse_frame_key(path)
        groups.setdefault(group, []).append((frame, path))
    for group_frames in groups.values():
        group_frames.sort(key=lambda item: item[0])
    return groups


def consecutive_frame_pairs(
    paths: list[str], expected_gap: int | None = None
) -> Iterator[tuple[int, str, int, str, int]]:
    groups = group_frames_by_scenario(paths)
    for group in sorted(groups):
        frames = groups[group]
        if len(frames) < 2:
            continue
        gaps = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
        step = expected_gap if expected_gap is not None else Counter(gaps).most_common(1)[0][0]
        for (idx_a, path_a), (idx_b, path_b) in zip(frames[:-1], frames[1:]):
            if idx_b - idx_a == step:
                yield idx_a, path_a, idx_b, path_b, idx_b - idx_a


def inter_frame_transform(
    future_a: torch.Tensor, step_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if step_offset <= 0 or step_offset > future_a.shape[-2]:
        raise ValueError(f"step_offset must be in [1, {future_a.shape[-2]}], got {step_offset}")
    pose = future_a[..., step_offset - 1, :]
    return pose[..., :2], torch.atan2(pose[..., 3], pose[..., 2])


@torch.no_grad()
def compute_replan_consistency_batch(
    traj_a: torch.Tensor,
    traj_b: torch.Tensor,
    step_offset: int,
    rel_pos: torch.Tensor,
    rel_heading: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    n, ta, _ = traj_a.shape
    tb = traj_b.shape[1]
    overlap = min(ta - step_offset, tb)
    if overlap <= 0:
        zeros = torch.zeros(n, device=traj_a.device, dtype=traj_a.dtype)
        return {"position_jump": zeros, "heading_jump": zeros, "overlap_len": 0}

    a = traj_a[:, step_offset : step_offset + overlap]
    b = traj_b[:, :overlap]
    theta = rel_heading.view(n, 1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    delta = a[..., :2] - rel_pos.view(n, 1, 2)
    a_xy = torch.stack(
        [
            cos_t * delta[..., 0] + sin_t * delta[..., 1],
            -sin_t * delta[..., 0] + cos_t * delta[..., 1],
        ],
        dim=-1,
    )
    heading_a = torch.atan2(a[..., 3], a[..., 2]) - theta
    heading_b = torch.atan2(b[..., 3], b[..., 2])
    heading_delta = torch.atan2(torch.sin(heading_a - heading_b), torch.cos(heading_a - heading_b))
    return {
        "position_jump": (a_xy - b[..., :2]).norm(dim=-1).mean(dim=1),
        "heading_jump": heading_delta.abs().mean(dim=1),
        "overlap_len": overlap,
    }
