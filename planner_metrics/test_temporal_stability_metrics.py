import torch

from planner_metrics.temporal_stability import (
    compute_curvature_rate_batch,
    compute_mean_abs_jerk_batch,
    compute_replan_consistency_batch,
    consecutive_frame_pairs,
    inter_frame_transform,
    parse_frame_key,
)


def _straight(length=80, offset=0.0):
    x = torch.arange(1, length + 1, dtype=torch.float32) + offset
    y = torch.zeros_like(x)
    cos = torch.ones_like(x)
    sin = torch.zeros_like(x)
    return torch.stack([x, y, cos, sin], dim=-1)


def test_parse_frame_key_basic_dataset_name():
    group, frame = parse_frame_key("/data/10-55-28/10-55-28_00000001_00000034.npz")
    assert group == "/data/10-55-28#00000001"
    assert frame == 34


def test_consecutive_frame_pairs_uses_modal_gap_and_skips_breaks():
    paths = [
        "/data/a/scene_00000001_00000001.npz",
        "/data/a/scene_00000001_00000004.npz",
        "/data/a/scene_00000001_00000007.npz",
        "/data/a/scene_00000001_00000020.npz",
    ]
    pairs = list(consecutive_frame_pairs(paths))
    assert [(a, b, gap) for a, _, b, _, gap in pairs] == [(1, 4, 3), (4, 7, 3)]


def test_replan_consistency_zero_for_perfect_straight_overlap():
    traj_a = _straight().unsqueeze(0)
    traj_b = _straight(offset=0.0).unsqueeze(0)
    rel_pos, rel_heading = inter_frame_transform(traj_a, step_offset=1)
    out = compute_replan_consistency_batch(traj_a, traj_b, 1, rel_pos, rel_heading)
    assert out["overlap_len"] == 79
    assert torch.allclose(out["position_jump"], torch.zeros(1), atol=1.0e-6)
    assert torch.allclose(out["heading_jump"], torch.zeros(1), atol=1.0e-6)


def test_single_trajectory_stability_zero_for_straight_constant_speed():
    traj = _straight().unsqueeze(0)
    jerk = compute_mean_abs_jerk_batch(traj)
    curvature_rate = compute_curvature_rate_batch(traj)
    assert torch.allclose(jerk, torch.zeros_like(jerk), atol=1.0e-4)
    assert torch.allclose(curvature_rate, torch.zeros_like(curvature_rate), atol=1.0e-4)
