import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from diffusion_planner.utils.normalizer import ObservationNormalizer
from copy import deepcopy


def visualize_inputs(inputs: dict, obs_noramlizer: ObservationNormalizer, save_path: Path):
    """
    draw the input data of the diffusion_planner model on the xy plane
    """
    inputs = deepcopy(inputs)
    inputs = obs_noramlizer.inverse(inputs)

    # Function to convert PyTorch tensors to NumPy arrays
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    for key in inputs:
        inputs[key] = to_numpy(inputs[key])
        print(f"{key=}, {inputs[key].shape=}")

    """
    key='ego_current_state', inputs[key].shape=(1, 10)
    key='neighbor_agents_past', inputs[key].shape=(1, 32, 21, 11)
    key='lanes', inputs[key].shape=(1, 70, 20, 12)
    key='lanes_speed_limit', inputs[key].shape=(1, 70, 1)
    key='lanes_has_speed_limit', inputs[key].shape=(1, 70, 1)
    key='route_lanes', inputs[key].shape=(1, 25, 20, 12)
    key='route_lanes_speed_limit', inputs[key].shape=(1, 25, 1)
    key='route_lanes_has_speed_limit', inputs[key].shape=(1, 25, 1)
    key='static_objects', inputs[key].shape=(1, 5, 10)
    key='sampled_trajectories', inputs[key].shape=(1, 11, 81, 4)
    key='diffusion_time', inputs[key].shape=(1,)
    """

    # initialize the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # ==== Ego ====
    ego_state = inputs["ego_current_state"][0]  # Use the first sample in the batch
    print(f"{ego_state=}")
    ego_x, ego_y = ego_state[0], ego_state[1]
    ego_heading = np.arctan2(ego_state[3], ego_state[2])

    # Ego vehicle's length and width
    car_length = 4.5  # Assumed value for vehicle length
    car_width = 2.0  # Assumed value for vehicle width
    dx = car_length / 2 * np.cos(ego_heading)
    dy = car_length / 2 * np.sin(ego_heading)

    # Draw the ego vehicle as an arrow
    ax.arrow(
        ego_x,
        ego_y,
        dx,
        dy,
        width=car_width / 2,
        head_width=car_width,
        head_length=car_length / 3,
        fc="r",
        ec="r",
        alpha=0.7,
        label="Ego Vehicle",
    )

    # ==== Neighbor agents ====
    neighbors = inputs["neighbor_agents_past"][0]  # Use the first sample in the batch
    last_timestep = neighbors.shape[1] - 1

    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i, last_timestep]

        # Skip zero vectors (masked objects)
        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            print(f"Agent {i} is skpped.")
            continue
        print(f"Agent {i} {neighbor}")

        n_x, n_y = neighbor[0], neighbor[1]
        n_heading = np.arctan2(neighbor[3], neighbor[2])

        # Set color and shape dimensions based on the vehicle type
        vehicle_type = np.argmax(neighbor[8:11]) if neighbor.shape[0] > 8 else 0
        if vehicle_type == 0:  # Vehicle
            color = "blue"
            shape_length = 4.0
            shape_width = 1.8
        elif vehicle_type == 1:  # Pedestrian
            color = "green"
            shape_length = 1.0
            shape_width = 0.5
        else:  # Bicycle
            color = "purple"
            shape_length = 1.8
            shape_width = 0.5

        # Draw the past trajectory as a dashed line
        past_x = [neighbors[i, t, 0] for t in range(last_timestep + 1)]
        past_y = [neighbors[i, t, 1] for t in range(last_timestep + 1)]
        ax.plot(past_x, past_y, color=color, alpha=0.9, linestyle="--")

        # Draw the current position as an arrow
        dx = shape_length / 2 * np.cos(n_heading)
        dy = shape_length / 2 * np.sin(n_heading)
        ax.arrow(
            n_x,
            n_y,
            dx,
            dy,
            width=shape_width / 2,
            head_width=shape_width,
            head_length=shape_length / 3,
            fc=color,
            ec=color,
            alpha=0.5,
        )
        ax.text(
            n_x + 10,
            n_y,
            f"Agent {i}",
            fontsize=8,
            color=color,
            ha="center",
            va="center",
        )

    # ==== Static objects ====
    static_objects = inputs["static_objects"][0]  # Use the first sample in the batch

    for i in range(static_objects.shape[0]):
        obj = static_objects[i]

        # Skip zero vectors (masked objects)
        if np.sum(np.abs(obj[:4])) < 1e-6:
            continue

        obj_x, obj_y = obj[0], obj[1]
        obj_heading = np.arctan2(obj[3], obj[2])
        obj_width = obj[4] if obj.shape[0] > 4 else 1.0
        obj_length = obj[5] if obj.shape[0] > 5 else 1.0

        # Set color based on the object type
        obj_type = np.argmax(obj[-4:]) if obj.shape[0] >= 10 else 0
        colors = ["orange", "gray", "yellow", "brown"]
        obj_color = colors[obj_type % len(colors)]

        # Draw the object as a rectangle
        rect = plt.Rectangle(
            (obj_x - obj_length / 2, obj_y - obj_width / 2),
            obj_length,
            obj_width,
            angle=np.degrees(obj_heading),
            color=obj_color,
            alpha=0.4,
        )
        ax.add_patch(rect)

    # ==== Lanes ====
    lanes = inputs["lanes"][0]  # Use the first sample in the batch

    for i in range(lanes.shape[0]):
        for j in range(lanes.shape[1]):
            lane_point = lanes[i, j]

            # Skip zero vectors (masked objects)
            if np.sum(np.abs(lane_point[:4])) < 1e-6:
                continue

            # Draw the lane boundaries
            if np.sum(np.abs(lane_point[4:8])) > 1e-6:
                left_x = lane_point[0] + lane_point[4]
                left_y = lane_point[1] + lane_point[5]
                ax.plot([lane_point[0], left_x], [lane_point[1], left_y], "y-", alpha=0.1)

                right_x = lane_point[0] + lane_point[6]
                right_y = lane_point[1] + lane_point[7]
                ax.plot([lane_point[0], right_x], [lane_point[1], right_y], "y-", alpha=0.1)

            # Draw the lane centerline
            if j + 1 < lanes.shape[1] and np.sum(np.abs(lanes[i, j + 1, :4])) > 1e-6:
                next_point = lanes[i, j + 1]
                ax.plot(
                    [lane_point[0], next_point[0]],
                    [lane_point[1], next_point[1]],
                    "r-",
                    alpha=0.1,
                    linewidth=1,
                )
                next_left_x = next_point[0] + next_point[4]
                next_left_y = next_point[1] + next_point[5]
                ax.plot(
                    [left_x, next_left_x],
                    [left_y, next_left_y],
                    "k-",
                    alpha=0.25,
                    linewidth=1,
                )

                next_right_x = next_point[0] + next_point[6]
                next_right_y = next_point[1] + next_point[7]
                ax.plot(
                    [right_x, next_right_x],
                    [right_y, next_right_y],
                    "k-",
                    alpha=0.25,
                    linewidth=1,
                )


    # プロットの装飾
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # エゴ車両中心の表示範囲を設定
    view_range = 100
    ax.set_xlim(ego_x - view_range, ego_x + view_range)
    ax.set_ylim(ego_y - view_range, ego_y + view_range)

    # 凡例を追加
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
