import cv2
import numpy as np
import torch

from diffusion_planner.utils.normalizer import ObservationNormalizer


def visualize_inputs_cv2(
    inputs: dict,
    obs_normalizer: ObservationNormalizer,
    image_size: tuple = (600, 600),
    view_range: float = 55.0,
) -> np.ndarray:
    """
    Draw the input data of the diffusion_planner model as a cv2 image
    Returns the image as a numpy array suitable for neural network input
    """
    inputs = obs_normalizer.inverse(inputs)

    # Function to convert PyTorch tensors to NumPy arrays
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    for key in inputs:
        inputs[key] = to_numpy(inputs[key])

    # Create a blank image
    img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # Get ego position for centering
    ego_state = inputs["ego_current_state"][0]
    ego_x, ego_y = ego_state[0], ego_state[1]

    # Define coordinate transformation functions
    def world_to_image(x, y):
        """Convert world coordinates to image coordinates"""
        # Center the ego vehicle in the image
        img_x = int((x - ego_x + view_range) * image_size[0] / (2 * view_range))
        img_y = int((view_range - (y - ego_y)) * image_size[1] / (2 * view_range))
        return img_x, img_y

    def draw_rotated_rect(img, center_x, center_y, width, height, angle, color, thickness=-1):
        """Draw a rotated rectangle"""
        center = world_to_image(center_x, center_y)
        # Scale dimensions to image space
        scaled_width = int(width * image_size[0] / (2 * view_range))
        scaled_height = int(height * image_size[1] / (2 * view_range))

        # Create rectangle points
        rect = cv2.boxPoints(
            ((center[0], center[1]), (scaled_width, scaled_height), -np.degrees(angle))
        )
        rect = np.intp(rect)

        if thickness == -1:
            cv2.fillPoly(img, [rect], color)
        else:
            cv2.drawContours(img, [rect], 0, color, thickness)

    def draw_arrow(img, start_x, start_y, end_x, end_y, color, thickness=2):
        """Draw an arrow from start to end"""
        start = world_to_image(start_x, start_y)
        end = world_to_image(end_x, end_y)
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.3)

    # ==== Lanes (draw first as background) ====
    lanes = inputs["lanes"][0]
    for i in range(lanes.shape[0]):
        traffic_light = lanes[i, 0, 8:12]
        if traffic_light[0] == 1:
            color = (0, 255, 0)  # Green
        elif traffic_light[1] == 1:
            color = (0, 255, 255)  # Yellow
        elif traffic_light[2] == 1:
            color = (0, 0, 255)  # Red
        else:
            color = (128, 128, 128)  # Gray

        # Draw lane boundaries with low opacity
        points_left = []
        points_right = []
        points_center = []

        for j in range(lanes.shape[1]):
            if np.sum(np.abs(lanes[i, j, :2])) < 1e-6:
                break

            cx, cy = lanes[i, j, 0], lanes[i, j, 1]
            lx = cx + lanes[i, j, 4]
            ly = cy + lanes[i, j, 5]
            rx = cx + lanes[i, j, 6]
            ry = cy + lanes[i, j, 7]

            points_left.append(world_to_image(lx, ly))
            points_right.append(world_to_image(rx, ry))
            points_center.append(world_to_image(cx, cy))

        if len(points_left) > 1:
            pts_left = np.array(points_left, np.int32)
            pts_right = np.array(points_right, np.int32)
            pts_center = np.array(points_center, np.int32)

            # Draw directly without transparency
            cv2.polylines(img, [pts_left], False, color, 1)
            cv2.polylines(img, [pts_right], False, color, 1)
            cv2.polylines(img, [pts_center], False, color, 1)

    # ==== Route lanes ====
    route_lanes = inputs["route_lanes"][0]
    for i in range(route_lanes.shape[0]):
        traffic_light = route_lanes[i, 0, 8:12]
        if traffic_light[0] == 1:
            color = (0, 255, 0)  # Green
        elif traffic_light[1] == 1:
            color = (0, 255, 255)  # Yellow
        elif traffic_light[2] == 1:
            color = (0, 0, 255)  # Red
        else:
            color = (128, 128, 128)  # Gray

        points_left = []
        points_right = []
        points_center = []

        for j in range(route_lanes.shape[1]):
            if np.sum(np.abs(route_lanes[i, j, :2])) < 1e-6:
                break

            cx, cy = route_lanes[i, j, 0], route_lanes[i, j, 1]
            lx = cx + route_lanes[i, j, 4]
            ly = cy + route_lanes[i, j, 5]
            rx = cx + route_lanes[i, j, 6]
            ry = cy + route_lanes[i, j, 7]

            points_left.append(world_to_image(lx, ly))
            points_right.append(world_to_image(rx, ry))
            points_center.append(world_to_image(cx, cy))

        if len(points_left) > 1:
            pts_left = np.array(points_left, np.int32)
            pts_right = np.array(points_right, np.int32)
            pts_center = np.array(points_center, np.int32)

            cv2.polylines(img, [pts_left], False, color, 2)
            cv2.polylines(img, [pts_right], False, color, 2)
            cv2.polylines(img, [pts_center], False, color, 2)

    # ==== Static objects ====
    static_objects = inputs["static_objects"][0]
    for i in range(static_objects.shape[0]):
        obj = static_objects[i]

        if np.sum(np.abs(obj[:4])) < 1e-6:
            continue

        obj_x, obj_y = obj[0], obj[1]
        obj_heading = np.arctan2(obj[3], obj[2])
        obj_width = obj[4] if obj.shape[0] > 4 else 1.0
        obj_length = obj[5] if obj.shape[0] > 5 else 1.0

        # Set color based on object type
        obj_type = np.argmax(obj[-4:]) if obj.shape[0] >= 10 else 0
        colors = [
            (0, 165, 255),
            (128, 128, 128),
            (0, 255, 255),
            (42, 42, 165),
        ]  # Orange, Gray, Yellow, Brown
        obj_color = colors[obj_type % len(colors)]

        # Draw directly without transparency
        draw_rotated_rect(img, obj_x, obj_y, obj_length, obj_width, obj_heading, obj_color)

    # ==== Neighbor agents ====
    neighbors = inputs["neighbor_agents_past"][0]
    last_timestep = neighbors.shape[1] - 1

    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i, last_timestep]

        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            continue

        n_x, n_y = neighbor[0], neighbor[1]
        n_heading = np.arctan2(neighbor[3], neighbor[2])
        vel_x, vel_y = neighbor[4], neighbor[5]
        len_y, len_x = neighbor[6], neighbor[7]

        # Set color based on vehicle type
        vehicle_type = np.argmax(neighbor[8:11]) if neighbor.shape[0] > 8 else 0
        if vehicle_type == 0:  # Vehicle
            color = (255, 0, 0)  # Blue
        elif vehicle_type == 1:  # Pedestrian
            color = (0, 255, 0)  # Green
        else:  # Bicycle
            color = (255, 0, 255)  # Purple

        # Draw past trajectory
        past_points = []
        for t in range(last_timestep + 1):
            px, py = neighbors[i, t, 0], neighbors[i, t, 1]
            if np.sum(np.abs(neighbors[i, t, :2])) > 1e-6:
                past_points.append(world_to_image(px, py))

        if len(past_points) > 1:
            pts = np.array(past_points, np.int32)
            cv2.polylines(img, [pts], False, color, 1)

        # Draw bounding box
        draw_rotated_rect(img, n_x, n_y, len_x, len_y, n_heading, color, thickness=2)

        # Draw velocity arrow
        if np.sqrt(vel_x**2 + vel_y**2) > 0.1:
            draw_arrow(img, n_x, n_y, n_x + vel_x / 2, n_y + vel_y / 2, (0, 165, 255), 2)

        # Draw future trajectory (simplified - single color)
        if "neighbor_agents_future" in inputs:
            neighbor_future = inputs["neighbor_agents_future"][0][i]
            for j in range(
                0, neighbor_future.shape[0], 10
            ):  # Sample every 10 points for performance
                future_x = neighbor_future[j, 0]
                future_y = neighbor_future[j, 1]
                if future_x == 0 and future_y == 0:
                    break

                pt = world_to_image(future_x, future_y)
                cv2.circle(img, pt, 2, (128, 0, 128), -1)  # Purple color

    # ==== Ego past trajectory ====
    if "ego_agent_past" in inputs:
        ego_past = inputs["ego_agent_past"][0]
        past_points = []
        for i in range(ego_past.shape[0]):
            px, py = ego_past[i, 0], ego_past[i, 1]
            past_points.append(world_to_image(px, py))

        if len(past_points) > 1:
            pts = np.array(past_points, np.int32)
            cv2.polylines(img, [pts], False, (0, 165, 255), 2)  # Orange

    # ==== Ego vehicle ====
    ego_heading = np.arctan2(ego_state[3], ego_state[2])
    car_length = 4.5
    car_width = 2.0

    # Draw ego vehicle as a filled rectangle
    draw_rotated_rect(img, ego_x, ego_y, car_length, car_width, ego_heading, (0, 0, 255))

    # Draw direction arrow
    arrow_end_x = ego_x + car_length * 0.7 * np.cos(ego_heading)
    arrow_end_y = ego_y + car_length * 0.7 * np.sin(ego_heading)
    draw_arrow(img, ego_x, ego_y, arrow_end_x, arrow_end_y, (255, 255, 255), 3)

    # ==== Ego future trajectory ====
    if "ego_agent_future" in inputs:
        ego_future = inputs["ego_agent_future"][0]
        for i in range(0, ego_future.shape[0], 10):  # Sample every 10 points for performance
            future_x = ego_future[i, 0]
            future_y = ego_future[i, 1]

            pt = world_to_image(future_x, future_y)
            cv2.circle(img, pt, 3, (255, 0, 255), -1)  # Magenta color

    # ==== Goal pose ====
    if "goal_pose" in inputs:
        goal_x, goal_y, goal_yaw = inputs["goal_pose"][0]
        goal_end_x = goal_x + 3 * np.cos(goal_yaw)
        goal_end_y = goal_y + 3 * np.sin(goal_yaw)
        draw_arrow(img, goal_x, goal_y, goal_end_x, goal_end_y, (255, 0, 0), 3)

    # ==== Status text ====
    ego_vel_x = ego_state[4]
    ego_vel_y = ego_state[5]
    ego_acc_x = ego_state[6]
    ego_acc_y = ego_state[7]
    ego_steering = ego_state[8]
    ego_yaw_rate = ego_state[9]

    if "turn_indicator" in inputs:
        turn_indicator = inputs["turn_indicator"][0]
        if turn_indicator == 1:
            turn_indicator_text = "None"
        elif turn_indicator == 2:
            turn_indicator_text = "<-"
        elif turn_indicator == 3:
            turn_indicator_text = "->"
        else:
            turn_indicator_text = "Unknown"
    else:
        turn_indicator_text = "N/A"

    # Draw status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    y_offset = 15
    x_pos = 10

    texts = [
        f"Vel X: {ego_vel_x:.2f} m/s",
        f"Vel Y: {ego_vel_y:.2f} m/s",
        f"Acc X: {ego_acc_x:.2f} m/s2",
        f"Acc Y: {ego_acc_y:.2f} m/s2",
        f"Steering: {ego_steering:.2f} rad",
        f"Yaw Rate: {ego_yaw_rate:.2f} rad/s",
        f"Turn: {turn_indicator_text}",
    ]

    for i, text in enumerate(texts):
        cv2.putText(img, text, (x_pos, y_offset + i * 15), font, font_scale, (0, 0, 0), thickness)

    # Return the image array (BGR format)
    return img

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from diffusion_planner.utils.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("args_json", type=Path)
    npz_path = parser.parse_args().npz_path
    args_json = parser.parse_args().args_json

    loaded = np.load(npz_path)
    config_obj = Config(args_json)

    data = {}
    for key, value in loaded.items():
        if key == "map_name" or key == "token":
            continue
        # add batch size axis
        data[key] = torch.tensor(np.expand_dims(value, axis=0))
    data = config_obj.observation_normalizer(data)

    img = visualize_inputs_cv2(data, config_obj.observation_normalizer)

    # Save image instead of displaying due to OpenCV GUI support issue
    output_path = "visualization_output.png"
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")
