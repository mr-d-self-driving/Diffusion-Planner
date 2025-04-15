import torch
import torch.nn as nn
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
import json
from mmengine import fileio
import io
import numpy as np


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        ego_current_state,
        neighbor_agents_past,
        static_objects,
        lanes,
        lanes_speed_limit,
        lanes_has_speed_limit,
        route_lanes,
        route_lanes_speed_limit,
        route_lanes_has_speed_limit,
    ):
        inputs = {
            'ego_current_state': ego_current_state,
            'neighbor_agents_past': neighbor_agents_past,
            'static_objects': static_objects,
            'lanes': lanes,
            'lanes_speed_limit': lanes_speed_limit,
            'lanes_has_speed_limit': lanes_has_speed_limit,
            'route_lanes': route_lanes,
            'route_lanes_speed_limit': route_lanes_speed_limit,
            'route_lanes_has_speed_limit': route_lanes_has_speed_limit,
        }
        encoder_outputs, decoder_outputs = self.model(inputs)
        return decoder_outputs  # or both if needed


# Load config
config_json_path = "args.json"
with open(config_json_path, "r") as f:
    config_json = json.load(f)
config_obj = Config(config_json_path)

# Init model
model = Diffusion_Planner(config_obj)
model.eval().cuda()
model.decoder.decoder.training = False

ckpt_path = "latest.pth"
ckpt = fileio.get(ckpt_path)
with io.BytesIO(ckpt) as f:
    ckpt = torch.load(f)
state_dict = ckpt["model"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)


# # Dummy inputs
dummy_inputs = {
    'ego_current_state': torch.tensor(np.random.randn(10).astype(np.float32)).cuda(),
    'neighbor_agents_past': torch.tensor(np.random.randn(32, 21, 11).astype(np.float32)).cuda(),
    'static_objects': torch.tensor(np.random.randn(5, 10).astype(np.float32)).cuda(),
    'lanes': torch.tensor(np.random.randn(70, 20, 12).astype(np.float32)).cuda(),
    'lanes_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.float32)).cuda(),
    'lanes_has_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.float32)).cuda(),
    'route_lanes': torch.tensor(np.random.randn(25, 20, 12).astype(np.float32)).cuda(),
    'route_lanes_speed_limit': torch.tensor(np.random.randn(25, 1).astype(np.float32)).cuda(),
    'route_lanes_has_speed_limit': torch.tensor(np.random.randn(25, 1).astype(np.float32)).cuda(),
}

sample_input = np.load("sample_input.npz")

# dev = model.parameters().__next__().device
for key in sample_input:
    print("key", key)
    print(sample_input[key].shape)
    if (key == "map_name" or key == "token" or key == "ego_agent_future" or key == "neighbor_agents_future"):
        continue
    if isinstance(sample_input[key], np.ndarray):
        dummy_inputs[key] = torch.tensor(sample_input[key]).cuda()
        dummy_inputs[key] = dummy_inputs[key].unsqueeze(0)

        print("dummy_inputs[key]\n", dummy_inputs[key])
dummy_inputs = config_obj.observation_normalizer(dummy_inputs)

# neighbors = dummy_inputs['neighbor_agents_past']
# print("DEBUG type of neighbors:", type(neighbors))

input_names = [
    'ego_current_state', 'neighbor_agents_past',
    'static_objects', 'lanes', 'lanes_speed_limit', 'lanes_has_speed_limit',
    'route_lanes', 'route_lanes_speed_limit', 'route_lanes_has_speed_limit'
]

wrapper = ONNXWrapper(model).eval().cuda()

model(dummy_inputs)
# Export
# torch.onnx.export(
#     model,
#     tuple(dummy_inputs),
#     "diffusion_planner.onnx",
#     input_names=input_names,
#     output_names=["output"],
#     dynamic_axes=None,
#     opset_version=17,
# )

# ort_session = ort.InferenceSession(
#     "model.onnx", providers=['CPUExecutionProvider'])

# # Create NumPy input matching the shape of your dummy_inputs
# inputs = {
#     "ego_current_state": np.random.randn(10).astype(np.float32),
#     "ego_agent_future": np.random.randn(80, 3).astype(np.float32),
#     "neighbor_agents_past": np.random.randn(32, 21, 11).astype(np.float32),
#     "neighbor_agents_future": np.random.randn(32, 80, 3).astype(np.float32),
#     "static_objects": np.random.randn(5, 10).astype(np.float32),
#     "lanes": np.random.randn(70, 20, 12).astype(np.float32),
#     "lanes_speed_limit": np.random.randn(70, 1).astype(np.float32),
#     "lanes_has_speed_limit": np.random.randn(70, 1).astype(np.float32),
#     "route_lanes": np.random.randn(25, 20, 12).astype(np.float32),
#     "route_lanes_speed_limit": np.random.randn(25, 1).astype(np.float32),
#     "route_lanes_has_speed_limit": np.random.randn(25, 1).astype(np.float32),
# }

# outputs = ort_session.run(None, inputs)
# print(outputs[0])  # Or whatever your output is
