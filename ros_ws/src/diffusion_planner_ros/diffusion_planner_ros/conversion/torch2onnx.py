import torch
import torch.nn as nn
import onnx
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
import json
from mmengine import fileio
import io
import numpy as np
import onnxruntime as ort
torch.backends.mha.set_fastpath_enabled(False)


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
        return decoder_outputs


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
# model = torch.compile(model)

# # Dummy inputs
dummy_inputs = {
    'ego_current_state': torch.tensor(np.random.randn(10).astype(np.float32)).cuda().unsqueeze(0),
    'neighbor_agents_past': torch.tensor(np.random.randn(32, 21, 11).astype(np.float32)).cuda().unsqueeze(0),
    'static_objects': torch.tensor(np.random.randn(5, 10).astype(np.float32)).cuda().unsqueeze(0),
    'lanes': torch.tensor(np.random.randn(70, 20, 12).astype(np.float32)).cuda().unsqueeze(0),
    'lanes_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.float32)).cuda().unsqueeze(0),
    'lanes_has_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.bool_)).cuda().unsqueeze(0),
    'route_lanes': torch.tensor(np.random.randn(25, 20, 12).astype(np.float32)).cuda().unsqueeze(0),
    'route_lanes_speed_limit': torch.tensor(np.random.randn(25, 1).astype(np.float32)).cuda().unsqueeze(0),
    'route_lanes_has_speed_limit': torch.tensor(np.random.randn(25, 1).astype(np.float32)).cuda().unsqueeze(0),
}

sample_input = np.load("sample_input.npz")

# dev = model.parameters().__next__().device
for key in sample_input:
    if key in ['map_name', 'token', 'ego_agent_future', 'neighbor_agents_future']:
        continue
    val = sample_input[key]
    if isinstance(val, np.ndarray):
        if val.dtype.kind in {'U', 'S'}:  # Unicode or string
            continue
        dummy_inputs[key] = torch.tensor(val).cuda().unsqueeze(
            0) if val.ndim > 0 else torch.tensor([val]).cuda()


dummy_inputs = config_obj.observation_normalizer(dummy_inputs)

# neighbors = dummy_inputs['neighbor_agents_past']
# print("DEBUG type of neighbors:", type(neighbors))

input_names = [
    'ego_current_state', 'neighbor_agents_past',
    'static_objects', 'lanes', 'lanes_speed_limit', 'lanes_has_speed_limit',
    'route_lanes', 'route_lanes_speed_limit', 'route_lanes_has_speed_limit'
]

wrapper = ONNXWrapper(model).eval().cuda()

# Export
input_tuple = (
    dummy_inputs['ego_current_state'],
    dummy_inputs['neighbor_agents_past'],
    dummy_inputs['static_objects'],
    dummy_inputs['lanes'],
    dummy_inputs['lanes_speed_limit'],
    dummy_inputs['lanes_has_speed_limit'],
    dummy_inputs['route_lanes'],
    dummy_inputs['route_lanes_speed_limit'],
    dummy_inputs['route_lanes_has_speed_limit'],
)
with torch.no_grad():
    output = wrapper(*input_tuple)
    for key in output.keys():
        print(f"Output {key} shape:", output[key].shape)
        torch_out = output[key].cpu().numpy()

# onnx_model = torch.onnx.export(
#     wrapper,
#     input_tuple,
#     "model.onnx",
#     input_names=input_names,
#     output_names=["output"],
#     # dynamic_axes=None,
#     dynamic_axes={name: {0: 'batch'}
#                   for name in input_names},  # optional, but useful
#     opset_version=17,

# )


ort_session = ort.InferenceSession(
    "model.onnx", providers=['CPUExecutionProvider'])

onnx_model = onnx.load("model.onnx")
print("Inputs:")
for input in onnx_model.graph.input:
    print(input.name)
print("\nOutputs:")
for output in onnx_model.graph.output:
    print(output.name)

# # Create NumPy input matching the shape of your dummy_inputs
# inputs = {
#     'ego_current_state': np.random.randn(1, 10).astype(np.float32),
#     'neighbor_agents_past': np.random.randn(1, 32, 21, 11).astype(np.float32),
#     'static_objects': np.random.randn(1, 5, 10).astype(np.float32),
#     'lanes': np.random.randn(1, 70, 20, 12).astype(np.float32),
#     'lanes_speed_limit': np.random.randn(1, 70, 1).astype(np.float32),
#     'lanes_has_speed_limit': np.random.randn(1, 70, 1).astype(np.bool_),
#     'route_lanes': np.random.randn(1, 25, 20, 12).astype(np.float32),
# }

inputs = {
    'ego_current_state': dummy_inputs['ego_current_state'].cpu().numpy(),
    'neighbor_agents_past': dummy_inputs['neighbor_agents_past'].cpu().numpy(),
    'static_objects': dummy_inputs['static_objects'].cpu().numpy(),
    'lanes': dummy_inputs['lanes'].cpu().numpy(),
    'lanes_speed_limit': dummy_inputs['lanes_speed_limit'].cpu().numpy(),
    'lanes_has_speed_limit': dummy_inputs['lanes_has_speed_limit'].cpu().numpy(),
    'route_lanes': dummy_inputs['route_lanes'].cpu().numpy(),
}

print("ONNX model input names:")
for i in ort_session.get_inputs():
    print(i.name, i.shape)

outputs = ort_session.run(None, inputs)
print(f"torch output, with shape {torch_out.shape}:")
print(torch_out)
print(f"onnx output, with shape {outputs[0].shape}:")
print(outputs)
