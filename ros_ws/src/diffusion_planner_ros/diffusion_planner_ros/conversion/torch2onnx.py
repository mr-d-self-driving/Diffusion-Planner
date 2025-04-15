import copy
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
    ):
        inputs = {
            'ego_current_state': ego_current_state,
            'neighbor_agents_past': neighbor_agents_past,
            'static_objects': static_objects,
            'lanes': lanes,
            'lanes_speed_limit': lanes_speed_limit,
            'lanes_has_speed_limit': lanes_has_speed_limit,
            'route_lanes': route_lanes,
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
model.eval()
model.encoder.encoder.eval()
model.decoder.decoder.eval()
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
    'ego_current_state': torch.tensor(np.random.randn(10).astype(np.float32)).unsqueeze(0),
    'neighbor_agents_past': torch.tensor(np.random.randn(32, 21, 11).astype(np.float32)).unsqueeze(0),
    'static_objects': torch.tensor(np.random.randn(5, 10).astype(np.float32)).unsqueeze(0),
    'lanes': torch.tensor(np.random.randn(70, 20, 12).astype(np.float32)).unsqueeze(0),
    'lanes_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.float32)).unsqueeze(0),
    'lanes_has_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.bool_)).unsqueeze(0),
    'route_lanes': torch.tensor(np.random.randn(25, 20, 12).astype(np.float32)).unsqueeze(0),
}


dummy_inputs = config_obj.observation_normalizer(dummy_inputs)
torch_inputs = copy.deepcopy(dummy_inputs)
onnx_inputs = copy.deepcopy(dummy_inputs)


input_names = [
    'ego_current_state', 'neighbor_agents_past',
    'static_objects', 'lanes', 'lanes_speed_limit', 'lanes_has_speed_limit',
    'route_lanes'
]

wrapper = ONNXWrapper(model).eval()

# Export
input_tuple = (
    torch_inputs['ego_current_state'],
    torch_inputs['neighbor_agents_past'],
    torch_inputs['static_objects'],
    torch_inputs['lanes'],
    torch_inputs['lanes_speed_limit'],
    torch_inputs['lanes_has_speed_limit'],
    torch_inputs['route_lanes'],
)

with torch.no_grad():
    output = wrapper(*input_tuple)
    for key in output.keys():
        print(f"Output {key} shape:", output[key].shape)
        torch_out = output[key].cpu().numpy()

onnx_model = torch.onnx.export(
    wrapper,
    input_tuple,
    "model.onnx",
    input_names=input_names,
    output_names=["output"],
    # dynamic_axes=None,
    dynamic_axes={name: {0: 'batch'}
                  for name in input_names},  # optional, but useful
    opset_version=20,
)


ort_session = ort.InferenceSession(
    "model.onnx", providers=['CPUExecutionProvider'])

onnx_model = onnx.load("model.onnx")
print("Inputs:")
for input in onnx_model.graph.input:
    print(input.name)
print("\nOutputs:")
for output in onnx_model.graph.output:
    print(output.name)

inputs = {
    'ego_current_state': onnx_inputs['ego_current_state'].cpu().numpy(),
    'neighbor_agents_past': onnx_inputs['neighbor_agents_past'].cpu().numpy(),
    'static_objects': onnx_inputs['static_objects'].cpu().numpy(),
    'lanes': onnx_inputs['lanes'].cpu().numpy(),
    'lanes_speed_limit': onnx_inputs['lanes_speed_limit'].cpu().numpy(),
    'lanes_has_speed_limit': onnx_inputs['lanes_has_speed_limit'].cpu().numpy(),
    'route_lanes': onnx_inputs['route_lanes'].cpu().numpy(),
}
inputs['lanes_has_speed_limit'] = inputs['lanes_has_speed_limit'].astype(
    np.bool_)


print("ONNX model input names:")
for i in ort_session.get_inputs():
    print(i.name, i.shape)

onnx_output = ort_session.run(None, inputs)
print(f"torch output, with shape {torch_out.shape}:")
# print(torch_out)
print(f"onnx output, with shape {onnx_output[0].shape}:")
# print(onnx_output[0])

abs_diff = np.abs(torch_out - onnx_output[0])
print(f"Max diff: {abs_diff.max()}")
print(f"Mean diff: {abs_diff.mean()}")
print(
    f"Close? {np.allclose(torch_out, onnx_output[0], rtol=1e-03, atol=1e-05)}")


for input in ort_session.get_inputs():
    print(f"{input.name}: {input.type}, {input.shape}")

print({k: v.dtype for k, v in dummy_inputs.items()})
