from cnn import PolicyNetwork, ValueNetwork
import torch

policy_net = PolicyNetwork()
policy_net.load_state_dict(torch.load("./saves/setting_1/policy_net_130.pth"))
policy_tensor = torch.randn(1, 14, 19, 19)
policy_output = policy_net(policy_tensor)
torch.onnx.export(policy_net, policy_tensor, "policy_output.onnx", export_params=True, opset_version=15, input_names=['board_tensor'], output_names=['output'])

value_net = ValueNetwork()
value_net.load_state_dict(torch.load("./saves/setting_1/value_net_130.pth"))
value_tensor = torch.randn(1, 13, 19, 19)
value_output = value_net(value_tensor)
torch.onnx.export(value_net, value_tensor, "value_output.onnx", export_params=True, opset_version=15, input_names=['board_tensor'], output_names=['output'])
