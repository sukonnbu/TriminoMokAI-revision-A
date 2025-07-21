import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from cnn import PolicyNetwork, ValueNetwork


def visualize_pytorch_kernels(model, layer_name=None, layer_index=0, max_filters=16):
    """
    PyTorch 모델의 커널 가중치 시각화

    Args:
        model: 훈련된 PyTorch 모델
        layer_name: 시각화할 레이어 이름
        layer_index: 레이어 인덱스
        max_filters: 시각화할 최대 필터 수
    """
    # 레이어 선택
    if layer_name:
        layer = dict(model.named_modules())[layer_name]
    else:
        conv_layers = [module for name, module in model.named_modules()
                       if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            print("합성곱 레이어를 찾을 수 없습니다.")
            return
        layer = conv_layers[layer_index]

    # 가중치 추출
    weights = layer.weight.data.cpu().numpy()  # [output_channels, input_channels, height, width]

    print(f"가중치 형태: {weights.shape}")

    # 필터 수 제한
    num_filters = min(weights.shape[0], max_filters)

    # 시각화
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('PyTorch CNN 커널 가중치', fontsize=16)

    for i in range(16):
        row, col = i // 4, i % 4

        if i < num_filters:
            # 첫 번째 입력 채널의 가중치만 시각화
            if weights.shape[1] == 1:
                kernel = weights[i, 0, :, :]
            else:
                kernel = np.mean(weights[i, :, :, :], axis=0)

            im = axes[row, col].imshow(kernel, cmap='RdBu',
                                       vmin=-np.abs(kernel).max(),
                                       vmax=np.abs(kernel).max())
            axes[row, col].set_title(f'Filter {i + 1}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        else:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    policy_net.load_state_dict(torch.load('policy_net.pth', map_location=torch.device('cpu')))
    value_net.load_state_dict(torch.load('value_net.pth', map_location=torch.device('cpu')))

    print("PolicyNet, Conv1")
    visualize_pytorch_kernels(policy_net, layer_index=0)
    print("PolicyNet, Conv2")
    visualize_pytorch_kernels(policy_net, layer_index=1)

    print("ValueNet, Conv1")
    visualize_pytorch_kernels(value_net, layer_index=0)
    print("ValueNet, Conv2")
    visualize_pytorch_kernels(value_net, layer_index=1)
