# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from build_model import build_model


# Grad-CAM用フック保存用
class GradCAM:
	def __init__(self, model: torch.nn.Module, target_layer: str = 'layer4'):
		self.model = model
		self.model.eval()
		self.target_layer = dict([*model.named_modules()])[target_layer]
		self.gradients = None
		self.activations = None

		# hookを登録
		self.target_layer.register_forward_hook(self._save_activation)
		self.target_layer.register_full_backward_hook(self._save_gradient)

	def _save_activation(self, module, input, output):
		self.activations = output.detach()

	def _save_gradient(self, module, grad_input, grad_output):
		self.gradients = grad_output[0].detach()

	def __call__(self, x: torch.Tensor, class_idx: int | None = None):
		x = x.requires_grad_()
		output = self.model(x)

		if class_idx is None:
			class_idx = int(torch.sigmoid(output).item() > 0.5)

		# バックプロパゲーション（targetクラスのみ）
		self.model.zero_grad()
		output[0, 0].backward(retain_graph=True)

		# 勾配と特徴マップを取得
		grads = self.gradients  # (1, C, H, W)
		acts = self.activations  # (1, C, H, W)
		weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over H,W

		# 重み付き特徴マップの和
		cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
		cam = F.relu(cam)

		# 正規化して返す
		cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
		cam -= cam.min()
		cam /= cam.max() + 1e-8  # avoid division by zero
		return cam, torch.sigmoid(output).item()


def visualize_gradcam(
	input_tensor: torch.Tensor, model: torch.nn.Module, device: str = 'cpu', name=False
):
	gradcam = GradCAM(model.to(device))
	input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)
	cam, prob = gradcam(input_tensor)

	cam_np = cam.squeeze().cpu().numpy()
	input_np = input_tensor.squeeze().detach().cpu().numpy()
	print(cam_np)
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.title(f'Input (Pred prob={prob:.2f})')
	plt.imshow(input_np, aspect=2, cmap='seismic', origin='lower', vmin=-2, vmax=2)
	plt.tight_layout()
	plt.subplot(1, 2, 2)
	plt.title('Grad-CAM')
	plt.imshow(cam_np, aspect=2, cmap='jet', origin='lower')
	# plt.colorbar()
	if name:
		plt.suptitle(f'{name}')
	plt.tight_layout()
	if name:
		plt.savefig(f'{name}.png', dpi=300)


data_dir = '/workspace/data'
event_file = data_dir + '/extract_event_2s.npy'
noise_file = data_dir + '/extract_noise_2s.npy'
time_file = data_dir + '/extract_event_time.npy'
event_data = np.load(event_file)
noise_data = np.load(noise_file)
times = np.load(time_file, allow_pickle=True)
# scaling (trace normalization)
event_data = (event_data - np.mean(event_data, axis=2, keepdims=True)) / np.std(
	event_data, axis=2, keepdims=True
)
noise_data = (noise_data - np.mean(noise_data, axis=2, keepdims=True)) / np.std(
	noise_data, axis=2, keepdims=True
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model().to(device)

model_dir = Path('/workspace/output/train')
model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))


for i in range(10):
	event_sample = torch.tensor(event_data[i], dtype=torch.float32).unsqueeze(0)
	# (1, 1, H, W)
	time = times[i].strftime('%Y-%m-%d %H-%M-%S')
	visualize_gradcam(event_sample, model, device=device, name=time)


for i in range(10):
	noise_sample = torch.tensor(noise_data[i], dtype=torch.float32).unsqueeze(0)
	# (1, 1, H, W)
	time = times[i].strftime('%Y-%m-%d %H-%M-%S')
	visualize_gradcam(noise_sample, model, device=device)

# %%
