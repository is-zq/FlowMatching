import torch
import os
from unet import MNISTUNet
from sampler import EulerSimulator, GaussianSampler, MNISTSampler, CFGVectorFieldODE
from trainer import CFGTrainer, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
import matplotlib.pyplot as plt

MODEL_PATH = "models/mnist_unet.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_trajectory(xs: torch.Tensor, ts: torch.Tensor = None, sample_idx: int = 0, 
                         ncols: int = 10, figsize=(20, 3), save_path: str = None):
    """
    可视化生成过程的轨迹。
    Args:
        xs: 生成的轨迹张量, shape (batch_size, nts, c, h, w)
        ts: 对应的时间步张量, shape (batch_size, nts, 1, 1, 1)，可选
        sample_idx: 可视化哪个样本 (0 <= sample_idx < batch_size)
        ncols: 显示的时间步数量（会均匀抽样）
        figsize: 图像大小
        save_path: 如果提供，则保存到文件
    """
    xs = xs.detach().cpu()
    bs, nts, c, h, w = xs.shape
    assert sample_idx < bs, f"sample_idx {sample_idx} 超出范围 (batch_size={bs})"

    # 均匀抽样 ncols 个时间步
    idxs = torch.linspace(0, nts-1, ncols).long()
    imgs = xs[sample_idx, idxs]  # (ncols, c, h, w)

    # 如果是单通道图像
    if c == 1:
        imgs = imgs.squeeze(1)  # (ncols, h, w)
        cmap = "gray"
    else:
        imgs = imgs.permute(0, 2, 3, 1)  # (ncols, h, w, c)
        cmap = None

    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    for i, ax in enumerate(axes):
        ax.axis("off")
        img = imgs[i]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if ts is not None:
            t_val = ts[0, idxs[i]].item()
            ax.set_title(f"t={t_val:.2f}", fontsize=10)
        else:
            ax.set_title(f"step {idxs[i].item()}", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    plt.show()

# 加载模型或训练模型
if os.path.exists(MODEL_PATH):
    unet = MNISTUNet(
        channels = [32, 64, 128],
        num_res_layer = 2,
        t_emb_dim = 40,
        y_emb_dim = 40,
    ).to(device)
    unet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    unet.eval()
else:
    path = GaussianConditionalProbabilityPath(
        p_init = GaussianSampler([1, 32, 32]),
        p_data = MNISTSampler(),
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)
    unet = MNISTUNet(
        channels = [32, 64, 128],
        num_res_layer = 2,
        t_emb_dim = 40,
        y_emb_dim = 40,
    ).to(device)
    cfg_trainer = CFGTrainer(path = path, model = unet, eta = 0.1)
    cfg_trainer.train(
        num_epochs = 10000,
        device = device,
        lr = 1e-3,
        save = True,
        save_path = MODEL_PATH,
        batch_size = 250
    )

p_init = GaussianSampler([1, 32, 32]).to(device)
simulator = EulerSimulator(ode = CFGVectorFieldODE(model = unet, guidance_scale = 3.0))
batch_size = 10
nts = 100
ts = torch.linspace(0, 1, nts).view(1, -1, 1, 1, 1).expand(batch_size, -1, 1, 1, 1).to(device)

with torch.no_grad():
    for yi in range(11):
        x, _ = p_init.sample(batch_size)
        y = (torch.ones(batch_size, dtype = torch.int64) * yi).to(device)
        xs = simulator.simulate_with_trajectory(x, ts, y = y)
        visualize_trajectory(xs, ts, sample_idx=0, ncols=10)