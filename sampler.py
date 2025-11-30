from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms

# 可采样抽象类
class Sampleable(ABC):
    """
        采样，返回[样本，标签]元组
        Args:
            - num_samples: 样本数量
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
    """
    @abstractmethod
    def sample(self, num_samples:int) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass

class GaussianSampler(nn.Module, Sampleable):
    def __init__(self, shape: list[int], std: float = 1.0) -> None:
        super().__init__()
        self.shape = shape  # 样本形状 (不包含batch_size)
        self.std = std  # 分布的标准差
        self.dummy = nn.Buffer(torch.zeros(1))  # 标识当前处于什么设备
        
    # 从高斯分布中采样噪声，标签为空
    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None
    
class MNISTSampler(nn.Module, Sampleable):
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            - num_samples: 样本数
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size,)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels

class ODE(ABC):
    """
    偏移系数, 即向量场ut(x)
        Args:
            - xt: 当前位置, shape(bs, c, h, w)
            - t: 当前时间, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
    """
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kargs):
        pass

class SDE(ABC):
    """
    偏移系数, 即向量场ut(x)
        Args:
            - xt: 当前位置, shape(bs, c, h, w)
            - t: 当前时间, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
    """
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kargs):
        pass

    """
    扩散系数, 即σt
        Args:
            - t: 当前时间, shape (bs, 1)
        Returns:
            - diffusion_coefficient: float
    """
    @abstractmethod
    def diffusion_coefficient(self, t: torch.float, **kargs):
        pass

class Simulator(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def step(self, xt:torch.Tensor, t:torch.Tensor, dt:torch.Tensor, **kargs) -> torch.Tensor:
        """
        采样更新一步
        Args:
            - xt: 当前位置, shape (bs, c, h, w)
            - t: 当前时间, shape (bs, 1, 1, 1)
            - dt: 时间跨度, shape (bs, 1, 1, 1)
        Returns:
            - nxt: 在 t+dt 时刻的位置 (bs, c, h, w)
        """
        pass
    
    @torch.no_grad()
    def simulate(self, x_init: torch.Tensor, ts: torch.Tensor, **kargs) -> torch.Tensor:
        """
        模拟从x_init按照ts走到x_final, 返回最终结果
        Args:
            - x_init: 初始位置, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: 最终位置, shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts-1)):
            t = ts[:, t_idx]    # shape (bs,1,1,1)
            h = ts[:, t_idx+1] - t
            x = self.step(x,t,h, **kargs)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kargs):
        """
        模拟从x_init按照ts走到x_final, 并返回每一步的结果
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

class EulerSimulator(Simulator):
    def __init__(self, ode:ODE) -> None:
        super().__init__()
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kargs) -> torch.Tensor:
        return xt + self.ode.drift_coefficient(xt, t, **kargs) * dt
    
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde:SDE) -> None:
        super().__init__()
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kargs) -> torch.Tensor:
        one = torch.ones_like(t)
        sigma = one - t
        return xt + self.sde.drift_coefficient(xt, t, **kargs) * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)

class GuidedVectorField(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass

class CFGVectorFieldODE(ODE):
    def __init__(self, model: GuidedVectorField, guidance_scale: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        return (1 - self.guidance_scale) * self.model(xt, t, torch.full_like(y, 10)) + self.guidance_scale * self.model(xt, t, y)