from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from unet import MNISTUNet
from tqdm import tqdm
from sampler import Sampleable

SAVE_PATH = "./models/default.pth"

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, save: bool = False, save_path: str = SAVE_PATH,**kwargs) -> torch.Tensor:
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            
        if save:
            torch.save(self.model.state_dict(), save_path)

        # Finish
        self.model.eval()

class ConditionalProbabilityPath(nn.Module, ABC):
    def __init__(self, p_init: Sampleable,p_data: Sampleable) -> None:
        super().__init__()
        self.p_init = p_init
        self.p_data = p_data
    
    @abstractmethod
    def sample_conditional_variable(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        采样条件变量z和标签y
        Args:
            - num_samples: 样本数量
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples,)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        从条件概率路径p_t(x|z)中采样
        Args:
            - z: (num_samples, c, h, w)
            - t: (num_samples, 1, 1, 1)
        Returns:
            - x: (num_samples, c, h, w)
        """
        pass
    
    @abstractmethod
    def conditional_vector_filed(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        条件向量场u_t(x|z)
        Args:
            - x: (num_samples, c, h, w)
            - z: (num_samples, c, h, w)
            - t: (num_samples, 1, 1, 1)
        Returns:
            - x: (num_samples, c, h, w)
        """
        pass

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)

class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)

class LinearAlpha(Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

class LinearBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1-t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return - torch.ones_like(t)

class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_init: Sampleable, p_data: Sampleable, alpha: Alpha, beta: Beta) -> None:
        super().__init__(p_init, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditional_variable(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
    
    def conditional_vector_filed(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)
        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: MNISTUNet, eta: float) -> None:
        assert eta > 0 and eta < 1
        super().__init__(model)
        self.path = path
        self.eta = eta
        
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, y = self.path.sample_conditional_variable(batch_size)
        t = torch.rand(batch_size, 1, 1, 1).to(z.device)
        x = self.path.sample_conditional_path(z, t)
        mask = torch.rand(batch_size) < self.eta
        y[mask] = 10
        u_theta = self.model(x, t, y)
        u_ref = self.path.conditional_vector_filed(x, z, t)
        return torch.square(u_theta - u_ref).sum(dim=(1,2,3)).mean()
