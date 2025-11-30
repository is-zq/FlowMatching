import math
import torch
import torch.nn as nn
from sampler import GuidedVectorField

class tEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
    
class ResidualLayer(nn.Module):
    def __init__(self, channel: int, t_emb_dim: int, y_emb_dim: int) -> None:
        super().__init__()
        # 预激活，即激活函数和BatchNorm放前面
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, channel)
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, channel)
        )
    
    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor, y_embedding: torch.Tensor) -> torch.Tensor:
        res = x.clone()
        x = self.block1(x)
        x = x + self.t_mlp(t_embedding).unsqueeze(-1).unsqueeze(-1) + self.y_mlp(y_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.block2(x)
        return x + res
    
class Encoder(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channel_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor, y_embedding: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embedding, y_embedding)
        return self.downsample(x)
        
class Midcoder(nn.Module):
    def __init__(self, channel: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channel, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        
    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor, y_embedding: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embedding, y_embedding)
        return x
    
class Decoder(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channel_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        
    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor, y_embedding: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for block in self.res_blocks:
            x = block(x, t_embedding, y_embedding)
        return x

class MNISTUNet(GuidedVectorField):
    def __init__(self, channels: list[int], num_res_layer: int, t_emb_dim: int, y_emb_dim: int) -> None:
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.t_embedder = tEmbedder(t_emb_dim)
        # y embedding, [0, 1, ..., 9, 10], 10表示空标签
        self.y_embedder = nn.Embedding(11, y_emb_dim)
        encoders = []
        decoders = []
        for (cur_c, nex_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(cur_c, nex_c, num_res_layer, t_emb_dim, y_emb_dim))
            decoders.append(Decoder(nex_c, cur_c, num_res_layer, t_emb_dim, y_emb_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
        self.midcoder = Midcoder(channels[-1], num_res_layer, t_emb_dim, y_emb_dim)
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> None:
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        x = self.init_conv(x)
        t_embedding = self.t_embedder(t)
        y_embedding = self.y_embedder(y)
        residual = []
        for encoder in self.encoders:
            x = encoder(x, t_embedding, y_embedding)
            residual.append(x.clone())
        self.midcoder(x, t_embedding, y_embedding)
        for idx, decoder in enumerate(self.decoders):
            x += residual.pop()
            x = decoder(x, t_embedding, y_embedding)
        return self.final_conv(x)
