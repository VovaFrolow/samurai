from typing import Optional

import torch
from torch import nn
from torch.nn import init
import torchvision

from ocl.modules.attention import Transformer

from ocl.utils import make_build_fn
from ocl.modules.utils import *


@make_build_fn(__name__, "initializer")
def build(config, name: str):
    pass  # No special module building needed


class RandomInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))

    def forward(self, inputs: torch.Tensor):
        noise = torch.randn(inputs.shape[0], self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()


class RandomSMMInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim * 2
        self.mean = nn.Parameter(torch.zeros(1, 1, 2*dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, 2*dim) * initial_std))

    def forward(self, inputs: torch.Tensor):
        noise = torch.randn(inputs.shape[0], self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()

class SMMInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(
        self, 
        n_slots: int, 
        dim: int, 
        hidden_dim: Optional[int] = None, # 128 (original), 256
        initial_std: Optional[float] = None,
        same_output_dim: bool = False,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        # self.nu_min = 3.0
        # self.nu_max = 10.0
        if hidden_dim is None:
            hidden_dim = dim * 4
        else:
            hidden_dim = hidden_dim
        # self.scale = 20
        self.heads = 4 
        self.transf = Transformer(dim, self.heads, dim // self.heads, depth=2)
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))
        init.xavier_uniform_(self.log_std)
        output_dim = dim
        if not same_output_dim:
            output_dim = dim * 2

        # self.nu = nn.Parameter(torch.ones(1, 1, dim) * 3.0)
        
        self.feat_agg = self.mu_proj = nn.Sequential(
            nn.Linear(2*dim, 4*dim),
            nn.LayerNorm(4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.mu_init = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # nn.ReLU(inplace=True)
            nn.Linear(hidden_dim, output_dim)
        )
        self.sigma_init = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # nn.ReLU(inplace=True)
            nn.Linear(hidden_dim, output_dim)
        )
        # self.nu_init = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, dim * 3)
        # )

    def forward(self, inputs: torch.Tensor):
        # ndim = inputs.ndim
        # if ndim == 4:
            # inputs = inputs[:, -1] # .flatten(0, 1)
            # max_inputs, mean_inputs = inputs.amax(dim=1), inputs.mean(dim=1)
            # inp = torch.cat([max_inputs, mean_inputs], dim=-1)
            # inputs = self.feat_agg(inp)
            # inputs = inputs[:, inputs.shape[1] // 2]
        b, n, d, device = *inputs.shape, inputs.device
        n_s = self.n_slots

        mu = self.mean.expand(b, -1, -1)
        logsigma = self.log_std.expand(b, -1, -1)
        # nu = self.nu.expand(b, -1, -1)

        # inputs = torch.cat([mu, logsigma, nu, inputs], dim=1)
        inputs = torch.cat([mu, logsigma, inputs], dim=1)
        inputs = self.transf(inputs)
        mu, logsigma, inputs = torch.split(inputs, [1, 1, n], dim=1)
        # mu, logsigma, nu, inputs = torch.split(inputs, [1, 1, 1, n], dim=1)
        # if ndim == 4:
        #     mu = mu.unflatten(0, (bs, -1))[:, t-1] # .mean(dim=1)
        #     logsigma = logsigma.unflatten(0, (bs, -1))[:, t-1] # .mean(dim=1)
        mu = self.mu_init(mu)
        logsigma = self.sigma_init(logsigma)
        # nu = self.nu_init(nu)

        mu = mu.expand(-1, n_s, -1)
        sigma = logsigma.exp().expand(-1, n_s, -1)
        # nu = nu.expand(-1, n_s, -1)
        # nu = self.nu_min + (self.nu_max - self.nu_min) / (1 + torch.exp(-self.scale * (nu - (self.nu_min + self.nu_max)/2)))

        # pairwise_dist = torch.cdist(mu, mu)
        # mu = mu * (1 + torch.softmax(-pairwise_dist, dim=-1))
        # if torch.any(nu < 3.0) or torch.any(nu > 20):
        #     nu = torch.clamp(nu, min=self.nu_min, max=self.nu_max)
        # slots_init = mu + sigma * torch.randn(mu.shape, device=device)
        slots_init = mu + sigma * torch.randn(mu.shape, device=device)
        # if ndim == 4:
        #     slots_init = slots_init.unflatten(0, (bs, -1)).mean(dim=1)
        
        return slots_init

class VideoSlotInit(nn.Module):
    def __init__(self, n_slots, dim, hidden_dim=None, temp_layers=2, initial_std=None, heads=4):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.heads = heads
        if hidden_dim is None:
            hidden_dim = dim * 4
        else:
            hidden_dim = hidden_dim
        self.temp_agg = Transformer(dim, self.heads, dim // self.heads, depth=temp_layers)
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))
        init.xavier_uniform_(self.log_std)
        # Проекции параметров
        self.mu_proj = nn.Sequential(
            nn.Linear(dim * 3, dim * 6),
            nn.LayerNorm(dim * 6),
            nn.GELU(),
            nn.Linear(dim * 6, dim)
        )
        
        self.sigma_proj = nn.Sequential(
            nn.Linear(dim * 3, dim * 6),
            nn.LayerNorm(dim * 6),
            nn.GELU(),
            nn.Linear(dim * 6, dim),
            nn.Softplus()
        )

        self.mu_init = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 2)
        )
        self.sigma_init = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 2)
        )
        
        # Learnable basis для разнообразия слотов
        self.slot_basis = nn.Parameter(torch.randn(n_slots, dim*2))
        nn.init.orthogonal_(self.slot_basis)
        
    def forward(self, inputs):
        """Инициализация слотов для видео данных
        Args:
            video: [B, T, N, D] (видео последовательность)
        Returns:
            slots: [B, num_slots, dim]
        """
        # video: [B, T, N, D]
        b, t, n, d = inputs.shape
        n_s = self.n_slots
        
        # 1. Временное кодирование
        mu = self.mean.expand(b*t, -1, -1)
        logsigma = self.log_std.expand(b*t, -1, -1)
        inputs = torch.cat([mu, logsigma, inputs.flatten(0, 1)], dim=1)
        inputs = self.temp_agg(inputs)
        mu, logsigma, inputs = torch.split(inputs, [1, 1, n], dim=1)
        # 2. временное агрегирование
        mu, logsigma = mu.unflatten(0, (b, t)), logsigma.unflatten(0, (b, t))
        mean_pool_mu = mu.mean(dim=1)  # [B,N,D]
        max_pool_mu = mu.amax(dim=1)  # [B,N,D]
        min_pool_mu = mu.amin(dim=1)  # [B,N,D]
        mean_pool_logsigma = logsigma.mean(dim=1)  # [B,N,D]
        max_pool_logsigma = logsigma.amax(dim=1)  # [B,N,D]
        min_pool_logsigma = logsigma.amin(dim=1)  # [B,N,D]
        
        # 3. Объединение признаков
        context_mu = torch.cat([min_pool_mu, mean_pool_mu, max_pool_mu], dim=1)  # [B,3N,D]
        context_logsigma = torch.cat([min_pool_logsigma, mean_pool_logsigma, max_pool_logsigma], dim=1)  # [B,3N,D]
        # 4. Генерация параметров
        # print(context_mu.shape, context_logsigma.shape)
        base_mu = self.mu_proj(context_mu.flatten(1, 2))# .unsqueeze(1) # [B,N,D]
        base_logsigma = self.sigma_proj(context_logsigma.flatten(1, 2))#.unsqueeze(1)
        # print(context_mu.shape, context_logsigma.shape, base_mu.shape, base_logsigma.shape, self.dim)
        # base_mu = mean_pool_mu + max_pool_mu
        # base_logsigma = mean_pool_logsigma + max_pool_logsigma
        mu = self.mu_init(base_mu).unsqueeze(1)
        logsigma = self.sigma_init(base_logsigma).unsqueeze(1)
        # print(context_mu.shape, context_logsigma.shape, base_mu.shape, base_logsigma.shape, mu.shape, logsigma.shape, self.dim)
        mu = mu.expand(-1, n_s, -1)
        sigma = logsigma.exp().expand(-1, n_s, -1)
        # 5. Контрастная инициализация
        slots = mu * self.slot_basis.unsqueeze(0)  # [B,K,D]
        
        # 6. Добавление шума с контролируемой дисперсией
        noise = torch.randn(b, self.n_slots, 2*d, device=inputs.device)
        slots = slots + noise * sigma
        
        return slots

class FixedLearnedInit(nn.Module):
    """Learned initialization with a fixed number of slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        if initial_std is None:
            initial_std = dim**-0.5
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * initial_std)

    def forward(self, inputs: torch.Tensor):
        return self.slots.expand(inputs.shape[0], -1, -1)
