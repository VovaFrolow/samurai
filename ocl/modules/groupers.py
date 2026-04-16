from typing import Optional, Tuple

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torch.distributions as dist

import torchvision

from ocl.modules.attention import Transformer
from ocl.modules import networks
from ocl.modules.utils import *
from ocl.utils import *


@make_build_fn(__name__, "grouper")
def build(config, name: str):
    pass  # No special module building needed


class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = networks.MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        # if n_iters is None:
        #     n_iters = self.n_iters

        for _ in range(self.n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)

        return {"slots": slots, "masks": pre_norm_attn}


class SlotMixture(nn.Module):
    """
    Slot Mixture module
    """

    def __init__(
        self, 
        num_slots: int, 
        dim: int,
        iters: int = 3, 
        eps: int = 1e-10, 
        hidden_dim: Optional[int] = None,
        use_mlp: bool = True
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        if hidden_dim is None:
            hidden_dim = dim * 4
        else:
            hidden_dim = hidden_dim

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru_mu = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.dim = dim
        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim * 2)
        self.norm_mu = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)
        if use_mlp:
            self.mlp_out = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 2, dim)
            )
        else:
            self.mlp_out = None

    def step(self, slots, k, v, b, n, d, pi_cl):
        slots = self.norm_slots(slots)
        slots_mu, slots_logsigma = slots.split(self.dim, dim=-1)
        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)

        # E step
        dots = ((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) ** 2 / torch.unsqueeze(torch.exp(q_logsigma) ** 2,
                                                                                          2)).sum(dim=-1) * self.scale
        dots_exp = (torch.exp(-dots) + self.eps) * pi_cl
        attn = dots_exp / dots_exp.sum(dim=1, keepdim=True)  # gammas
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # M step for mus
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)

        # M step for prior probs of each gaussian
        pi_cl_new = attn.sum(dim=-1, keepdim=True)
        pi_cl_new = pi_cl_new / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)

        # NN update for mus
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        updates_mu = updates_mu.reshape(b, -1, d)
        updates_mu = updates_mu + self.mlp_mu(self.norm_mu(updates_mu))
        # if torch.isnan(updates_mu).any():
        #     print('updates_mu Nan appeared')

        # M step for logsigmas for new mus
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', (
            (torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2)) ** 2 + self.eps, attn)))
        # if torch.isnan(updates_logsigma).any():
        #     print('updates_logsigma Nan appeared')

        # new gaussians params
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)

        log_likelihood = torch.tensor(0, device=slots.device)

        return slots, pi_cl_new, -log_likelihood, attn

    def forward(self, slots: torch.Tensor, inputs: torch.Tensor, n_iters: Optional[int] = None):
        b, n, d, device = *inputs.shape, inputs.device
        
        n_s = self.num_slots

        pi_cl = (torch.ones(b, n_s, 1) / n_s).to(device)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        if n_iters is None:
            n_iters = self.iters
        
        for _ in range(n_iters):
            slots, pi_cl, log_dict, attn = self.step(slots, k, v, b, n, d, pi_cl)
        slots, pi_cl, log_dict, attn = self.step(slots.detach(), k, v, b, n, d, pi_cl)
        
        if self.mlp_out is not None:
            slots = self.mlp_out(slots)

        return {"slots": slots, "masks": attn}


class StabilizedSMM(nn.Module):
    """
    Stabilized Slot Mixture Module
    """

    def __init__(
        self, 
        num_slots: int, 
        dim: int,
        step: int,
        decay_steps: Optional[int],
        temperature: float,
        final_temperature: Optional[float],
        used_multiples: Optional[set], 
        attn_smooth: Optional[str] = None, 
        init_temperature: float = 1.0,
        iters: int = 3, 
        eps: int = 1e-10, 
        hidden_dim: Optional[int] = None, # 128 (original), 256
        gau_min: float = 0.1,
        gau_max: float = 2.0, 
        attn_smooth_size: int = 5,
        drop_rate: float = 0.2,
        # use_mlp: bool = True
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.current_step = step
        self.eps = eps
        self.nu_min = 3.0
        self.nu_max = 20.0
        self.scale = dim ** -0.5
        if hidden_dim is None:
            hidden_dim = dim * 4
        else:
            hidden_dim = hidden_dim

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # self.nu = nn.Parameter(torch.ones(batch_size, num_slots, 1, 1) * 3.0)

        self.slot_attn_smooth = attn_smooth
        self.temperature = temperature

        if self.slot_attn_smooth != None:
            kernel_size = attn_smooth_size
            if self.slot_attn_smooth.lower() == 'gaussian':
                sigma_min = gau_min
                sigma_max = gau_max
                self.knconv = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, 
                                                               sigma=(sigma_min, sigma_max))
            elif self.slot_attn_smooth.lower() == 'conv':
                self.knconv = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=kernel_size,
                                     padding=kernel_size // 2,
                                     bias=False)
            elif self.slot_attn_smooth.lower() == 'wnconv':
                self.knconv = WNConv(in_channels=1,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2,
                                    bias=False)

        self.gru_mu = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.dim = dim
        # self.mlp_mu = networks.MLP(dim, dim, [hidden_dim], initial_layer_norm=True, residual=True)
        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim * 2)
        self.norm_mu = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)
        # if use_mlp:
        self.mlp_out = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, dim)
        )
        # else:
        #     self.mlp_out = None
        # self.mlp_out = networks.MLP(2*dim, dim, [2*hidden_dim], initial_layer_norm=False, residual=False)
        if final_temperature is not None:
            self.init_temperature = init_temperature
            self.final_temperature = final_temperature
            self.decay_steps = decay_steps
            self.used_multiples = used_multiples
        self.drop_rate = drop_rate
        if drop_rate is not None:
            self.dropout = nn.Dropout(drop_rate)
    
    # def compute_mahalanobis_distance(self, x, mu, sigma):
    #     """
    #     Вычисление расстояния Махаланобиса с анизотропной ковариацией
    #     x: [batch, num_slots, num_features]
    #     mu: [batch, num_slots, num_features] 
    #     sigma: [batch, num_slots, num_features]
    #     """
    #     diff = x - mu  # Разница между входными данными и mu
        
    #     # Вычисление квадратичной формы Махаланобиса
    #     mahalanobis = torch.sum(diff**2 / (torch.exp(sigma) + self.eps), dim=-1)
    
    #     return mahalanobis * self.scale

    # def compute_energy_regularization(self, slots):
    #     mu, sigma = slots.chunk(2, dim=-1)  # Предполагаем, что slots имеют размерность (b, k, d)
        
    #     # Вычисление попарных расстояний между слотами
    #     # mu имеет размерность (64, 7, 512)
    #     # Используем cdist для вычисления расстояний между слотами
    #     slot_distances = torch.cdist(mu, mu)  # Размерность (64, 7, 7)

    #     # Применяем функцию для получения энергии
    #     energy = torch.mean(torch.exp(-slot_distances), dim=[1, 2])  # Суммируем по слотам
        
    #     # Нормализованный энергетический штраф
    #     energy_reg = torch.sigmoid(energy)
    #     return energy_reg.unsqueeze(1).unsqueeze(2)
    
    # def log_t_pdf(self, z, sigma, nu, d):
    #     log_prob = (
    #         torch.lgamma((nu + d) / 2) - 
    #         torch.lgamma(nu / 2) - 
    #         0.5 * torch.log(torch.pi * nu * sigma**2) - 
    #         ((nu + d) / 2) * torch.log(1 + z**2 / nu)
    #     )
    #     return log_prob

    def step(self, slots, k, v, b, n, d, pi_cl, last=False):
        slots = self.norm_slots(slots)
        slots_mu, slots_logsigma = slots.split(self.dim, dim=-1)
        # slots_mu, slots_logsigma, slots_nu = slots.split(self.dim, dim=-1)

        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)
        # q_nu = self.to_q(slots_nu)

        # E step
        # diff = torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)
        # sigma = torch.unsqueeze(torch.exp(q_logsigma), 2)
        # Вычисление расстояния для t-распределения
        # dots = ((diff / sigma)**2 * (nu / (nu - 2))).sum(dim=-1) * self.scale
        # nu = nu.squeeze(-1)
        # Плотность t-распределения
        # dots_exp = torch.exp((1 + dots / nu)**(-(nu + n) / 2)) * pi_cl
        # dots = self.compute_mahalanobis_distance(
        #     torch.unsqueeze(k, 1), 
        #     torch.unsqueeze(q_mu, 2), 
        #     torch.unsqueeze(q_logsigma, 2)
        # ) * self.scale
        diff = torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)
        sigma = torch.unsqueeze(torch.exp(q_logsigma), 2)
        dots = ((diff / sigma)**2).sum(dim=-1) * self.scale
        # dots = (((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) / torch.unsqueeze(torch.exp(q_logsigma),
        #                                                                                   2))**2).sum(dim=-1) * self.scale
        # dots = (((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) / torch.unsqueeze(torch.exp(q_logsigma),
        #                                                                                   2))**2 * (q_nu.unsqueeze(2) / (q_nu.unsqueeze(2) - 2))).sum(dim=-1) * self.scale
        # dots_exp = (torch.exp((1 + dots / q_nu)**(-(q_nu + n) / 2)) + self.eps) * pi_cl
        # pi_cl = 1 / (torch.sqrt(torch.tensor(2 * torch.pi)) * torch.exp(q_logsigma.mean(dim=-1).view(b, self.num_slots, 1)))
        
        # dots_exp = (torch.exp(-0.5 * dots) + self.eps) * pi_cl
        dots_exp = torch.exp(-0.5 * dots + self.eps) * pi_cl
        # gammas = dots_exp
        gammas = (dots_exp / self.temperature + self.eps) / ((dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        # attn = gammas
        attn = (gammas / self.temperature + self.eps) / ((gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps) # [b, k, n]
        # if self.slot_attn_smooth is not None:
        #     attn_logits = gammas.reshape(-1, n)[:, None, :] # [b*k, 1, n]
        #     # attn_logits = attn_origin.reshape(-1, n)[:, None, :] # [b*k, 1, n]
        #     img_size = int(n**0.5)
        #     attn_logits = attn_logits.reshape(-1, 1, img_size, img_size) # [b*k, 1, img_size, img_size]
        #     attn_logits = self.knconv(attn_logits) # [b*k, 1, img_size, img_size]
        #     attn_logits = attn_logits.reshape(b, self.num_slots, n) # [b, k, n]
        #     attn = (attn_logits / self.temperature + self.eps) / ((attn_logits / self.temperature).sum(dim=-1, keepdim=True) + self.eps) # [b, k, n]
        # else:
        #     attn = (gammas / self.temperature + self.eps) / ((gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps) # [b, k, n]
        
        # M step for prior probs of each gaussian
        pi_cl_new = attn.sum(dim=-1, keepdim=True)
        pi_cl_new = (pi_cl_new + self.eps) / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)
        new_dots_exp = torch.exp(-0.5 * dots + self.eps) * pi_cl_new
        new_gammas = (new_dots_exp / self.temperature + self.eps) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        new_attn = (new_gammas / self.temperature + self.eps) / ((new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps)
        pi_cl_new = new_attn.sum(dim=-1, keepdim=True)
        pi_cl_new = (pi_cl_new + self.eps) / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)
        # M step for mus
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)
        # NN update for mus
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        # updates_mu = self.mlp_mu(updates_mu)
        updates_mu = self.norm_mu(updates_mu.reshape(b, -1, d))
        updates_mu = self.mlp_mu(updates_mu) + updates_mu
        # if torch.isnan(updates_mu).any():
        #     print('updates_mu Nan appeared')
        # M step for nus
        # new_dots = (((torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)) / torch.unsqueeze(torch.exp(q_logsigma),
        #                                                                                   2))**2 * (q_nu.unsqueeze(2) / (q_nu.unsqueeze(2) - 2))).sum(dim=-1) * self.scale
        # # Байесовский подход с регуляризацией
        # weighted_diff = pi_cl_new * new_dots
        
        # # Апостериорная оценка
        # nu_posterior = (
        #     2.0 + 0.5 * pi_cl_new 
        # ) / (
        #     0.1 + 0.5 * weighted_diff
        # )
        # updates_nu = self.nu_min + (self.nu_max - self.nu_min) / (1 + torch.exp(-20 * (nu_posterior - (self.nu_min + self.nu_max)/2)))
        # updates_nu = torch.clamp(nu_posterior, min=self.nu_min, max=self.nu_max)
        # new_dots_exp = (torch.exp((1 + new_dots / updates_nu)**(-(updates_nu + n) / 2)) + self.eps) * pi_cl_new
        # new_gammas = (new_dots_exp / self.temperature) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        # new_attn = (new_gammas / self.temperature) / (new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps # [b, k, n]
        # nu = nu.unsqueeze(-1) 
        # new_diff = torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)
        # new_dots = ((new_diff / sigma)**2 * (nu / (nu - 2))).sum(dim=-1) * self.scale
        # nu = nu.squeeze(-1) 
        # new_dots_exp = torch.exp((1 + new_dots / nu)**(-(nu + n) / 2)) * pi_cl_new
        # new_gammas = (new_dots_exp / self.temperature) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        # new_attn = (new_gammas / self.temperature) / (new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps # [b, k, n]
        # Вычисление весов принадлежности
        # z = new_diff / sigma
        # nu = nu.unsqueeze(-1)
        # log_weights = self.log_t_pdf(z, sigma, nu, n)
        # weights = torch.softmax(log_weights, dim=1)
        # nu_update = torch.zeros_like(nu)
        # for k in range(self.num_slots):
        #         # Эвристика оценки nu
        #         psi_term = torch.sum(
        #             weights[:, k] * torch.log(1 + new_diff[:, k]**2 / nu[:, k])
        #         ) / torch.sum(weights[:, k])
        #         # Аппроксимация nu
        #         nu_update[k] = 2 / (psi_term - 1)
        # updates_nu = torch.clamp(nu_update, self.nu_min, self.nu_max)
        # M step for logsigmas for new mus
        # For GMM
        new_diff = torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)
        new_dots = ((new_diff / sigma)**2).sum(dim=-1) * self.scale
        # new_dots = (((torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)) / torch.unsqueeze(torch.exp(q_logsigma),
        #                                                                                   2))**2).sum(dim=-1) * self.scale
        new_dots_exp = torch.exp(-0.5 * new_dots + self.eps) * pi_cl_new
        new_gammas = (new_dots_exp / self.temperature + self.eps) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        new_attn = (new_gammas / self.temperature + self.eps) / ((new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps)
        # new_attn = new_gammas
        # if self.slot_attn_smooth is not None:
        #     attn_logits = new_gammas.reshape(-1, n)[:, None, :] # [b*k, 1, n]
        #     # attn_logits = attn_origin.reshape(-1, n)[:, None, :] # [b*k, 1, n]
        #     img_size = int(n**0.5)
        #     attn_logits = attn_logits.reshape(-1, 1, img_size, img_size) # [b*k, 1, img_size, img_size]
        #     attn_logits = self.knconv(attn_logits) # [b*k, 1, img_size, img_size]
        #     attn_logits = attn_logits.reshape(b, self.num_slots, n) # [b, k, n]
        #     new_attn = (attn_logits / self.temperature + self.eps) / ((attn_logits / self.temperature).sum(dim=-1, keepdim=True) + self.eps) # [b, k, n]
        # else:
        #     new_attn = (new_gammas / self.temperature + self.eps) / ((new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps) # [b, k, n]
        # new_attn = (new_gammas / self.temperature + self.eps) / ((new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps)
        # updates_logsigma = 0.5 * torch.log(
        #     torch.einsum('bijd,bij->bid', 
        #         ((torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2))**2 + self.eps),
        #         new_attn * (updates_nu / (updates_nu - 2))  # Модификация веса
        #     )
        # )
        # For GMM
        pi_cl_new = new_attn.sum(dim=-1, keepdim=True)
        pi_cl_new = (pi_cl_new + self.eps) / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)
        vdiff = torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2)
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', vdiff**2, new_attn) + self.eps)
        #     ((torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2))**2 + self.eps),
        #     new_attn
        #     )
        # )
        # updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', 
        #     ((torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2))**2 + self.eps) * (updates_nu / (updates_nu - 2)),
        #     new_attn
        #     )
        # )
        # if torch.isnan(updates_logsigma).any():
        #     print('updates_logsigma Nan appeared')
        conv_gammas = None
        if last:
            # new_dots = (((torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)) / torch.unsqueeze(torch.exp(updates_logsigma),
            #                                                                                 2))**2 * (updates_nu.unsqueeze(2) / (updates_nu.unsqueeze(2) - 2))).sum(dim=-1) * self.scale
            new_diff = torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)
            new_sigma = torch.unsqueeze(torch.exp(updates_logsigma), 2)
            new_dots = ((new_diff / new_sigma)**2).sum(dim=-1) * self.scale
            # new_dots = (((torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)) / torch.unsqueeze(torch.exp(updates_logsigma),
            #                                                                                 2))**2).sum(dim=-1) * self.scale
            # new_dots_exp = (torch.exp((1 + new_dots / updates_nu)**(-(updates_nu + n) / 2)) + self.eps) * pi_cl_new
            new_dots_exp = torch.exp(-0.5 * new_dots + self.eps) * pi_cl_new
            new_gammas = (new_dots_exp / self.temperature + self.eps) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
            if self.slot_attn_smooth is not None:
                attn_logits = new_gammas.reshape(-1, n)[:, None, :] # [b*k, 1, n]
                # attn_logits = attn_origin.reshape(-1, n)[:, None, :] # [b*k, 1, n]
                img_size = int(n**0.5)
                attn_logits = attn_logits.reshape(-1, 1, img_size, img_size) # [b*k, 1, img_size, img_size]
                attn_logits = self.knconv(attn_logits) # [b*k, 1, img_size, img_size]
                conv_gammas = attn_logits.reshape(b, self.num_slots, n) # [b, k, n]
        # new gaussians params
        # For GMM
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)
        # slots = torch.cat((updates_mu, updates_logsigma, updates_nu), dim=-1)
        # Косвенная энергетическая регуляризация
        # energy_reg = self.compute_energy_regularization(slots)
        # Мягкое ограничение через энергетический штраф
        # slots = slots * (1.0 - energy_reg.detach())
        # Вычисление log-likelihood для t-распределения
        # log_likelihood = torch.log(dots_exp + self.eps).sum()
        log_likelihood = torch.tensor(0, device=slots.device)
        # log_likelihood += torch.log(pi_cl * dots_exp + self.eps).sum()
        # print(-log_likelihood)
        return slots, pi_cl_new, -log_likelihood, new_gammas, conv_gammas # pi_cl_new

    def forward(self, slots: torch.Tensor, inputs: torch.Tensor, n_iters: Optional[int] = None): #, current_step: Optional[int] = None):
        # if current_step is not None:
        #     self.current_step = current_step
        #     if self.current_step <= self.decay_steps:
        #         scale = (self.final_temperature / self.init_temperature)**(self.current_step / (self.decay_steps - 1))
        #         temp = (self.init_temperature * scale)
        #         temp_rounded = round(temp, 4)
        #         multiple = temp_rounded // 0.5
        #         if temp_rounded % 0.5 == 0 and multiple not in self.used_multiples:
        #             self.temperature = round(temp, 1)
        #             self.used_multiples.append(multiple)
        #             print(f"Update temperature: self.temperature={self.temperature}, current_step is {self.current_step}")
        b, n, d, device = *inputs.shape, inputs.device
        
        n_s = self.num_slots

        pi_cl = (torch.ones(b, n_s, 1) / n_s).to(device)
        # nu = self.nu.to(device)[:b]
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        if n_iters is None:
            n_iters = self.iters
        
        for _ in range(n_iters):
            slots, pi_cl, log_dict, attn, conv_attn = self.step(slots, k, v, b, n, d, pi_cl)
        slots, pi_cl, log_dict, attn, conv_attn = self.step(slots.detach(), k, v, b, n, d, pi_cl, last=True)
        log_l = log_dict
        
        # if self.mlp_out is not None:
        slots = self.mlp_out(slots)

        return {"slots": slots, "masks": attn}#, "conv_masks": conv_attn} # self.mlp_out(self.dropout(slots))?
        
        # if self.input_type == "image":
        #     return {"slots": self.mlp_out(slots), "masks": attn} # self.mlp_out(self.dropout(slots))?
        # else:
        #     return {"slots": slots, "masks": attn} # self.mlp_out(self.dropout(slots))?

class TemporalGMM(nn.Module):
    def __init__(
            self, 
            num_slots: int, 
            dim: int, 
            attn_smooth: Optional[str] = None, 
            iters: int = 3, 
            temp_reg: int = 0.1,
            temperature: float = 1.0,
            hidden_dim: Optional[int] = None, # 128 (original), 256
            gau_min: float = 0.1,
            gau_max: float = 2.0, 
            attn_smooth_size: int = 5,
            drop_rate: float = 0.2,
            logsigma_gate: float = 0.1, # 0.9
            mu_gate: float = 0.1,
            momentum: float = 0.2,
            eps: int = 1e-10, 
            use_mlp: bool = True,
            # chunk: int = 4,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.logsigma_gate = logsigma_gate
        self.mu_gate = mu_gate
        self.momentum = momentum
        self.eps = eps
        
        self.dim = dim
        self.num_slots = num_slots
        self.temp_reg = temp_reg
        self.scale = dim ** -0.5
        if hidden_dim is None:
            hidden_dim = dim * 4
        else:
            hidden_dim = hidden_dim
        # Гиперболические преобразования
        self.hyp_mu = nn.Linear(dim, dim)
        self.p_sigma = nn.Linear(dim, dim)
        self.hyp_prev_mu = nn.Linear(dim, dim)
        self.p_prev_sigma = nn.Linear(dim, dim)
        self.p_mu = nn.Linear(dim, dim)
        
        # Диффузионные параметры
        self.norm_diffusion_mu = nn.LayerNorm(dim * 2)
        self.diffusion_mu = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, dim)
        )
        self.norm_diffusion_logsigma = nn.LayerNorm(dim * 2)
        self.diffusion_logsigma = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, dim)
        )
        
        # Временные регуляризаторы
        self.temp_attn_mu = nn.MultiheadAttention(dim, num_heads=4, dropout=0.1, batch_first=True)
        # self.temp_attn_logsigma = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.temp_norm_mu = nn.LayerNorm(dim)
        # self.temp_norm_logsigma = nn.LayerNorm(dim)
        self.temp_norm_attn_mu = nn.LayerNorm(dim)
        # self.temp_norm_attn_logsigma = nn.LayerNorm(dim)
        self.temp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        self.temp_logsigma = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Динамические веса
        # self.mu_gate = nn.Parameter(torch.ones(1, num_slots, dim))
        # self.sigma_gate = nn.Parameter(torch.ones(1, num_slots, dim))
        
        # Временная память
        self.memory_net = nn.GRUCell(2*dim, 2*dim)
        self.gru = nn.GRUCell(dim, dim)
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.slot_attn_smooth = attn_smooth
        self.temperature = temperature

        if self.slot_attn_smooth != None:
            kernel_size = attn_smooth_size
            if self.slot_attn_smooth.lower() == 'gaussian':
                sigma_min = gau_min
                sigma_max = gau_max
                self.knconv = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, 
                                                               sigma=(sigma_min, sigma_max))
            elif self.slot_attn_smooth.lower() == 'conv':
                self.knconv = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=kernel_size,
                                     padding=kernel_size // 2,
                                     bias=False)
            elif self.slot_attn_smooth.lower() == 'wnconv':
                self.knconv = WNConv(in_channels=1,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2,
                                    bias=False)
            elif self.slot_attn_smooth.lower() == 'wnconv3d':
                self.knconv = WNConv3d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=kernel_size,
                                       padding=kernel_size // 2,
                                       bias=False)
        else:
            self.knconv = None

        self.gru_mu = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim * 2)
        self.norm_prev_slots = nn.LayerNorm(dim * 2)
        # self.norm_slots = nn.LayerNorm(dim)
        self.norm_mu = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)
        if use_mlp:
            self.mlp_out = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 2, dim)
            )
        else:
            self.mlp_out = None
        if drop_rate is not None:
            self.dropout = nn.Dropout(drop_rate)

    # def hyperbolic_project(self, x):
    #     return x / (1 + torch.norm(x, dim=-1, keepdim=True))
    def hyperbolic_distance(self, x, y, eps=1e-6):
        """Расстояние в модели Пуанкаре между точками x и y"""
        # Обеспечиваем, что векторы находятся внутри единичного шара
        x_norm = torch.sum(x**2, dim=-1, keepdim=True)
        y_norm = torch.sum(y**2, dim=-1, keepdim=True)
        
        # Проверка и корректировка для числовой стабильности
        # x_norm = torch.clamp(x_norm, 0.0, 0.999)
        # y_norm = torch.clamp(y_norm, 0.0, 0.999)
        max_norm = 1.0 - eps
        x_norm = torch.sigmoid(x_norm)
        x_norm = torch.clamp(x_norm, min=0.0, max=max_norm)
        y_norm = torch.sigmoid(y_norm)
        y_norm = torch.clamp(y_norm, min=0.0, max=max_norm)
        
        # Вычисление гиперболического расстояния
        dot_product = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = torch.clamp(2 * ((x_norm - 2*dot_product + y_norm)), min=1e-7)
        denominator = torch.clamp((1-x_norm) * (1-y_norm), min=1e-7)
        
        return torch.acosh(1 + numerator / denominator)

    def hyperbolic_project(self, x, eps=1e-7):
        """Проецирует векторы на диск Пуанкаре с обеспечением стабильности"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Ограничиваем норму для числовой стабильности
        norm = torch.sigmoid(norm)
        max_norm = 1.0 - eps
        norm = torch.clamp(norm, min=0.0, max=max_norm)
        
        # Проецирование на гиперболическое пространство
        scaled_x = x / (1 + norm)
        # Дополнительная проверка для избежания краевых случаев
        scaled_norm = torch.norm(scaled_x, dim=-1, keepdim=True)
        if torch.any(scaled_norm >= 1.0):
            scaled_x = scaled_x / (scaled_norm + 1e-3)
            # print(torch.norm(scaled_x, dim=-1).max())
            # scaled_x = scaled_x / (scaled_norm * 1.000001)
        
        return scaled_x
    
    def euclidean_project(self, x, eps=1e-6):
        """Проецирует векторы из гиперболического пространства в евклидово"""
        # Нормируем векторы
        norm = torch.norm(x, dim=-1, keepdim=True)
        if torch.any(norm >= 1):
            print("base points norm >= 1 in euclidean_project func")
            norm = torch.sigmoid(norm)
        # Вычисляем евклидическое представление
        # Проекция на евклидическое пространство
        # Формула: x_euclidean = (2 * x) / (1 - norm^2)
        # Убедимся, что norm < 1 для корректности
        norm_squared = norm ** 2
        # if torch.any(norm_squared == 1):
        #     norm_squared = torch.where(norm_squared == 1, torch.tensor(1 - eps), norm_squared)
        norm_squared = torch.clamp(norm_squared, max=1-eps)  # Избегаем деления на ноль

        # Проекция
        x_euclidean = (2 * x) / (1 - norm_squared)
        
        return x_euclidean

    
    def exp_map(self, x_tangent, base_point, eps=1e-6):
        """Экспоненциальное отображение из касательного пространства в гиперболическое
        
        Args:
            x_tangent: вектор в касательном пространстве
            base_point: точка в гиперболическом пространстве (центр касательного пространства)
        """
        # Нормализуем для численной стабильности
        x_norm = torch.norm(x_tangent, dim=-1, keepdim=True)
        base_norm = torch.norm(base_point, dim=-1, keepdim=True)
        if torch.any(base_norm >= 1):
            print("base_points norm >= 1 in exp_map func")
        
        # Избегаем деления на 0
        x_norm = torch.clamp(x_norm, min=eps)
        coef = torch.tanh(x_norm) / x_norm
        
        # Вычисляем конформный множитель для модели Пуанкаре
        factor = 1 - base_norm**2
        
        # Применяем отображение
        mapped = base_point + factor * coef * x_tangent
        
        # Проецируем результат на единичный шар
        return self.hyperbolic_project(mapped)

    def log_map(self, y, base_point, eps=1e-6):
        """Логарифмическое отображение из гиперболического пространства в касательное
        
        Args:
            y: точка в гиперболическом пространстве
            base_point: центр касательного пространства
        """
        # Вычисляем разницу векторов
        diff = y - base_point
        
        # Вычисляем норму разницы
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)
        base_norm = torch.norm(base_point, dim=-1, keepdim=True)
        if torch.any(base_norm >= 1):
            print("base_points norm >= 1 in log_map func")
        
        # Избегаем деления на 0
        diff_norm = torch.clamp(diff_norm, min=eps)
        
        # Вычисляем конформный множитель для модели Пуанкаре
        factor = 1 - base_norm**2
        if torch.any(factor == 0):
            factor = torch.where(factor == 0, torch.tensor(eps), factor)
        # Применяем отображение
        raw_ratio = diff_norm / factor
        norm_ratio = torch.tanh(raw_ratio)
        norm_ratio = torch.clamp(norm_ratio, min=-1+eps, max=1-eps)
        coef = torch.atanh(norm_ratio) / diff_norm
        return coef * diff

    def geodesic_interpolation(self, p1, p2, t):
        """Геодезическая интерполяция между двумя точками в гиперболическом пространстве
        
        Args:
            p1, p2: точки в гиперболическом пространстве
            t: параметр интерполяции, 0 <= t <= 1
        """
        # Конвертируем p2 в касательное пространство относительно p1
        v = self.log_map(p2, p1)
        
        # Интерполируем в касательном пространстве
        interpolated = self.exp_map(t * v, p1)
        
        return interpolated

    def step(self, slots, prev_slots, k, v, b, n, d, pi_cl, last=False, first=False):
        """Шаг EM алгоритма для временной GMM в гиперболическом пространстве"""
        # norm_mu = self.norm_slots(slots[..., :self.dim])
        # norm_logsigma = self.norm_slots(slots[..., self.dim:])
        # Гиперболическое проецирование параметров
        slots = self.norm_slots(slots)
        norm_mu, norm_logsigma = slots[..., :self.dim], slots[..., self.dim:]
        slots_mu = self.hyperbolic_project(self.hyp_mu(norm_mu))
        # Для logsigma не требуется гиперболическое проецирование, так как это параметр масштаба
        slots_logsigma = self.p_sigma(norm_logsigma)
        
        # Диффузионный процесс для параметров с временной зависимостью
        if prev_slots is not None:
            # norm_prev_mu = self.norm_slots(prev_slots[..., :self.dim])
            # norm_prev_logsigma = self.norm_slots(prev_slots[..., self.dim:])
            prev_slots = self.norm_slots(prev_slots)
            norm_prev_mu, norm_prev_logsigma = prev_slots[..., :self.dim], prev_slots[..., self.dim:]
            prev_hyp_mu = self.hyperbolic_project(self.hyp_prev_mu(norm_prev_mu))
            prev_logsigma = self.p_prev_sigma(norm_prev_logsigma)
            
            # Прогнозирование диффузии с ограничением градиентов для стабильности
            # mu_diff = self.diffusion_mu(torch.cat([prev_mu, slots_mu], -1))
            # sigma_diff = self.diffusion_sigma(torch.cat([prev_logsigma, slots_logsigma], -1))
            # slots_mu = prev_mu + self.mu_gate * mu_diff
            # slots_logsigma = prev_logsigma + self.sigma_gate * sigma_diff
            # slots_mu = self.mu_gate * slots_mu + (1 - self.mu_gate) * (prev_mu + mu_diff) # self.momentum
            # slots_logsigma = self.sigma_gate * slots_logsigma + (1 - self.sigma_gate) * (prev_logsigma + sigma_diff) # self.momentum
            # 2. Правильное вычисление разницы в касательном пространстве
            # Переводим slots_mu в касательное пространство относительно prev_mu
            slots_mu_tangent = self.log_map(slots_mu, prev_hyp_mu)
            # Вычисляем дифференциал в касательном пространстве
            # Concat слотов в касательном пространстве и евклидового пространства логсигм
            mu_tangent_input = torch.cat([prev_hyp_mu, slots_mu_tangent], -1) # поменять порядок
            mu_tangent_diff = self.diffusion_mu(mu_tangent_input)
            # Для сигмы оставляем обычное евклидово пространство
            logsigma_input = torch.cat([prev_logsigma, slots_logsigma], -1) # поменять порядок
            logsigma_diff = self.diffusion_logsigma(logsigma_input)
            # 3. Геодезическая интерполяция в гиперболическом пространстве
            # Вычисляем целевую точку, добавляя дифференциал в касательном пространстве
            target_mu = self.exp_map(mu_tangent_diff, prev_hyp_mu)
            # slots_hyp_mu = self.momentum * slots_mu + (1 - self.momentum) * (prev_hyp_mu + mu_tangent_diff)
            
            # Интерполируем между предыдущим mu и целевым mu с коэффициентом mu_gate
            slots_hyp_mu = self.geodesic_interpolation(slots_mu, target_mu, self.momentum) # self.mu_gate
            
            # Для сигмы используем обычное EMA (она в евклидовом пространстве)
            slots_logsigma = (1 - self.momentum) * slots_logsigma + self.momentum * (prev_logsigma + logsigma_diff) # prev_logsigma + sigma_diff * self.momentum # self.sigma_gate

        slots_mu = self.euclidean_project(slots_hyp_mu) # self.temp_norm_mu(self.euclidean_project(slots_hyp_mu))

        # p_mu = self.p_mu(norm_mu)
        # Временная согласованность через attention механизм
        # Это помогает слотам «помнить» свою историю
        temp_attn_mu, _ = self.temp_attn_mu(
            slots_mu, 
            k, 
            v
        )
        # temp_attn_mu, _ = self.temp_attn(
        #     p_mu, 
        #     slots_mu, 
        #     slots_mu
        # )
        # dots = torch.einsum("bsd, bfd -> bsf", p_mu, slots_mu) * self.scale
        # pre_norm_attn = torch.softmax(dots, dim=1)
        # attn = pre_norm_attn + self.eps
        # attn = attn / attn.sum(-1, keepdim=True)

        # updates = torch.einsum("bsf, bfd -> bsd", attn, values)
        # updated_mu = self.gru(temp_attn_mu.squeeze(1).flatten(0, 1), norm_mu.flatten(0, 1))
        # slots_mu = updated_mu.unflatten(0, norm_mu.shape[:2])
        slots_mu = slots_mu + self.temp_mu(self.temp_norm_attn_mu(temp_attn_mu.squeeze(1)))
        # slots_mu = slots_mu + self.temp_mu(self.temp_norm(temp_attn_mu.squeeze(1)))
        # Обновляем центры с учетом self-attention и проецируем обратно
        # slots_hyp_mu = slots_hyp_mu + self.hyperbolic_project(temp_attn_mu.squeeze(1))
        # slots_mu = self.euclidean_project(slots_hyp_mu)
        # # Временная согласованность через attention
        # temp_attn_mu, _ = self.temp_attn(
        #     slots_mu, 
        #     slots_mu, 
        #     slots_mu
        # )
        # slots_mu = slots_mu + temp_attn_mu.squeeze(1)

        
        # temp_attn_logsigma, _ = self.temp_attn(
        #     sigma_diff,
        #     slots_logsigma,
        #     slots_logsigma
        # )
        # slots_logsigma = slots_logsigma + temp_attn_logsigma.squeeze(1)
        
        # Объединенные параметры
        # slots = torch.cat([slots_mu, slots_logsigma], -1)
        
        # E-шаг
        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)
        diff = k.unsqueeze(1) - q_mu.unsqueeze(2)
        sigma = torch.exp(q_logsigma).unsqueeze(2)
        dots = ((diff / sigma)**2).sum(-1) * self.scale
        # Временная регуляризация
        # if prev_slots is not None:
        #     temp_penalty = (slots_mu - prev_mu).norm(dim=-1, keepdim=True)
        #     dots += self.temp_reg * temp_penalty
        # if prev_slots is not None:
        #     slots_hyp_mu = self.hyperbolic_project(slots_mu)
        #     temp_penalty = self.hyperbolic_distance(slots_hyp_mu.unsqueeze(2), prev_hyp_mu.unsqueeze(2))
        #     dots += self.temp_reg * self.euclidean_project(temp_penalty).squeeze(-1)
        
        dots_exp = torch.exp(-0.5 * dots + self.eps) * pi_cl
        gammas = (dots_exp / self.temperature + self.eps) / ((dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        attn = (gammas / self.temperature + self.eps) / ((gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps) # [b, k, n]
        
        # M step for prior probs of each gaussian
        pi_cl_new = attn.sum(dim=-1, keepdim=True)
        pi_cl_new = (pi_cl_new + self.eps) / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)
        new_dots_exp = torch.exp(-0.5 * dots + self.eps) * pi_cl_new
        new_gammas = (new_dots_exp / self.temperature + self.eps) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        new_attn = (new_gammas / self.temperature + self.eps) / ((new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps)
        pi_cl_new = new_attn.sum(dim=-1, keepdim=True)
        pi_cl_new = (pi_cl_new + self.eps) / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)
        # M step for mus
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)
        # NN update for mus
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        updates_mu = self.norm_mu(updates_mu.reshape(b, -1, d))
        updates_mu = self.mlp_mu(updates_mu) + updates_mu
        if torch.isnan(updates_mu).any():
            print('updates_mu Nan appeared')
        # M step for logsigmas for new mus
        # For GMM
        new_diff = torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)
        new_dots = ((new_diff / sigma)**2).sum(dim=-1) * self.scale
        new_dots_exp = torch.exp(-0.5 * new_dots + self.eps) * pi_cl_new
        new_gammas = (new_dots_exp / self.temperature + self.eps) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
        new_attn = (new_gammas / self.temperature + self.eps) / ((new_gammas / self.temperature).sum(dim=-1, keepdim=True) + self.eps)
        # For GMM
        pi_cl_new = new_attn.sum(dim=-1, keepdim=True)
        pi_cl_new = (pi_cl_new + self.eps) / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)
        vdiff = torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2)
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', vdiff**2, new_attn) + self.eps)
        if torch.isnan(updates_logsigma).any():
            print('updates_logsigma Nan appeared')
        if last:
            new_diff = torch.unsqueeze(k, 1) - torch.unsqueeze(updates_mu, 2)
            new_sigma = torch.unsqueeze(torch.exp(updates_logsigma), 2)
            new_dots = ((new_diff / new_sigma)**2).sum(dim=-1) * self.scale
            new_dots_exp = torch.exp(-0.5 * new_dots + self.eps) * pi_cl_new
            new_gammas = (new_dots_exp / self.temperature + self.eps) / ((new_dots_exp / self.temperature).sum(dim=1, keepdim=True) + self.eps)
            if self.knconv is not None and not isinstance(self.knconv, WNConv3d):
                attn_logits = new_gammas.reshape(-1, n)[:, None, :] # [b*k, 1, n]
                img_size = int(n**0.5)
                attn_logits = attn_logits.reshape(-1, 1, img_size, img_size) # [b*k, 1, img_size, img_size]
                attn_logits = self.knconv(attn_logits) # [b*k, 1, img_size, img_size]
                new_gammas = attn_logits.reshape(b, self.num_slots, n) # [b, k, n]
        # new gaussians params
        # For GMM
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)
        log_likelihood = torch.tensor(0, device=slots.device)

        return slots, pi_cl_new, -log_likelihood, new_gammas
    
    def forward(self, slots, inputs, n_iters=None, prev_state=None):
        b, n, d = inputs.shape
        device = inputs.device
        # Инициализация памяти
        # if prev_state is None:
        #     prev_mu = torch.zeros(b, self.num_slots, d).to(device)
        #     prev_sigma = torch.ones(b, self.num_slots, d).to(device)
        #     prev_slots = torch.cat([prev_mu, prev_sigma], dim=-1)
        # else:
        #     prev_slots = prev_state
        if prev_state is not None:
            h = self.norm_slots(prev_state)
        else:
            h = torch.zeros(b, self.num_slots, 2*d).to(device)
        # Обновление памяти
        slots = self.norm_slots(slots)
        slots_flat = slots.view(-1, 2*d)
        h_flat = h.view(-1, 2*d)
        new_h = self.memory_net(slots_flat, h_flat)
        new_h = new_h.view(b, self.num_slots, 2*d)
        slots = slots + new_h

        pi_cl = (torch.ones(b, self.num_slots, 1) / self.num_slots).to(device)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        if n_iters is None:
            n_iters = self.iters
        # Основной шаг обработки
        for _ in range(n_iters):
            slots, pi_cl, log_l, attn = self.step(
                slots, prev_slots=new_h, # .detach(),
                k=k, v=v, b=b, n=n, d=d, pi_cl=pi_cl
            )
        
        slots, pi_cl, log_dict, attn = self.step(
            slots.detach(), prev_slots=new_h.detach(),
            k=k, v=v, b=b, n=n, d=d, pi_cl=pi_cl, last=True
        )
        log_l = log_dict
        
        if self.mlp_out is not None:
            slots = self.mlp_out(slots)
        
        return {
            "slots": slots,
            "masks": attn,
            # "memory": new_h
        }