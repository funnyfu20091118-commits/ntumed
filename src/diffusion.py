"""
Diffusion utilities: noise schedule, forward process, sampling (DDPM / DDIM).
"""
import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """Handles the forward noising and reverse sampling for latent diffusion."""

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = "linear", device: str = "cuda"):
        self.num_timesteps = num_timesteps
        self.device = device

        # Beta schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            t = np.linspace(0, num_timesteps, steps) / num_timesteps
            alphas_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
            betas = np.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        """Recover x0 from predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha

    @torch.no_grad()
    def ddpm_sample_step(self, model, x_t, t_idx, text_emb):
        """Single DDPM reverse step."""
        B = x_t.shape[0]
        t = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)

        eps_pred = model(x_t, t, text_emb)

        # Posterior mean
        mean = (
            self.posterior_mean_coef1[t_idx] * self.predict_x0_from_eps(x_t, t, eps_pred)
            + self.posterior_mean_coef2[t_idx] * x_t
        )

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            variance = torch.exp(0.5 * self.posterior_log_variance[t_idx])
            return mean + variance * noise
        return mean

    @torch.no_grad()
    def ddpm_sample(self, model, shape, text_emb):
        """Full DDPM reverse sampling."""
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.num_timesteps)):
            x = self.ddpm_sample_step(model, x, t, text_emb)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, text_emb, ddim_steps: int = 50, eta: float = 0.0):
        """DDIM deterministic or stochastic sampling."""
        # Create sub-sequence of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=self.device)

        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            B = x.shape[0]
            t = torch.full((B,), t_cur, device=x.device, dtype=torch.long)

            eps_pred = model(x, t, text_emb)

            # Predicted x0
            alpha_cur = self.alphas_cumprod[t_cur]
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)

            x0_pred = (x - torch.sqrt(1 - alpha_cur) * eps_pred) / torch.sqrt(alpha_cur)
            x0_pred = torch.clamp(x0_pred, -3, 3)  # clip for stability

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_cur) * (1 - alpha_cur / alpha_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * eps_pred
            noise = torch.randn_like(x) if t_cur > 0 else 0
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise

        return x
