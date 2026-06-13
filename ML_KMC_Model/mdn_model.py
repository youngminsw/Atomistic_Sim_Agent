# ===== mdn_model.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiOutputMDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gaussians, dropout_rate=0.3):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.pi    = nn.Linear(hidden_dim, num_gaussians)
        self.mu    = nn.Linear(hidden_dim, num_gaussians * output_dim)
        self.sigma = nn.Linear(hidden_dim, num_gaussians * output_dim)
        self.K     = num_gaussians
        self.D     = output_dim

    def forward(self, x):
        h     = self.hidden(x)
        pi    = F.softmax(self.pi(h), dim=1)                   # [B, K]
        mu    = self.mu(h).view(-1, self.K, self.D)            # [B, K, D]
        
        sigma = torch.exp(self.sigma(h)).clamp(min=1e-4, max=1.0).view(-1, self.K, self.D)

        return pi, mu, sigma

# MDN loss (for training)
def mdn_multi_loss(pi, mu, sigma, target):
    target = target.unsqueeze(1)  # [B, 1, D]
    m = torch.distributions.Normal(mu, sigma)  # [B, K, D]
    log_probs = m.log_prob(target).sum(dim=2)  # [B, K]
    log_pi = torch.log(pi + 1e-8)              # [B, K]

    log_weighted = log_pi + log_probs          # [B, K]
    log_sum = torch.logsumexp(log_weighted, dim=1)  # [B]
    nll = -log_sum                             # [B]
    return nll.mean()
