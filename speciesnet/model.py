import torch
import torch.nn as nn

# === Device Selection ===
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

class AugmentedSpeciesNet(nn.Module):
    def __init__(self, base_model, original_outputs, target_labels):
        super().__init__()
        self.base_model = base_model
        self.extra_head = nn.Linear(original_outputs, target_labels)

        # Weight initialization for stability
        nn.init.xavier_uniform_(self.extra_head.weight)
        nn.init.zeros_(self.extra_head.bias)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] → [B, H, W, C]
        base_logits = self.base_model(x.to(x.device))  # Ensure consistent device usage

        # Normalize base_logits per batch
        base_logits = (base_logits - base_logits.mean(dim=1, keepdim=True)) / \
                      (base_logits.std(dim=1, keepdim=True) + 1e-6)

        return self.extra_head(base_logits)