import torch
import torch.nn as nn

class AugmentedSpeciesNet(nn.Module):
    def __init__(self, base_model, original_outputs, extra_outputs=1):
        super().__init__()
        self.base_model = base_model
        self.extra_head = nn.Linear(original_outputs, original_outputs + extra_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] → [B, H, W, C]
        with torch.no_grad():
            base_logits = self.base_model(x)  # [B, original_outputs]
        return self.extra_head(base_logits)   # [B, original_outputs + 1]