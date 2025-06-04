import torch
import torch.nn as nn
import torch.nn.functional as F

class AugmentedSpeciesNet(nn.Module):
    def __init__(self, base_model, original_outputs, target_labels):
        super().__init__()
        self.base_model = base_model
        self.extra_head = nn.Linear(original_outputs, target_labels)

        # Freeze base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Weight initialization for stability
        nn.init.xavier_uniform_(self.extra_head.weight)
        nn.init.zeros_(self.extra_head.bias)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] → [B, H, W, C]
        base_logits = self.base_model(x.to(x.device))  # Ensure consistent device usage

        out = self.extra_head(base_logits)

        out = F.softmax(out, dim=1)

        return out