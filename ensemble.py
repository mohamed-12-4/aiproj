"""LoRA-Ensemble adapted from the official implementation to DINOv3 segmentation."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from huggingface_hub import snapshot_download

from third_party.lora_ensemble.lora import Init_Weight, LoRA


class MemberProjection(nn.Module):
    """Route one member at a time through the upstream, unmodified LoRA class."""

    def __init__(self, linear, members, settings):
        super().__init__()
        self.member = 0
        self.adapters = nn.ModuleList([
            LoRA(linear, settings["r"], linear.in_features,
                 out_dim=linear.out_features, init_type=Init_Weight.XAVIER_UNIFORM,
                 init_settings={"gain": settings["init_gain"]})
            for _ in range(members)
        ])

    def forward(self, x):
        return self.adapters[self.member](x)


class CoralEnsemble(nn.Module):
    def __init__(self, model_config, lora_config, ensemble_config, cache_dir,
                 local_files_only=False):
        super().__init__()
        self.model_config = dict(model_config)
        self.lora_config = dict(lora_config)
        self.ensemble_config = dict(ensemble_config)
        self.n_members = ensemble_config["members"]
        if self.n_members < 1 or lora_config["r"] < 1:
            raise ValueError("Use positive member count and LoRA rank.")
        source = model_config["id"]
        if local_files_only:
            source = snapshot_download(source, revision=model_config.get("revision"),
                                       cache_dir=cache_dir, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(
            source, revision=model_config.get("revision"),
            cache_dir=cache_dir, local_files_only=local_files_only,
            attn_implementation="sdpa", output_hidden_states=True,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()  # Disable DINOv3's random RoPE coordinate rescaling.
        self.patch_size = self.backbone.config.patch_size
        self.prefix_tokens = 1 + self.backbone.config.num_register_tokens
        self.projections = []  # Modules are registered through the backbone below.
        counts = {name: 0 for name in lora_config["target_modules"]}
        for module in list(self.backbone.modules()):
            for name, child in list(module.named_children()):
                if name in counts and isinstance(child, nn.Linear):
                    projection = MemberProjection(child, self.n_members, lora_config)
                    setattr(module, name, projection)
                    self.projections.append(projection)
                    counts[name] += 1
        if any(count != self.backbone.config.num_hidden_layers for count in counts.values()):
            raise ValueError(f"Unexpected DINOv3 attention projection layout: {counts}")
        hidden = self.backbone.config.hidden_size
        channels = model_config["head_channels"]
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden * len(model_config["feature_layers"]), hidden, 1),
                nn.BatchNorm2d(hidden), nn.GELU(),
                nn.Conv2d(hidden, channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(channels, 4, 1),
            ) for _ in range(self.n_members)
        ])

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbone/adapters contain no train-time dropout in this method.
        # Heads still train their own BatchNorm; gradients through adapters remain enabled.
        self.backbone.eval()
        return self

    def member_logits(self, images, member):
        for projection in self.projections:
            projection.member = member
        batch, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Input height/width must be divisible by the patch size.")
        states = self.backbone(pixel_values=images).hidden_states
        features = []
        for layer in self.model_config["feature_layers"]:
            tokens = states[layer][:, self.prefix_tokens:]
            features.append(tokens.transpose(1, 2).reshape(
                batch, -1, height // self.patch_size, width // self.patch_size,
            ))
        logits = self.decoders[member](torch.cat(features, dim=1))
        return F.interpolate(logits, (height, width), mode="bilinear", align_corners=False)

    def prediction_stats(self, images, with_members=False):
        """Probability mean, predictive entropy and member disagreement (MI)."""
        mean = None
        expected_entropy = 0
        members = []
        for member in range(self.n_members):
            log_prob = self.member_logits(images, member).float().log_softmax(1)
            prob = log_prob.exp()
            mean = prob / self.n_members if mean is None else mean + prob / self.n_members
            expected_entropy = expected_entropy - (prob * log_prob).sum(1) / self.n_members
            if with_members:
                members.append(prob)
        entropy = -(mean * mean.clamp_min(1e-8).log()).sum(1)
        return {"probabilities": mean, "entropy": entropy,
                "mutual_information": (entropy - expected_entropy).clamp_min(0),
                "members": members}

    def forward(self, images):
        # Existing prediction/evaluation tools accept log probabilities as logits.
        return self.prediction_stats(images)["probabilities"].clamp_min(1e-8).log()

    def trainable_state(self):
        state = {name: value.detach().cpu().clone() for name, value in self.named_parameters()
                 if value.requires_grad}
        state.update({name: value.detach().cpu().clone() for name, value in self.named_buffers()
                      if name.startswith("decoders.")})
        return state

    def load_trainable_state(self, state):
        expected = set(self.trainable_state())
        if expected != set(state):
            raise ValueError(f"Checkpoint trainable keys differ: {expected ^ set(state)}")
        tensors = dict(self.named_parameters()) | dict(self.named_buffers())
        with torch.no_grad():
            for name, value in state.items():
                tensors[name].copy_(value)
