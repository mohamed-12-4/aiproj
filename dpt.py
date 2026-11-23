"""Single DINOv3 with PEFT LoRA and a DPT (or matched shallow) decoder.

DPT uses the same Transformers components as the authors' public implementation:
https://huggingface.co/EPFL-ECEO/coralscapes-vit-b-dpt/blob/main/coralscapes_hub_model.py
Only the original, revision-pinned DINOv3 encoder is loaded as pretrained weights.
"""

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, BitsAndBytesConfig, DPTConfig
from transformers.models.dpt.modeling_dpt import DPTNeck, DPTSemanticSegmentationHead


class DPTDecoder(nn.Module):
    def __init__(self, hidden_size, patch_size, settings):
        super().__init__()
        self.config = DPTConfig(
            hidden_size=hidden_size, patch_size=patch_size,
            num_labels=settings["num_classes"],
            semantic_loss_ignore_index=settings["ignore_index"],
            is_hybrid=False, readout_type="project",
            neck_hidden_sizes=settings["dpt_neck_channels"],
            reassemble_factors=settings["dpt_reassemble_factors"],
            fusion_hidden_size=settings["dpt_fusion_channels"],
            head_in_index=-1, use_auxiliary_head=False,
            use_batch_norm_in_fusion_residual=True,
            semantic_classifier_dropout=settings["dpt_dropout"],
        )
        self.neck = DPTNeck(self.config)
        self.head = DPTSemanticSegmentationHead(self.config)

    def forward(self, features, height, width):
        return self.head(self.neck(features, patch_height=height, patch_width=width))


class CoralDINO(nn.Module):
    def __init__(self, model_config, lora_config, cache_dir, local_files_only=False):
        super().__init__()
        self.model_config = dict(model_config)
        self.lora_config = dict(lora_config)
        if model_config["num_classes"] != 4 or model_config["ignore_index"] != 255:
            raise ValueError("Coral health requires four classes and ignore_index=255.")
        if len(model_config["feature_layers"]) != 4:
            raise ValueError("Use exactly four feature layers for the DPT comparison.")
        source = model_config["id"]
        if local_files_only:
            source = snapshot_download(source, revision=model_config["revision"],
                                       cache_dir=cache_dir, local_files_only=True)
        quantization = model_config.get("quantization")
        load_options = {}
        if quantization:
            if not torch.cuda.is_available():
                raise RuntimeError("This QLoRA recipe requires a CUDA GPU; CPU fallback is disabled.")
            compute_dtype = getattr(torch, quantization["compute_dtype"])
            load_options = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type=quantization["quant_type"],
                    bnb_4bit_use_double_quant=quantization["double_quant"],
                    bnb_4bit_compute_dtype=compute_dtype),
                "device_map": {"": torch.cuda.current_device()}, "dtype": compute_dtype,
            }
        backbone = AutoModel.from_pretrained(
            source, revision=model_config["revision"], cache_dir=cache_dir,
            local_files_only=local_files_only, attn_implementation="sdpa",
            output_hidden_states=True,
            **load_options,
        )
        if quantization:
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
        if model_config.get("encoder_checkpointing"):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
        self.patch_size = backbone.config.patch_size
        self.prefix_tokens = 1 + backbone.config.num_register_tokens
        hidden = backbone.config.hidden_size
        backbone.requires_grad_(False)
        self.backbone = get_peft_model(backbone, LoraConfig(**lora_config))
        self.decoder_type = model_config["decoder"]
        if self.decoder_type == "dpt":
            self.decoder = DPTDecoder(hidden, self.patch_size, model_config)
        elif self.decoder_type == "shallow":
            channels = model_config["head_channels"]
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden * 4, hidden, 1), nn.BatchNorm2d(hidden), nn.GELU(),
                nn.Conv2d(hidden, channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(channels, 4, 1),
            )
        else:
            raise ValueError("MODEL['decoder'] must be dpt or shallow.")
        if model_config.get("decoder_checkpointing"):
            from paper_suite.execution import checkpoint_decoder_module
            checkpoint_decoder_module(self.decoder)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Keep DINOv3's random RoPE rescaling disabled while training LoRA dropout.
        self.backbone.eval()
        if self.model_config.get("encoder_checkpointing"):
            # HF checks each block's training flag for checkpointing. Keep the
            # embedding/RoPE/drop-path modules in eval mode to retain fixed features.
            for layer in self.backbone.get_base_model().layer:
                layer.training = mode
        for module in self.backbone.modules():
            if hasattr(module, "lora_dropout"):
                module.lora_dropout.train(mode)
        return self

    def forward(self, images):
        batch, _, height, width = images.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError("Input height/width must be divisible by the patch size.")
        states = self.backbone(pixel_values=images).hidden_states
        features = []
        # Preserve the existing recipe's hidden-state tuple indices (0 = embedding).
        for layer in self.model_config["feature_layers"]:
            state = states[layer]
            patches = state[:, self.prefix_tokens:]
            if patches.shape[1] != (height // patch) * (width // patch):
                raise ValueError("DINOv3 patch-token count does not match the input grid.")
            if self.decoder_type == "dpt":
                # DPT projects the CLS readout into patches; register tokens are excluded.
                features.append(torch.cat((state[:, :1], patches), dim=1))
            else:
                features.append(patches.transpose(1, 2).reshape(
                    batch, -1, height // patch, width // patch,
                ))
        logits = (self.decoder(features, height // patch, width // patch)
                  if self.decoder_type == "dpt" else self.decoder(torch.cat(features, dim=1)))
        return F.interpolate(logits, (height, width), mode="bilinear", align_corners=False)

    def trainable_state(self):
        state = {name: value.detach().cpu().clone() for name, value in self.named_parameters()
                 if value.requires_grad}
        state.update({name: value.detach().cpu().clone() for name, value in self.named_buffers()
                      if name.startswith("decoder.")})
        return state

    def load_trainable_state(self, state):
        expected = self.trainable_state()
        if set(expected) != set(state):
            raise ValueError(f"Checkpoint trainable keys differ: {set(expected) ^ set(state)}")
        tensors = dict(self.named_parameters()) | dict(self.named_buffers())
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != tensors[name].shape:
                    raise ValueError(f"Checkpoint shape differs for {name}.")
                tensors[name].copy_(value)
