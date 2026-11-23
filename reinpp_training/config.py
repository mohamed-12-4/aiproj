"""Released Rein++ architecture plus explicit V2 and training adaptations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

COMMIT = "38b688ea7236b0aa58d1186dcaf37a240816bd81"
DATA_REVISION = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"
ENCODER_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
PROJECT = Path(__file__).resolve().parents[1]


def author_root():
    local = PROJECT / "third_party/rein_plus_author/projects/rein++"
    return local if local.exists() else Path("/app")


def model_config():
    root = author_root()
    configs = root / "configs" if (root / "configs").exists() else root / "author_configs"
    released = runpy.run_path(str(configs / "rdg_c2abm_dinov3_rein.py"))
    model = deepcopy(released["model"])
    model["backbone"]["reins_config"].update(type="ReinPPMultiHeadReins", num_heads=4,
                                                 act="gelu", mlp_ratio=.5)
    model["data_preprocessor"]["seg_pad_val"] = 0
    head = model["decode_head"]
    head.update(num_classes=96, ignore_index=0, num_things_classes=0, num_stuff_classes=96)
    head["loss_cls"]["class_weight"] = [1.] * 96 + [.1]
    # The release intentionally omits training configuration. Reuse the authors'
    # published Mask2Former matching and point-loss configuration, unchanged.
    training_source = PROJECT / "third_party/rein_plus_author/configs/_base_/models/rein_dinov2_mask2former.py"
    if not training_source.exists():
        training_source = Path("/app/author_train_head.py")
    original = runpy.run_path(str(training_source))["model"]
    head["train_cfg"] = deepcopy(original["decode_head"]["train_cfg"])
    model["train_cfg"] = {}
    return model


def optimizer_config():
    no_decay = dict(lr_mult=1., decay_mult=0.)
    return dict(optim_wrapper_cfg=dict(type="OptimWrapper", optimizer=dict(
                    type="AdamW", lr=1e-4, weight_decay=.05, eps=1e-8, betas=(.9,.999))),
                paramwise_cfg=dict(custom_keys={"norm":dict(decay_mult=0.), "query_embed":no_decay,
                    "level_embed":no_decay, "learnable_tokens":no_decay, "reins.scale":no_decay},
                    norm_decay_mult=0.))


def train_pipeline():
    return [dict(type="LoadImageFromFile"),dict(type="LoadAnnotations",reduce_zero_label=False,imdecode_backend="pillow"),
            dict(type="RandomChoiceResize",scales=[int(512*x*.1) for x in range(5,21)],
                 resize_type="ResizeShortestEdge",max_size=2048),
            dict(type="RandomCrop",crop_size=(512,512),cat_max_ratio=.75,ignore_index=0),
            dict(type="RandomFlip",prob=.5),dict(type="PhotoMetricDistortion"),dict(type="PackSegInputs")]


def experiment_config():
    return dict(author_commit=COMMIT, dataset_revision=DATA_REVISION, encoder_revision=ENCODER_REVISION,
                model=model_config(), optimizer=optimizer_config(), train_pipeline=train_pipeline(),
                epochs=40,batch_size=4,seed=42,poly_power=.9,poly_horizon=40000,warmup_steps=0,
                ema_decay=.999,ema_kind="effective_token_matrix_and_other_trainable_parameters",
                precision="bf16_backbone_fp32_mask2former",num_workers=8,
                adaptations=["Official released DINOv3-L architecture and query-linked Mask2Former",
                             "User-selected paper multi-head (4 heads) and GELU with half-width MLP",
                             "Native V2 IDs 1..95 supervised, void 0 ignored; 96 semantic outputs plus no-object",
                             "Fresh pretrained encoder, fresh adapter/head, 40 epochs instead of paper 40000 iterations",
                             "Training driver reconstructed; authors have released inference/evaluation only",
                             "Author published optimizer, loss weights, Hungarian matching and data augmentation",
                             "BF16 backbone and FP32 Mask2Former CUDA operators; effective-token EMA added"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)
