"""Paper-component configuration and effective-token EMA regression tests."""
import os
os.environ.setdefault("USE_TF","0")
os.environ.setdefault("USE_FLAX","0")
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib")
from copy import deepcopy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from reinpp_training.config import experiment_config
from reinpp_training.ema import EffectiveTokenEMA
from reinpp_training.metrics import mapped_metrics


class Tokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.learnable_tokens_a=nn.Parameter(torch.full((2,3,1),2.))
        self.learnable_tokens_b=nn.Parameter(torch.full((2,1,4),3.))
    def get_tokens(self,layer):
        all_tokens=self.learnable_tokens_a@self.learnable_tokens_b
        return all_tokens if layer==-1 else all_tokens[layer]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module();self.backbone.reins=Tokens()
        self.decode_head=nn.Linear(1,1,bias=False)
        self.backbone.register_buffer("counter",torch.tensor(0))


class ReinPPTests(unittest.TestCase):
    def test_user_selected_architecture_and_published_optimizer(self):
        c=experiment_config();m=c["model"];r=m["backbone"]["reins_config"]
        self.assertEqual(r["type"],"ReinPPMultiHeadReins")
        self.assertEqual((r["act"],r["mlp_ratio"],r["num_heads"],r["lora_dim"]),("gelu",.5,4,16))
        h=m["decode_head"]
        self.assertEqual(h["type"],"ReinPPReinMask2FormerHead")
        self.assertTrue(h["replace_query_feat"] and h["replace_query_embed"])
        self.assertEqual((h["num_classes"],h["ignore_index"]),(96,0))
        self.assertEqual(len(h["loss_cls"]["class_weight"]),97)
        self.assertEqual([h[k]["loss_weight"] for k in ("loss_cls","loss_mask","loss_dice")],[2.,5.,5.])
        self.assertEqual(h["train_cfg"]["num_points"],12544)
        opt=c["optimizer"];self.assertEqual(opt["optim_wrapper_cfg"]["optimizer"]["lr"],1e-4)
        for key in ("learnable_tokens","reins.scale","norm"):
            self.assertEqual(opt["paramwise_cfg"]["custom_keys"][key]["decay_mult"],0)
        self.assertEqual((c["epochs"],c["batch_size"],c["poly_horizon"]),(40,4,40000))

    def test_ema_averages_effective_tokens_and_restores_raw_parameters(self):
        model=Model();ema=EffectiveTokenEMA(model,decay=.5)
        with torch.no_grad():
            model.backbone.reins.learnable_tokens_a.fill_(4)
            model.backbone.reins.learnable_tokens_b.fill_(5)
        ema.update(model)
        with ema.apply(model):
            torch.testing.assert_close(model.backbone.reins.get_tokens(-1),torch.full((2,3,4),13.))
        torch.testing.assert_close(model.backbone.reins.get_tokens(-1),torch.full((2,3,4),20.))
        self.assertNotIn("backbone.reins.learnable_tokens_a",ema.inference_state(model))
        self.assertIn("backbone.reins.effective_ema_tokens",ema.inference_state(model))

    def test_ema_retains_rank_growth_instead_of_refactorizing(self):
        model=Model();reins=model.backbone.reins
        reins.learnable_tokens_a=nn.Parameter(torch.tensor([[[1.],[0.]]]))
        reins.learnable_tokens_b=nn.Parameter(torch.tensor([[[1.,0.]]]))
        ema=EffectiveTokenEMA(model,decay=.5)
        with torch.no_grad():
            reins.learnable_tokens_a.copy_(torch.tensor([[[0.],[1.]]]))
            reins.learnable_tokens_b.copy_(torch.tensor([[[0.,1.]]]))
        ema.update(model)
        with ema.apply(model):
            torch.testing.assert_close(reins.get_tokens(-1)[0],torch.eye(2)*.5)
            self.assertEqual(int(torch.linalg.matrix_rank(reins.get_tokens(-1)[0])),2)

    def test_ema_resume_and_exception_restore(self):
        model=Model();ema=EffectiveTokenEMA(model,.5)
        with torch.no_grad():model.decode_head.weight.add_(1)
        ema.update(model);before=deepcopy(model.state_dict())
        with self.assertRaises(RuntimeError):
            with ema.apply(model):raise RuntimeError("evaluation failed")
        for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name])
        resumed=EffectiveTokenEMA(model,.1);resumed.load_state_dict(deepcopy(ema.state_dict()))
        ema.update(model);resumed.update(model)
        torch.testing.assert_close(ema.tokens,resumed.tokens)
        self.assertEqual(ema.num_updates,resumed.num_updates)

    def test_health_mapping_matches_existing_protocol(self):
        from paper_suite.runner import mapped_metrics as previous
        c=torch.zeros((96,96),dtype=torch.long)
        c[4,4],c[4,0],c[14,4],c[3,3],c[2,2]=10,2,3,5,7
        actual=mapped_metrics(c);expected=previous(c)["health_4"]
        for key in ("dead","per_class_iou","coral_miou","legacy_dead_recalls","fine_dead_recalls"):
            self.assertEqual(actual[key],expected[key])


if __name__=="__main__":unittest.main()
