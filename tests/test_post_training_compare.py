"""Checks for gradient isolation, teacher swaps, and cross-dataset scoring."""
import unittest
import torch
from torch import nn

from post_training_compare.adaptation import Teacher, instance_loss
from post_training_compare.evaluate import metrics
from post_training_compare.labels import native_lookup, target_group


class PostTrainingChecks(unittest.TestCase):
    def test_unmapped_predictions_are_errors(self):
        result=metrics([dict(group=1,predicted_group=1),dict(group=2,predicted_group=0)])
        self.assertEqual(result["accuracy"],.5)
        self.assertEqual(result["macro_recall"],.5)
        self.assertEqual(result["classes"]["Dead hard coral"]["recall"],0.)

    def test_explicit_ontology(self):
        lookup=native_lookup()
        self.assertEqual(len(lookup),96)
        self.assertEqual(lookup[0],0)
        self.assertEqual(lookup[14],0)
        self.assertEqual(lookup[91],1)
        self.assertEqual(lookup[88],10)
        self.assertEqual(lookup[4],2)
        self.assertEqual(lookup[23],9)
        self.assertEqual(target_group("Shadow"),0)
        self.assertEqual(target_group("Bleached Hard Coral"),1)

    def test_teacher_swap_restores_after_exception(self):
        model=nn.Module();model.backbone=nn.Module()
        model.backbone.reins=nn.Linear(2,2);model.backbone.core=nn.Linear(2,2)
        model.decode_head=nn.Linear(2,2)
        teacher=Teacher(model)
        with torch.no_grad():model.backbone.reins.weight.add_(1)
        before=model.backbone.reins.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            with teacher.apply(model):
                self.assertTrue(torch.equal(model.backbone.reins.weight,teacher.shadow["backbone.reins.weight"]))
                raise RuntimeError("intentional")
        self.assertTrue(torch.equal(model.backbone.reins.weight,before))
        self.assertNotIn("backbone.core.weight",teacher.shadow)
        teacher.update(model,0)
        self.assertTrue(torch.equal(teacher.shadow["backbone.reins.weight"],before))

    def test_query_distillation_gradients_and_teacher_isolation(self):
        torch.manual_seed(4)
        classes=torch.randn(1,2,97,dtype=torch.float64,requires_grad=True)
        masks=torch.randn(1,2,2,2,dtype=torch.float64,requires_grad=True)
        teacher_classes=torch.full((1,2,97),-5.,dtype=torch.float64)
        teacher_classes[0,0,2]=5.;teacher_classes[0,1,96]=5.
        teacher_classes.requires_grad_(True)
        teacher_masks=torch.randn(1,2,2,2,dtype=torch.float64,requires_grad=True)
        def loss(c,m):return instance_loss([c],[m],teacher_classes,teacher_masks)
        self.assertTrue(torch.autograd.gradcheck(loss,(classes,masks),fast_mode=True))
        loss(classes,masks).backward()
        self.assertIsNone(teacher_classes.grad)
        self.assertIsNone(teacher_masks.grad)
        self.assertTrue(torch.isfinite(classes.grad).all())
        self.assertGreater(float(masks.grad[0,0].abs().sum()),0)
        self.assertEqual(float(masks.grad[0,1].abs().sum()),0)


if __name__=="__main__":unittest.main()
