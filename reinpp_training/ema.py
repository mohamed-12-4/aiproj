"""Exact EMA of effective low-rank Rein tokens, plus the trained head and MLPs."""
from contextlib import contextmanager
import torch


class EffectiveTokenEMA:
    def __init__(self, model, decay=.999):
        self.decay, self.num_updates = float(decay), 0
        self.module = model.backbone.reins
        self.excluded = {"backbone.reins.learnable_tokens_a", "backbone.reins.learnable_tokens_b"}
        self.names = [n for n,p in model.named_parameters() if p.requires_grad and n not in self.excluded]
        self.shadow = {n:p.detach().float().clone() for n,p in model.named_parameters() if n in self.names}
        self.tokens = self.module.get_tokens(-1).detach().float().clone()
        self.buffers = {n:b.detach().clone() for n,b in model.named_buffers()}

    @torch.no_grad()
    def update(self, model):
        for name,param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach().float(), 1-self.decay)
        # Averaging A and B separately would not produce EMA(A @ B).
        self.tokens.lerp_(self.module.get_tokens(-1).detach().float(), 1-self.decay)
        for name,buffer in model.named_buffers():
            self.buffers[name].copy_(buffer)
        self.num_updates += 1

    @contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        saved = {n:parameters[n].detach().clone() for n in self.names}
        saved_buffers = {n:b.detach().clone() for n,b in buffers.items()}
        original = self.module.get_tokens
        def get_tokens(layer):
            return self.tokens if layer == -1 else self.tokens[layer]
        try:
            with torch.no_grad():
                for name,value in self.shadow.items():
                    parameters[name].copy_(value)
                for name,value in self.buffers.items():
                    buffers[name].copy_(value)
            self.module.get_tokens = get_tokens
            yield model
        finally:
            self.module.get_tokens = original
            with torch.no_grad():
                for name,value in saved.items():
                    parameters[name].copy_(value)
                for name,value in saved_buffers.items():
                    buffers[name].copy_(value)

    def state_dict(self):
        return dict(kind="effective_token_matrix_ema",decay=self.decay,num_updates=self.num_updates,
                    shadow=self.shadow,tokens=self.tokens,buffers=self.buffers)

    def load_state_dict(self,state):
        if state["kind"] != "effective_token_matrix_ema" or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA checkpoint architecture mismatch")
        self.decay,self.num_updates = state["decay"],state["num_updates"]
        self.tokens.copy_(state["tokens"])
        for name,value in state["shadow"].items():
            self.shadow[name].copy_(value)
        for name,value in state["buffers"].items():
            self.buffers[name].copy_(value)

    def inference_state(self,model):
        with self.apply(model):
            state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                     if n.startswith(("backbone.reins.","decode_head.")) and n not in self.excluded}
            state["backbone.reins.effective_ema_tokens"] = self.tokens.detach().cpu().clone()
        return state
