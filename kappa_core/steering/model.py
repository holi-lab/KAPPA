import torch as t
import torch.nn as nn
from typing import Optional

# --------------------------
# Base
# --------------------------
class SteeringModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def _unit_random(self, d: int) -> t.Tensor:
        v = t.randn(d, device=self.device)
        return v / (v.norm(p=2) + 1e-12)

    def forward(self, x):
        raise NotImplementedError
    
    @classmethod
    def from_state_dict(cls, state):
        raise NotImplementedError("Must be implemented in subclasses.")

# --------------------------
# KAPPA
# --------------------------
class KAPPAModule(SteeringModuleBase):
    """
    desired_pred_proj = desired_neg if (x·answer_vec)<0 else desired_pos
    """
    def __init__(
        self,
        answer_mat: Optional[t.Tensor] = None,
        pred_mat: Optional[t.Tensor] = None,
        answer_thresh: Optional[float] = None,
        pred_thresh: Optional[float] = None,
        w:  Optional[t.Tensor] = None,
        beta: Optional[t.Tensor] = None,
        hidden_dim: Optional[int] = None,  # for lazy initialization
        n_classes: int = 4,
    ):
        super().__init__()
        self.register_buffer("answer_mat",    None)
        self.register_buffer("pred_mat",      None)
        self.register_buffer("answer_thresh", None)
        self.register_buffer("pred_thresh",   None)
        self.register_buffer("w",             None)
        self.register_buffer("beta",          None)

        if hidden_dim is not None:
            # if hidden_dim is provided, initialize the vectors immediately
            self._lazy_init(n_classes, hidden_dim)

        for name, val in [("answer_mat", answer_mat), ("pred_mat", pred_mat),
                          ("answer_thresh", answer_thresh), ("pred_thresh", pred_thresh),
                          ("w", w), ("beta", beta)]:
            if val is not None:
                setattr(self, name, val.to(self.device))

    def _lazy_init(self, n_classes:int, hidden_dim: int):
        N = n_classes
        d = hidden_dim

        # --- matrices (N, d) ---
        if self.answer_mat is None:
            # (N, hidden_dim)
            self.answer_mat = t.stack([self._unit_random(d) for _ in range(N)], dim=0)
        if self.pred_mat is None:
            self.pred_mat = t.stack([self._unit_random(d) for _ in range(N)], dim=0)

        # --- thresholds ---
        # default scalar 0 → treat as (N,)
        if self.answer_thresh is None:
            self.answer_thresh = t.zeros(N, device=self.device)
        if self.pred_thresh is None:
            self.pred_thresh = t.zeros(N, device=self.device)

        # --- weights ---
        if self.w is None:
            self.w = t.tensor(1.0, device=self.device)
        if self.beta is None:
            self.beta = t.tensor(0.0, device=self.device)


    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        hidden = x[:, -1, :]  # (batch, hidden_dim)

        dtype, device = hidden.dtype, hidden.device

        # answer_mat, pred_mat: (N, hidden_dim)
        answer_mat = self.answer_mat.to(dtype=dtype, device=device)
        pred_mat   = self.pred_mat.to(dtype=dtype, device=device)

        # thresh, w, beta: Scalar or (N,)
        def _prep_param(p):
            p = p.to(dtype=dtype, device=device)
            if p.ndim == 0:
                # scalar -> (1, 1) -> broadcasting with (batch, N)
                return p.view(1, 1)
            elif p.ndim == 1:
                # (N,) -> (1, N)
                return p.view(1, -1)
            else:
                raise ValueError(f"Unexpected ndim for param: {p.ndim}")

        answer_thresh = _prep_param(self.answer_thresh)
        pred_thresh   = _prep_param(self.pred_thresh)
        w             = _prep_param(self.w)
        beta          = _prep_param(self.beta)

        # hidden: (batch, d)
        # answer_mat: (N, d) -> answer_mat.T: (d, N)
        # ans_proj: (batch, N)
        ans_proj  = hidden @ answer_mat.T
        pred_proj = hidden @ pred_mat.T   # (batch, N)

        # apply element-wise to (batch, N)
        desired_pred_proj = w * (ans_proj - answer_thresh)  # (batch, N)
        desired_pred_proj += t.where(
            ans_proj < answer_thresh,
            -beta,   # (1, N) broadcast
            beta,
        )
        desired_pred_proj += pred_thresh  # (batch, N)

        delta_proj = desired_pred_proj - pred_proj  # (batch, N)

        # weighted summation of pred_mat[n] with delta_proj[:, n] as weights
        # delta_proj: (batch, N), pred_mat: (N, d)
        # Result: (batch, d)
        steering_vector = delta_proj @ pred_mat

        return steering_vector


    @classmethod
    def from_state_dict(cls, state):
        n_classes = state["answer_mat"].shape[0]
        hidden_dim = state["answer_mat"].shape[1]
        obj = cls(n_classes=n_classes, hidden_dim=hidden_dim)
        obj.load_state_dict(state)
        return obj