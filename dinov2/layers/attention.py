# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,         # ? set to 12
        qkv_bias: bool = False,     # ? set to True
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads    # ? 768 // 12 = 64
        self.scale = head_dim**-0.5    # ? 64^-0.5 = 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)    # ? 768 * 3 = 2304        
        self.attn_drop = nn.Dropout(attn_drop)               # ? this is ignored in evaluation mode
        self.proj = nn.Linear(dim, dim, bias=proj_bias)      # ? 768 to 768
        self.proj_drop = nn.Dropout(proj_drop)               # ? this is ignored in evaluation mode

    def forward(self, x: Tensor) -> Tensor:              # ? x is B, 257, 768
        B, N, C = x.shape       
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # ? B, 257, 768 -> qkv -> B, 257, 2304 -> reshape -> B, 257, 3, 12, 64 -> permute -> 3, B, 12, 257, 64

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]   # ? we could have also done the self.scale later   1, B, 12, 257, 64 each
        attn = q @ k.transpose(-2, -1)                  # ? after this step   -> 1, B, 12, 257, 257

        attn = attn.softmax(dim=-1)                     # ? I don't think it makes a difference if you do softmax at dim = -1 or -2  but need to make sure that the inner product with v is is done with the softmaxed dimension
        attn = self.attn_drop(attn)                     

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # ? 1, B, 12, 257, 64 -> 1, 12, B, 257, 64 -> B, 257, 768
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
