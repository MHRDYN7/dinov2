# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,  # ? initialize with 518
        patch_size: Union[int, Tuple[int, int]] = 16, # ? initialize with 14
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)   # ? (518, 518)
        patch_HW = make_2tuple(patch_size) # ? (14, 14)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],    # ? 518 // 14 = 37
            image_HW[1] // patch_HW[1],    # ? 518 // 14 = 37
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size    # ? (37, 37) but not used
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]   # ? 37 * 37 = 1369

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding       # ? not needed

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)    # ? kernal and stride are (14, 14)  
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()    # ? norm_layer is none so nn.Identity()   not needed probably

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape                # ? H, W = 224, 224 as x is after transformation
        patch_H, patch_W = self.patch_size  # ? patch_H, patch_W = 14, 14

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"  # ? 224 % 14 = 0
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"   # ? 224 % 14 = 0

        x = self.proj(x)  # B C H W    # ? x shape is (B, 768, 16, 16)
        H, W = x.size(2), x.size(3)    # ? H, W = 16, 16
        x = x.flatten(2).transpose(1, 2)  # B HW C   # ? after flatten x is (B, 768, 256) and after transpose x is (B, 256, 768)
        x = self.norm(x)   # ? x shape is (B, 256, 768)
        if not self.flatten_embedding:    # ? False
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x           # ? x shape is s

    def flops(self) -> float:             # ? this func is never called
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
