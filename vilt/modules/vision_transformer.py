""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
from cgi import print_arguments
import math
import logging
from functools import partial
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import urllib
import warnings

from functools import partial
from tqdm import tqdm

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained,load_pretrained_Moe
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.resnetv2 import ResNetV2
from timm.models.registry import register_model
from torchvision import transforms

_logger = logging.getLogger(__name__)


def download_clip(
    url: str = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    root: str = os.path.expanduser("~/.cache/clip"),
):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


inception_unnormalize = transforms.Compose(
    [UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models (my experiments)
    "vit_small_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
    ),
    # patch models (weights ported from official Google JAX impl)
    "vit_base_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_base_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    # patch models, imagenet21k (weights ported from official Google JAX impl)
    "vit_base_patch16_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_huge_patch14_224_in21k": _cfg(
        url="",  # FIXME I have weights for this but > 2GB limit for github release binaries
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    # hybrid models (weights ported from official Google JAX impl)
    "vit_base_resnet50_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=0.9,
        first_conv="patch_embed.backbone.stem.conv",
    ),
    "vit_base_resnet50_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
        first_conv="patch_embed.backbone.stem.conv",
    ),
    # hybrid models (my experiments)
    "vit_small_resnet26d_224": _cfg(),
    "vit_small_resnet50d_s3_224": _cfg(),
    "vit_base_resnet26d_224": _cfg(),
    "vit_base_resnet50d_224": _cfg(),
    # deit models (FB weights)
    "vit_deit_tiny_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    ),
    "vit_deit_small_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    ),
    "vit_deit_base_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    ),
    "vit_deit_base_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_deit_tiny_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth"
    ),
    "vit_deit_small_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    ),
    "vit_deit_base_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
    ),
    "vit_deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # self.mlp_1 = Mlp(
        #     in_features=dim,
        #     hidden_features=mlp_hidden_dim,
        #     act_layer=act_layer,
        #     drop=drop,
        # )
        # self.mlp_Uni = Mlp(
        #     in_features=dim,
        #     hidden_features=mlp_hidden_dim,
        #     act_layer=act_layer,
        #     drop=drop,
        # )

    def forward(self, x):
        _x, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

        # if y == None:
        #     _x, attn = self.attn(self.norm1(x), mask=mask)
        #     x = x + self.drop_path(_x)
        #     if type == "image":
        #         x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     elif type == "text":
        #         x = x + self.drop_path(self.mlp_1(self.norm2(x)))
        #     elif type == "Uni":
        #         x = x + self.drop_path(self.mlp_Uni(self.norm2(x)))
        #     return x, attn
        # else:
        #     _x, _y = self.attn(self.norm1(x),y=self.norm1(y), y_kv=self.norm1(y_kv), mask=mask)
        #     x = x + self.drop_path(_x)
        #     y = y + self.drop_path(_y)
        #     if type == "image":
        #         x = x + self.drop_path(self.mlp_Uni(self.norm2(x)))
        #         y = y + self.drop_path(self.mlp_Uni(self.norm2(y)))
        #     else:
        #         x = x + self.drop_path(self.mlp_Uni(self.norm2(x)))
        #         y = y + self.drop_path(self.mlp_Uni(self.norm2(y)))
        #     return x, y

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)   
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)
        self.gamma_2_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k):
        # print(x_q.shape)
        # print(x_kv.shape)
        # print(pos_q.shape)
        # print(pos_k.shape)
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q),
         k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        add_norm_before_transformer=False,
        no_patch_embed_bias=False,
        config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        drop_rate = drop_rate if config is None else config["drop_rate"]
        decoder_depth = 8
        self.real_size = (24,8)
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # trunc_normal_(self.decoder_embed, std=0.02)
        # MAE decoder specifics
        decoder_embed_dim = 512
        decoder_num_heads = 16
        decoder_depth = 8
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 192 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.norm_pix_loss = False
        # --------------------------------------------------------------------------
        self.decoder = nn.ModuleList(
            [
                RegressorBlock(
                    dim=embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )
        self.fc_mask = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm_mask = nn.LayerNorm(decoder_embed_dim)
        self.fc_visual= nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm_visual= nn.LayerNorm(decoder_embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = self.build_2d_sincos_position_embedding(self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def build_2d_sincos_position_embedding(self, embed_dim, temperature=10000.):
        h, w = self.real_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = torch.cat([pe_token, pos_emb], dim=1)
        return pos_embed

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def mask_tokens(self, orig_image, feats):
        """
        Prepare masked tokens inputs/labels for masked patch prediction: 80% MASK, 10% random, 10% original.
        """
        img_unnorm = orig_image * 0.5 + 0.5
        _, _, ph, pw = self.patch_embed.proj.weight.shape
        with torch.no_grad():
            img_unnorm_patch = F.conv2d(
                img_unnorm,
                weight=torch.ones(3, 1, ph, pw).to(img_unnorm) / (ph * pw),
                bias=None,
                stride=(ph, pw),
                padding=0,
                groups=3,
            )
        labels = (
            ((img_unnorm_patch * 255).long().flatten(start_dim=2, end_dim=3))
            .permute(0, 2, 1)
            .contiguous()
        )

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape[:-1], 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape[:-1], 0.8)).bool() & masked_indices
        )
        feats[indices_replaced] = self.mask_token.to(feats)

        return feats, labels

    def visual_embed(self, _x, max_image_len=200, mask_it=False):
        # s_1 = time.time()
        _, _, ph, pw = self.patch_embed.proj.weight.shape

        x = self.patch_embed(_x)
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :]
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        # s_2 = time.time()
        # print('get_mask:',s_2-s_1)
        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        # s_3 = time.time()
        # print('get_pos_embed:',s_3-s_2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
                ),
                dim=-1,
            )[None, None, :, :, :]
            .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
            .flatten(1, 3)
        )
        x_mask = x_mask.flatten(1)
        # s_4 = time.time()
        # print('get_patch_index:',s_4-s_3)
        if mask_it:
            x, label = self.mask_tokens(_x, x)
        # s_5 = time.time()
        # print('mask_it:',s_5-s_4)
        if (
            max_image_len < 0
            or max_image_len is None
            or not isinstance(max_image_len, int)
        ):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            eff = x_h * x_w
            max_image_len = eff.max()
        else:
            eff = x_h * x_w
            max_image_len = min(eff.max(), max_image_len)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [
            non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
        ]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_len - v for v in valid_nums]
        # s_6 = time.time()
        # print('no_name:',s_6-s_5)
        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float().to(x), max_image_len)
                # print(valid_choice.device)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(
                    torch.ones(nv).float().to(x), p, replacement=True
                )
                # print(pad_choice.device)
                select.append(
                    torch.cat(
                        [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
                    )
                )
        
        # s_7 = time.time()
        # print('no_name_2:',s_7-s_6)
        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(B, -1, C)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, C)

        if mask_it:
            label = label[select[:, 0], select[:, 1]].view(B, -1, 3)

            label[x_mask == 0] = -100
            label = torch.cat(
                [torch.full((label.shape[0], 1, 3), -100).to(label), label,], dim=1,
            )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.pos_embed[:, 0, :][:, None, :].expand(B, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)
        # s_8 = time.time()
        # print('final:',s_8-s_7)
        if mask_it:
            return x, x_mask, (patch_index, (H, W)), label
        else:
            return x, x_mask, (patch_index, (H, W)), None


    def forward_features(self, _x, max_image_len=144, mask_it=False):
        x, x_mask, patch_index, label = self.visual_embed(
            _x, max_image_len=max_image_len, mask_it=mask_it
        )

        for blk in self.blocks:
            x, _ = blk(x, mask=x_mask)

        x = self.norm(x)
        return x, x_mask, label

    def forward(self, x, max_image_len=-1):
        x, _, _ = self.forward_features(x, max_image_len=max_image_len)
        x = x[:, 0]
        x = self.head(x)
        return x

    #开始MAE
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep,ids_mask

    def forward_encoder_mae(self,_x,mask_ratio):
        x = self.patch_embed(_x)
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :]
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        x = x + pos_embed
        x, mask, ids_restore,ids_keep,ids_mask = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for i, blk in enumerate(self.blocks):
            x, _attn = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore,ids_keep,ids_mask

    def forward_decoder_mae(self,x,ids_restore):
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for i,blk in enumerate(self.decoder_blocks):
            x,_ = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        # print(imgs.shape)
        # print(self.patch_size)
        # print(self.stride_size)
        assert imgs.shape[2] % p == 0

        h =  imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    def forward_loss_mae(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_encoder_nomask(self,_x):
        x = self.patch_embed(_x)
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :]
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        x = x + pos_embed
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for i, blk in enumerate(self.blocks):
            x, _attn = blk(x)
        x = self.norm(x)
        return x


    def forward_encoder_cae(self,_x,mask_ratio):
        x = self.patch_embed(_x)
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :]
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        x = x + pos_embed
        x, mask,  ids_restore, ids_keep, ids_mask = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for i, blk in enumerate(self.blocks):
            x, _attn = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore, ids_keep, ids_mask


    def forward_decoder_cae(self,x_visual,ids_restore,ids_keep,ids_mask):
        x_mask = self.mask_token.repeat(x_visual.shape[0], ids_restore.shape[1]+1 - x_visual.shape[1], 1)
        pos_emb_all = self.decoder_pos_embed.expand(ids_keep.size(0), -1, -1)
        pos_emb = pos_emb_all[:,1:]
        D = pos_emb.size(2)
        pos_emb_visual = torch.gather(pos_emb, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        pos_emb_visual = torch.cat([pos_emb_all[:,:1],pos_emb_visual],dim=1)
        pos_emb_mask = torch.gather(pos_emb, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))

        for blk in self.decoder:
            x_mask = blk(x_mask, torch.cat([x_visual, x_mask], dim=1), pos_emb_mask, torch.cat([pos_emb_visual, pos_emb_mask], dim=1))
        latent_mask = self.fc_mask(x_mask)
        latent_mask = self.norm_mask(latent_mask)

        latent_visual = self.fc_visual(x_visual)
        latent_visual = self.norm_visual(latent_visual)

        return latent_mask, latent_visual

    
        
    def forward_loss_cae(self,latent_mask,latent_visual,target_mask,target_visual):
        loss_retraining = F.mse_loss(latent_mask, target_mask, reduction="mean")
        loss_distill = F.mse_loss(latent_visual, target_visual, reduction="mean")
        return loss_retraining + loss_distill



        


class DistilledVisionTransformer(VisionTransformer):
    """ Vision Transformer with distillation token.

    Paper: `Training data-efficient image transformers & distillation through attention` -
        https://arxiv.org/abs/2012.12877

    This impl of distilled ViT is taken from https://github.com/facebookresearch/deit
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def visual_embed(self, _x, max_image_len=200, mask_it=False):
        _, _, ph, pw = self.patch_embed.proj.weight.shape

        x = self.patch_embed(_x)
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :]
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 2:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
                ),
                dim=-1,
            )[None, None, :, :, :]
            .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
            .flatten(1, 3)
        )
        x_mask = x_mask.flatten(1)

        if mask_it:
            x, label = self.mask_tokens(_x, x)

        if (
            max_image_len < 0
            or max_image_len is None
            or not isinstance(max_image_len, int)
        ):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            eff = x_h * x_w
            max_image_len = eff.max()
        else:
            eff = x_h * x_w
            max_image_len = min(eff.max(), max_image_len)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [
            non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
        ]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_len - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_len)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(
                    torch.ones(nv).float(), p, replacement=True
                )
                select.append(
                    torch.cat(
                        [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
                    )
                )

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(B, -1, C)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, C)
        if mask_it:
            label = label[select[:, 0], select[:, 1]].view(B, -1, 3)

            label[x_mask == 0] = -100
            label = torch.cat(
                [torch.full((label.shape[0], 1, 3), -100).to(label), label,], dim=1,
            )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        pos_embed = torch.cat(
            (self.pos_embed[:, :2, :].expand(B, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 2).to(x_mask), x_mask], dim=1)

        if mask_it:
            return x, x_mask, (patch_index, (H, W)), label
        else:
            return x, x_mask, (patch_index, (H, W)), None

    def forward_features(self, _x, max_image_len=144, mask_it=False):
        x, x_mask, patch_index, label = self.visual_embed(
            _x, max_image_len=max_image_len, mask_it=mask_it
        )

        for blk in self.blocks:
            x, _ = blk(x, mask=x_mask)

        x = self.norm(x)
        return x, x_mask, label

    def forward(self, x, max_image_len=-1):
        x, _, _ = self.forward_features(x, max_image_len=max_image_len)
        x = x[:, 0]
        x = self.head(x)
        return x


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-1]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = DistilledVisionTransformer if distilled else VisionTransformer
    model = model_cls(
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        **kwargs,
    )
    model.default_cfg = default_cfg

    if pretrained:
        # load_pretrained(
        #     model,
        #     num_classes=num_classes,
        #     in_chans=kwargs.get("in_chans", 3),
        #     filter_fn=partial(checkpoint_filter_fn, model=model),
        #     strict=False,
        # )
        load_pretrained_Moe(
            model,
            num_classes=num_classes,
            in_chans=kwargs.get("in_chans", 3),
            filter_fn=partial(checkpoint_filter_fn, model=model),
            strict=False,
        )
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ My custom 'small' ViT model. Depth=8, heads=8= mlp_ratio=3."""
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3.0,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault("qk_scale", 768 ** -0.5)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_huge_patch14_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet50_224_in21k(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid model from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9),
        num_classes=0,
        global_pool="",
        in_chans=kwargs.get("in_chans", 3),
        preact=False,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )
    model_kwargs = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        hybrid_backbone=backbone,
        representation_size=768,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_resnet50_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet50_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9),
        num_classes=0,
        global_pool="",
        in_chans=kwargs.get("in_chans", 3),
        preact=False,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_resnet50_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[4],
    )
    model_kwargs = dict(
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_small_resnet26d_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet50D 3-stages, stride 16. No pretrained weights.
    """
    backbone = resnet50d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[3],
    )
    model_kwargs = dict(
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_small_resnet50d_s3_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[4],
    )
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_resnet26d_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = resnet50d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[4],
    )
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_resnet50d_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_tiny_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_small_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_distilled_patch16_384",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model

