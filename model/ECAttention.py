"""ECAttention module containing the Expand and Compress class."""

import math
from functools import partial
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.mlp import Mlp
from timm.layers.weight_init import trunc_normal_
from timm.models import register_model

from utils import cholesky_orthogonalization, cholesky_orthogonalization_QR


class PositionalEncodingFourier(nn.Module):
    """Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.SyncBatchNorm(out_planes),
    )


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise ("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = (
            x.shape[2],
            x.shape[3],
        )  # (H // self.patch_size[0], W // self.patch_size[1]), value will be used in positional encoding

        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        return x, (Hp, Wp)


class ClassAttention(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

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
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        qc = q[:, :, 0:1]  # CLS token
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        cls_tkn = self.proj(cls_tkn)
        x = torch.cat([self.proj_drop(cls_tkn), x[:, 1:]], dim=1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

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
        eta=None,
        tokens_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # FIXME: A hack for models pre-trained with layernorm over all the tokens not just the CLS
        self.tokens_norm = tokens_norm

    def forward(self, x, H, W, mask=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x[:, 0:1] = self.norm2(x[:, 0:1])

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x


class Expand(nn.Module):
    """Expand module for ECAttention."""

    def __init__(
        self,
        dim: int,
        heads: int = 6,
        dim_head: int = 64,
        subspace_rank: int = 20,
        num_tokens: int = 196,
        withoutChol: bool = False,
        eps: float = 0.01,
    ) -> None:
        """Initialize the Expand module.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            subspace_rank (int): Rank of the subspace for the expansion operator.

        """
        super().__init__()
        self.heads = heads
        self.dim = dim  # d
        self.dim_head = dim_head
        # self.subspace_rank = subspace_rank
        self.rank_of_space = subspace_rank * heads
        self.current_epoch = 0
        self._epoch_changed = False
        self.default_n = num_tokens
        self.withoutChol = withoutChol
        self.eps = eps  # cholesky_orthogonalization epsilon

        assert self.rank_of_space <= dim, (
            "Estimated rank for the whole space of expansion operator must be less than or equal to"
            "dim of input tensor"
        )
        self.register_buffer(
            "Omega", torch.randn(self.default_n, self.rank_of_space), persistent=True
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for regeneration of random matrices."""
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self._epoch_changed = True
        else:
            self._epoch_changed = False

    def _regenerate_random_matrices(
        self, n: int, device: Union[str, torch.device, None] = None
    ) -> None:
        """Regenerate random matrices."""
        if device is None:
            device = self.Omega.device
        if self.Omega.size(0) != n:
            self.Omega = torch.randn(n, self.rank_of_space, device=device)
        else:
            self.Omega.data.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Expand module.

        Args:
            x (torch.Tensor): Input tensor of shape (b, n, d), where b is the batch size,
                n is the number of tokens, and d is the dimension of the model.

        Returns:
            torch.Tensor: Output tensor of shape (b, n, d) after applying the null space projection.

        """
        b, n, d = x.shape
        assert d == self.dim, "Input tensor dim must be equal to the dim of the model"
        orig_dtype = x.dtype
        x_t = x.transpose(-1, -2)  # [b, d, n]
        if (
            (self.Omega is None)
            or self.Omega.size(0) != n
            or self._epoch_changed
            or self.Omega.device != x.device
        ):
            self._regenerate_random_matrices(n, x.device)
        Y = torch.matmul(x_t, self.Omega)
        Y = Y.float()
        if self.withoutChol:  # only for inference time and peak memory measurement
            Q = Y
        else:
            Q = cholesky_orthogonalization(Y, eps=self.eps)
        # Q = Y  # just for memory and time measurement.
        Q = Q.to(orig_dtype)
        QQT = torch.matmul(Q, Q.transpose(1, 2))
        null_proj = x_t - torch.matmul(QQT, x_t)
        result = null_proj.transpose(-1, -2)
        return result  # (b, n, d)





class Compress(nn.Module):
    """Compress module for ECAttention."""

    def __init__(
        self,
        dim: int,
        heads: int = 6,
        dim_head: int = 64,
        subspace_rank: int = 20,
        debug: bool = False,
        num_tokens: int = 196,
        return_Us: bool = False,
        withoutChol: bool = False,
        eps: float = 0.01,
    ) -> None:
        """Initialize the Compress module.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            subspace_rank (int): Rank of the subspace for the compression operator.

        """
        super().__init__()
        self.heads = heads
        self.dim = dim  # d
        self.dim_head = dim_head
        self.subspace_rank = subspace_rank
        self.current_epoch = 0
        self._epoch_changed = False
        self.default_n = num_tokens

        self.save_attention = False
        self.attention_scores = None
        self.return_Us = return_Us
        self.withoutChol = withoutChol
        self.eps = eps  # cholesky_orthogonalization epsilon

        assert subspace_rank <= dim_head, (
            "Estimated rank for the subspace of compression operator must be less than "
            "or equal to dim of each head"
        )

        self.Us = nn.Parameter(torch.randn(dim, dim_head * heads), requires_grad=True)
        if debug:
            torch.nn.init.orthogonal_(
                self.Us
            )  # Initialize Us in main.py  instead of here if debug is False
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)

        self.register_buffer(
            "Omega", torch.randn(self.default_n, self.subspace_rank), persistent=True
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for regeneration of random matrices."""
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self._epoch_changed = True
        else:
            self._epoch_changed = False

    def _regenerate_random_matrix(
        self, n: int, device: Union[str, torch.device, None] = None
    ) -> None:
        """Regenerate the random matrix Omega."""
        if device is None:
            device = self.Omega.device
        if self.Omega.size(0) != n:
            self.Omega = torch.randn(n, self.subspace_rank, device=device)
        else:
            self.Omega.data.normal_()

    # def forward(self, x: torch.Tensor) -> "tuple[torch.Tensor, nn.Parameter]":
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Compress module.

        Args:
            x (torch.Tensor): Input tensor of shape (b, n, d), where b
                is the batch size, n is the number of tokens, and d is the dimension of the model.

        Returns:
            torch.Tensor: Output tensor of shape (b, n, d) after applying the null
                space projection and compression.

        """
        orig_dtype = x.dtype
        b, n, d = x.shape
        assert d == self.dim, "Input tensor dim must be equal to the dim of the model"

        x_t = x.transpose(1, 2)  # [b, d, n]
        Us_T = rearrange(self.Us, "d (h p) -> h p d", h=self.heads)
        alphas = torch.einsum("hpd,bdn->bhpn", Us_T, x_t)
        if (
            self.Omega is None
            or self.Omega.size(0) != n
            or self._epoch_changed
            or self.Omega.device != x.device
        ):
            self._regenerate_random_matrix(n, x.device)

        Qs = []
        for i in range(self.heads):
            alphas_i = alphas[:, i, :, :]
            Y = torch.einsum("bpn,nc->bpc", alphas_i, self.Omega)
            Y = Y.float()
            # Q, _ = torch.linalg.qr(Y, mode="reduced")
            if self.withoutChol:  # only for inference time and peak memory measurement
                Q = Y
            else:
                Q = cholesky_orthogonalization(Y, eps=self.eps)
            # Q = Y  # just for memory and time measurement.
            Q = Q.to(orig_dtype)
            Qs.append(Q)
        Qs = torch.stack(Qs, dim=1)
        QQT = torch.matmul(Qs, Qs.transpose(-1, -2))
        col_proj = torch.einsum("bhpp,bhpn->bhpn", QQT, alphas)
        null_proj = alphas - col_proj
        col_proj_for_norm = rearrange(col_proj, "b h p n -> b n h p")
        norms = torch.linalg.norm(col_proj_for_norm, ord=2, dim=-1)
        softmax_scores = F.softmax(norms / self.temperature, dim=-1)

        # Save attention scores if required, e.g., for visualization
        if self.save_attention:
            self.attention_scores = softmax_scores.detach().cpu()

        w_null_proj = torch.einsum("bnh,bhpn->bhpn", softmax_scores, null_proj)
        w_null_proj_re = rearrange(w_null_proj, "b h p n -> b (h p) n")
        Us_w_null_proj = torch.matmul(self.Us, w_null_proj_re)  # (b, d, n)

        if self.return_Us:
            return Us_w_null_proj.transpose(-1, -2), Us_T  # for Toy data experiments.
        # Since Us_T is not used, returning it may cause  error.
        #  "making sure all forward function outputs participate in calculating loss. "
        return Us_w_null_proj.transpose(-1, -2)







class ECAttention(nn.Module):
    """ECAttention module combining Expand and Compress operators."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        # dim_head: int = 64,
        subspace_rank: int = 20,
        num_tokens: int = 196,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        withoutChol=False,
        eps=0.01,
    ):
        super().__init__()
        """Initialize the ECAttention module.
        Args:
            dim (int): Dimension of the input tensor.
            num_heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            subspace_rank (int): Rank of each subspace.
        """
        dim_head = dim // num_heads
        assert subspace_rank <= dim_head, (
            "Estimated rank for the subspace of expansion and compression operators must be less than "
            "or equal to dim of each head"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.subspace_rank = subspace_rank
        self.dim_head = dim_head
        self.eps = eps

        self.expandoperator = Expand(
            dim,
            heads=self.num_heads,
            dim_head=self.dim_head,
            subspace_rank=self.subspace_rank,
            num_tokens=num_tokens,
            withoutChol=withoutChol,
            eps=self.eps,
        )
        self.compressoperator = Compress(
            dim,
            heads=self.num_heads,
            dim_head=self.dim_head,
            subspace_rank=self.subspace_rank,
            num_tokens=num_tokens,
            withoutChol=withoutChol,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """Forward pass of the ECAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (b, n, d), where b is the batch size,
                n is the number of tokens, and d is the dimension of the model.

        Returns:
            tuple: A tuple containing:
                - expand (torch.Tensor): Output tensor after applying the expansion operator.
                - compress (torch.Tensor): Output tensor after applying the compression operator.

        """
        assert x.dim() == 3, "Input tensor must be of shape (b, n, d)"
        assert x.shape[2] == self.dim, (
            "Input tensor dim must be equal to the dim of the model"
        )
        # b, n, d = x.shape
        expand = self.expandoperator(x)
        # compress, Us = self.compressoperator(x)
        compress = self.compressoperator(x)
        return expand, compress













class Block_constrainWeight(nn.Module):
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
        attn_layer=ECAttention,
        num_tokens=196,
        eta=None,
        eta2=None,
        subspace_rank=20,
        withoutChol=False,
        eps=0.01,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            subspace_rank=subspace_rank,
            num_tokens=num_tokens,
            withoutChol=withoutChol,
            eps=eps,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # if attn_layer != ECAttention:
        #     self.mlp = Mlp(
        #         in_features=dim,
        #         hidden_features=mlp_hidden_dim,
        #         act_layer=act_layer,
        #         drop=drop,
        #     )

        # self.gamma1 = nn.Parameter(torch.tensor(eta2), requires_grad=True)
        # self.gamma2 = nn.Parameter(torch.tensor(eta2), requires_grad=True)
        eta2_value = eta2 if eta2 is not None else 0.1
        self._gamma2_raw = nn.Parameter(
            torch.log(torch.exp(torch.tensor(eta2_value)) - 1.0), requires_grad=True
        )
        self._gamma1_raw = nn.Parameter(
            torch.log(torch.exp(torch.tensor(eta2_value)) - 1.0), requires_grad=True
        )

    @property
    def gamma2(self):
        return F.softplus(self._gamma2_raw)

    @property
    def gamma1(self):
        return F.softplus(self._gamma1_raw)

    def forward(self, x, H, W):
        expand, compress = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma1 * expand - self.gamma2 * compress)
        return x
    
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
        attn_layer=ECAttention,
        num_tokens=196,
        eta=None,
        eta2=None,
        subspace_rank=20,
        withoutChol=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            subspace_rank=subspace_rank,
            num_tokens=num_tokens,
            withoutChol=withoutChol,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # if attn_layer != ECAttention:
        #     self.mlp = Mlp(
        #         in_features=dim,
        #         hidden_features=mlp_hidden_dim,
        #         act_layer=act_layer,
        #         drop=drop,
        #     )

        self.gamma1 = nn.Parameter(torch.tensor(eta2), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.tensor(eta2), requires_grad=True)

    def forward(self, x, H, W):
        expand, compress = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma1 * expand - self.gamma2 * compress)
        return x








class VisionTransformer_constrainWeight(nn.Module):
    """Vision Transformer with ECAttention blocks."""

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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        attn_layer=ECAttention,
        cls_attn_layers=2,
        use_pos=True,
        patch_proj="linear",
        eta=None,
        eta2=0.1,
        tokens_norm=False,
        subspace_rank=20,
        withoutChol=False,
        eps=0.01,
        **kwargs,
    ):
        """Vision Transformer with ECAttention blocks.

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
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            eta2: (float) initialization value for expansion and compression operations' cofficients
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA

        """
        super().__init__()
        kwargs.pop("pretrained_cfg", None)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.subspace_rank = subspace_rank

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, embed_dim=embed_dim, patch_size=patch_size
        )
        # self.patch_embed = ConvPatchEmbed_crate(
        #     img_size=img_size, embed_dim=embed_dim, patch_size=patch_size
        # )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.eps = eps

        print("Attention mechanism: ", attn_layer)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                Block_constrainWeight(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_layer=attn_layer,
                    num_tokens=num_patches,
                    eta2=eta2,
                    subspace_rank=self.subspace_rank,
                    withoutChol=withoutChol,
                    eps=self.eps,
                )
                for i in range(depth)
            ]
        )

        self.cls_attn_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    eta=eta,
                    tokens_norm=tokens_norm,
                )
                for i in range(cls_attn_layers)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)
        self.use_pos = use_pos

        # Classifier head
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, Compress):
            self._init_param_weights(m.Us, method="orthogonal")
            # self._init_param_weights(m.Us, method="truncated_normal")
            # self._init_param_weights(m.Us, method="kaiming_normal")

    def _init_param_weights(self, param, method="orthogonal"):
        if method == "orthogonal":
            nn.init.orthogonal_(param)
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif method == "kaiming_normal":
            nn.init.kaiming_normal_(param, a=math.sqrt(5))
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(param)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(param)
        elif method == "truncated_normal":
            trunc_normal_(param, std=0.02)
        else:
            print(f"Warning: Unknown Us init method '{method}', using orthogonal")
            nn.init.orthogonal_(param)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for regeneration of random matrices in Compress and Expand."""
        for blk in self.blocks:
            blk.attn.expandoperator.set_epoch(epoch)
            blk.attn.compressoperator.set_epoch(epoch)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos:
            pos_encoding = (
                self.pos_embeder(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            )
            x = x + pos_encoding

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)

        x = self.norm(x)[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with ECAttention blocks."""

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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        attn_layer=ECAttention,
        cls_attn_layers=2,
        use_pos=True,
        patch_proj="linear",
        eta=None,
        eta2=0.1,
        tokens_norm=False,
        subspace_rank=20,
        withoutChol=False,
        **kwargs,
    ):
        """Vision Transformer with ECAttention blocks.

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
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            eta2: (float) initialization value for expansion and compression operations' cofficients
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA

        """
        super().__init__()
        kwargs.pop("pretrained_cfg", None)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.subspace_rank = subspace_rank

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, embed_dim=embed_dim, patch_size=patch_size
        )
        # self.patch_embed = ConvPatchEmbed_crate(
        #     img_size=img_size, embed_dim=embed_dim, patch_size=patch_size
        # )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        print("Attention mechanism: ", attn_layer)

        dpr = [drop_path_rate for i in range(depth)]
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
                    attn_layer=attn_layer,
                    num_tokens=num_patches,
                    eta2=eta2,
                    subspace_rank=self.subspace_rank,
                    withoutChol=withoutChol,
                )
                for i in range(depth)
            ]
        )

        self.cls_attn_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    eta=eta,
                    tokens_norm=tokens_norm,
                )
                for i in range(cls_attn_layers)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)
        self.use_pos = use_pos

        # Classifier head
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, Compress):
            self._init_param_weights(m.Us, method="orthogonal")
            # self._init_param_weights(m.Us, method="truncated_normal")
            # self._init_param_weights(m.Us, method="kaiming_normal")

    def _init_param_weights(self, param, method="orthogonal"):
        if method == "orthogonal":
            nn.init.orthogonal_(param)
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif method == "kaiming_normal":
            nn.init.kaiming_normal_(param, a=math.sqrt(5))
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(param)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(param)
        elif method == "truncated_normal":
            trunc_normal_(param, std=0.02)
        else:
            print(f"Warning: Unknown Us init method '{method}', using orthogonal")
            nn.init.orthogonal_(param)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for regeneration of random matrices in Compress and Expand."""
        for blk in self.blocks:
            blk.attn.expandoperator.set_epoch(epoch)
            blk.attn.compressoperator.set_epoch(epoch)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos:
            pos_encoding = (
                self.pos_embeder(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            )
            x = x + pos_encoding

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)

        x = self.norm(x)[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x











# def ECAttention_tiny_layer12_p16(num_classes=1000):
@register_model
def ECAttention_tiny_layer12_p16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model






@register_model
def ECAttention_small_layer12_p16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model


@register_model
def ECAttention_small_layer12_p16_rank20_constrainweight(pretrained=False, **kwargs):
    model = VisionTransformer_constrainWeight(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model





@register_model
def ECAttention_small_layer12_p16_H6(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model











@register_model
def ECAttention_medium_layer24_p16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=512,
        depth=24,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model









    return model


@register_model
def ECAttention_large_layer24_p16_d1024(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model


@register_model
def ECAttention_large_layer24_p16_L48(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=512,
        depth=48,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model




@register_model
def ECAttention_large_layer48_p16_rank20_constrainweight(pretrained=False, **kwargs):
    model = VisionTransformer_constrainWeight(
        patch_size=16,
        embed_dim=512,
        depth=48,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model









@register_model
def ECAttention_large_layer24_p16_L48_d1024(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=48,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model


@register_model
def ECAttention_large_layer24_p16_L36_d1024(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=36,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        **kwargs,
    )
    return model





@register_model
def ECAttention_small_layer12_p16_rank20_constrainweight_eps_1e_2(
    pretrained=False, **kwargs
):
    model = VisionTransformer_constrainWeight(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        eps=1e-2,
        **kwargs,
    )
    return model


@register_model
def ECAttention_small_layer12_p16_rank20_constrainweight_eps_1e_1(
    pretrained=False, **kwargs
):
    model = VisionTransformer_constrainWeight(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        eta2=0.1,
        subspace_rank=20,
        tokens_norm=True,
        attn_layer=ECAttention,
        eps=1e-1,
        **kwargs,
    )
    return model

