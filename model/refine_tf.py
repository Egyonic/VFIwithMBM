## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from model.biformer_models.biformer import Block as BiformerBlock
from model.efficientvit.models.nn.ops import EfficientViTBlock


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False, stride=1):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=stride, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, in_channel, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channel, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=49,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 4, 4, 2],
                 num_refinement_blocks=2,

                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.c_down1 = nn.Conv2d(64 + 1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.down1_2 = Downsample(dim, dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1 + 32), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.c_down2 = nn.Conv2d(128 + 1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.down2_3 = Downsample(dim * 2 + 32, dim * 2)  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4 + 64), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.c_down3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.down3_4 = Downsample(int(dim * 4 + 64), dim * 4)  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8 + 128), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8 + 128)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(dim * 8 + 128, dim * 4 + 64, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4 + 64), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4 + 64)  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(dim * 4 + 64, dim * 2 + 32, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 + 32), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2 + 32)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 + 16 + dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 + 16 + dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(dim * 1 + 16 + dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide):
        inp_img = torch.cat((img0, img1, mask, mask_guide[0], warped_img0, warped_img1, c0[0], c1[0], flow), 1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        f1 = self.c_down1(torch.cat([mask_guide[1], c0[1], c1[1]], 1))
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(torch.cat([inp_enc_level2, f1], 1))

        f2 = self.c_down2(torch.cat([mask_guide[2], c0[2], c1[2]], 1))
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(torch.cat([inp_enc_level3, f2], 1))

        f3 = self.c_down3(torch.cat([c0[3], c1[3]], 1))
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(torch.cat([inp_enc_level4, f3], 1))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        return torch.sigmoid(out_dec_level1)


class Restormer_BRA(nn.Module):
    def __init__(self,
                 inp_channels=49,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 4, 4, 2],
                 num_refinement_blocks=2,

                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer_BRA, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            BiformerBlock(
                dim=dim, n_win=14, num_heads=heads[0], kv_downsample_mode="ada_avgpool", kv_per_win=2, topk=6,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[0])])

        self.c_down1 = nn.Conv2d(64 + 1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.down1_2 = Downsample(dim, dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 2 + 32, n_win=7, num_heads=heads[1], kv_downsample_mode="ada_avgpool", kv_per_win=1, topk=4,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[1])])

        self.c_down2 = nn.Conv2d(128 + 1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.down2_3 = Downsample(dim * 2 + 32, dim * 2)  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 4 + 64, n_win=7, num_heads=heads[2], kv_downsample_mode="ada_avgpool", kv_per_win=1, topk=4,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[2])])

        self.c_down3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.down3_4 = Downsample(int(dim * 4 + 64), dim * 4)  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 8 + 128, n_win=7, num_heads=heads[3], kv_downsample_mode="ada_avgpool", kv_per_win=1, topk=4,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8 + 128)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(dim * 8 + 128, dim * 4 + 64, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 4 + 64, n_win=7, num_heads=heads[0], kv_downsample_mode="ada_avgpool", kv_per_win=1, topk=4,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4 + 64)  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(dim * 4 + 64, dim * 2 + 32, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 2 + 32, n_win=7, num_heads=heads[0], kv_downsample_mode="ada_avgpool", kv_per_win=1, topk=4,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2 + 32)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 1 + 16 + dim, n_win=7, num_heads=heads[0], kv_downsample_mode="ada_avgpool", kv_per_win=2, topk=4,
                mlp_ratio=2, auto_pad=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            BiformerBlock(
                dim=dim * 1 + 16 + dim, n_win=14, num_heads=heads[0], kv_downsample_mode="ada_avgpool", kv_per_win=2, topk=6,
                mlp_ratio=2, auto_pad=True) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(dim * 1 + 16 + dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide):
        inp_img = torch.cat((img0, img1, mask, mask_guide[0], warped_img0, warped_img1, c0[0], c1[0], flow), 1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        f1 = self.c_down1(torch.cat([mask_guide[1], c0[1], c1[1]], 1))
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(torch.cat([inp_enc_level2, f1], 1))

        f2 = self.c_down2(torch.cat([mask_guide[2], c0[2], c1[2]], 1))
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(torch.cat([inp_enc_level3, f2], 1))

        f3 = self.c_down3(torch.cat([c0[3], c1[3]], 1))
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(torch.cat([inp_enc_level4, f3], 1))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        return torch.sigmoid(out_dec_level1)


class Restormer_EffcientVit(nn.Module):
    def __init__(self,
                 inp_channels=49,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 4, 4, 2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=1,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 bias=False,
                 ):
        super(Restormer_EffcientVit, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.c_down1 = nn.Conv2d(64 + 1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.down1_2 = Downsample(dim, dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1 + 32), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.c_down2 = nn.Conv2d(128 + 1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.down2_3 = Downsample(dim * 2 + 32, dim * 2)  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            EfficientViTBlock(in_channels=dim * 4 + 64, dim=dim, heads_ratio=1.0, expand_ratio=2.0)
            for i in range(num_blocks[2])])

        self.c_down3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.down3_4 = Downsample(int(dim * 4 + 64), dim * 4)  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            EfficientViTBlock(in_channels=dim * 8 + 128, dim=dim, heads_ratio=1.0, expand_ratio=2.0)
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8 + 128)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(dim * 8 + 128, dim * 4 + 64, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            EfficientViTBlock(in_channels=dim * 4 + 64, dim=dim, heads_ratio=1.0, expand_ratio=2.0)
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4 + 64)  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(dim * 4 + 64, dim * 2 + 32, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 + 32), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2 + 32)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 + 16 + dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 + 16 + dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim * 1 + 16 + dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide):
        inp_img = torch.cat((img0, img1, mask, mask_guide[0], warped_img0, warped_img1, c0[0], c1[0], flow), 1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        f1 = self.c_down1(torch.cat([mask_guide[1], c0[1], c1[1]], 1))
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(torch.cat([inp_enc_level2, f1], 1))

        f2 = self.c_down2(torch.cat([mask_guide[2], c0[2], c1[2]], 1))
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(torch.cat([inp_enc_level3, f2], 1))

        f3 = self.c_down3(torch.cat([c0[3], c1[3]], 1))
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(torch.cat([inp_enc_level4, f3], 1))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return torch.sigmoid(out_dec_level1)



if __name__ == "__main__":
    # flownet = IFNet_bf_resnet_local_mae()
    flownet = Restormer_EffcientVit(inp_channels=49,out_channels=3, dim=32)
    input = torch.rand(1, 17, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flownet.to(device)
    input_cuda = input.to(device)
    output = flownet(input_cuda)
    print('finish')
