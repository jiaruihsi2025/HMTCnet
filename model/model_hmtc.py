import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch import Tensor
from math import sqrt
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patches_resolution = img_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class UpScalingModule(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super(UpScalingModule, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, features: List[Tensor]) -> Tensor:
        assert len(features) == 4
        B, num_patches, embed_dim = features[0].shape
        H = W = int(sqrt(num_patches))
        spatial_features = []
        for feat in features:
            spatial_feat = feat.transpose(1, 2).view(B, embed_dim, H, W)
            spatial_features.append(spatial_feat)
        x = torch.stack(spatial_features, dim=-1)
        x = rearrange(x, 'b c h w (s1 s2) -> b c (h s1) (w s2)',
                      s1=self.scale_factor, s2=self.scale_factor)
        return x


class SpatialFeatureEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_layers: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 mode: str = 'serial'):
        super(SpatialFeatureEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.mode = mode
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
            for _ in range(num_layers)
        ])
        self.upscale = UpScalingModule(scale_factor=2)
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        B, num_patches, d_model = x.shape
        x = self.pos_encoding(x)
        intermediate_features = []
        if self.mode == 'serial':
            for i, layer in enumerate(self.encoder_layers):
                x = layer(x)
                intermediate_features.append(x)
        else:
            for i, layer in enumerate(self.encoder_layers):
                x_i = layer(x)
                intermediate_features.append(x_i)
        if self.mode == 'serial':
            features_for_upscale = intermediate_features[-4:]
        else:
            features_for_upscale = intermediate_features[:4]
        hrt_features = self.upscale(features_for_upscale)
        return hrt_features, intermediate_features


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super(CrossModalFusion, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.cross_attention = MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hrt: Tensor, pan: Tensor) -> Tensor:
        attn_output, _ = self.cross_attention(
            query=hrt,
            key=pan,
            value=pan
        )
        x = self.norm1(hrt + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class SFN(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 img_size: int = 64,
                 d_model: int = 64,
                 nhead: int = 2,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 patch_size: int = 8,
                 mode: str = 'serial',
                 target_size: int = 512):
        super(SFN, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.d_model = d_model
        self.patch_size = patch_size
        self.mode = mode
        self.target_size = target_size
        self.num_encoder_layers = num_encoder_layers
        self.lrms_patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.pan_patch_embed = PatchEmbedding(target_size, patch_size, 1, d_model)
        self.encoders = nn.ModuleList()
        current_dim = d_model
        for i in range(num_encoder_layers):
            encoder = SpatialFeatureEncoder(
                d_model=current_dim,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                mode=mode
            )
            self.encoders.append(encoder)
        self.cross_fusion = CrossModalFusion(d_model, nhead)
        self.output_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, in_channels, 3, padding=1)
        )

    def forward(self, lrms: Tensor, pan: Tensor = None) -> Tuple[Tensor, List[Tensor]]:
        B, C, H, W = lrms.shape
        x = self.lrms_patch_embed(lrms)
        all_features = []
        current_features = x
        for i, encoder in enumerate(self.encoders):
            hrt_features, intermediate_features = encoder(current_features)
            all_features.extend(intermediate_features)
            if i < len(self.encoders) - 1:
                B_enc, C_enc, H_enc, W_enc = hrt_features.shape
                current_features = hrt_features.flatten(2).transpose(1, 2)
        if pan is not None:
            pan_patches = self.pan_patch_embed(pan)
            B_hr, C_hr, H_hr, W_hr = hrt_features.shape
            hrt_features_flat = hrt_features.flatten(2).transpose(1, 2)
            if hrt_features_flat.shape[1] != pan_patches.shape[1]:
                hrt_features_flat = F.interpolate(
                    hrt_features_flat.transpose(1, 2),
                    size=pan_patches.shape[1],
                    mode='linear'
                ).transpose(1, 2)
            fused_features = self.cross_fusion(hrt_features_flat, pan_patches)
            H_fused = W_fused = int(sqrt(fused_features.shape[1]))
            hrt_features = fused_features.transpose(1, 2).view(B, -1, H_fused, W_fused)
        if hrt_features.shape[2] != self.target_size:
            hrt_features = F.interpolate(hrt_features, size=(self.target_size, self.target_size),
                                         mode='bilinear', align_corners=True)
        hrt = self.output_proj(hrt_features)
        return hrt, all_features


class TFN(nn.Module):
    def __init__(self,
                 lrms_channels: int = 4,
                 pan_channels: int = 1,
                 hrt_channels: int = 4,
                 base_channels: int = 64,
                 target_size: int = 512):
        super(TFN, self).__init__()
        self.base_channels = base_channels
        self.target_size = target_size
        self.lrms_stream = nn.Sequential(
            nn.Conv2d(lrms_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        self.pan_stream = nn.Sequential(
            nn.Conv2d(pan_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((target_size, target_size)),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        self.hrt_stream = nn.Sequential(
            nn.Conv2d(hrt_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        total_channels = 3 * base_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, lrms_channels, 3, padding=1),
        )

    def forward(self, lrms: Tensor, pan: Tensor, hrt: Tensor) -> Tensor:
        lrms_features = self.lrms_stream(lrms)
        pan_features = self.pan_stream(pan)
        hrt_features = self.hrt_stream(hrt)
        fused_features = torch.cat([lrms_features, pan_features, hrt_features], dim=1)
        output = self.fusion(fused_features)
        return output


class MyModel(nn.Module):
    def __init__(self,
                 lrms_channels: int = 4,
                 pan_channels: int = 1,
                 img_size: int = 64,
                 target_size: int = 512,
                 d_model: int = 64,
                 nhead: int = 2,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 patch_size: int = 8,
                 sfn_mode: str = 'serial',
                 base_channels: int = 64):
        super(MyModel, self).__init__()
        self.target_size = target_size
        self.sfn = SFN(lrms_channels, img_size, d_model, nhead, num_encoder_layers,
                       dim_feedforward, dropout, patch_size, sfn_mode, target_size)
        self.tfn = TFN(lrms_channels, pan_channels, lrms_channels, base_channels, target_size)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(lrms_channels * 2, lrms_channels, 3, padding=1),
            nn.BatchNorm2d(lrms_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(lrms_channels, lrms_channels, 1),
        )

    def forward(self, lrms: Tensor, pan: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        hrt_features, sfn_features = self.sfn(lrms, pan)
        texture_features = self.tfn(lrms, pan, hrt_features)
        combined = torch.cat([hrt_features, texture_features], dim=1)
        final_output = self.final_fusion(combined)
        return hrt_features, texture_features, final_output


if __name__ == "__main__":
    batch_size = 2
    lrms = torch.randn(batch_size, 4, 64, 64)
    pan = torch.randn(batch_size, 1, 512, 512)
    model = MyModel()
    print("模型初始化成功！")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    with torch.no_grad():
        hrt, texture, output = model(lrms, pan)
    print(f"\n输出形状:")
    print(f"HRT features: {hrt.shape}")
    print(f"Texture features: {texture.shape}")
    print(f"Final output: {output.shape}")
    assert output.shape[2] == 512 and output.shape[3] == 512, "输出尺寸不正确"
    print("✓ 所有测试通过！")