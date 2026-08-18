"""
model.py - Siamese NCC-style localizer for reference-in-search matching.

This file is imported by inference.py (`from model import SiamNCCLocalizer`)
and must always match whatever class definition your training notebook
actually saved weights with. If you edit the architecture in the notebook,
copy the change here too -- a mismatch here is exactly what produces
load_state_dict "Missing key(s)/Unexpected key(s)" errors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import BACKBONE_LAYERS

BACKBONE_CHANNELS = [32, 64, 128, 128]


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Backbone(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        layers = []
        c_in = in_channels
        for (k, s, p), c_out in zip(BACKBONE_LAYERS, BACKBONE_CHANNELS):
            layers.append(_ConvBlock(c_in, c_out, k, s, p))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x):
        return self.net(x)


class SiamNCCLocalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone(in_channels=1)
        self.corr_scale = nn.Parameter(torch.tensor(10.0))
        self.corr_bias = nn.Parameter(torch.tensor(0.0))
        self.offset_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def _embed(self, x):
        feat = self.backbone(x)
        return F.normalize(feat, p=2, dim=1, eps=1e-6)

    def _xcorr(self, search_feat, ref_feat):
        B, C, Hs, Ws = search_feat.shape
        Br, Cr, Hr, Wr = ref_feat.shape
        assert B == Br and C == Cr, "reference/search batch or channel mismatch"
        outs = []
        for i in range(B):
            s = search_feat[i:i+1]
            k = ref_feat[i:i+1]
            outs.append(F.conv2d(s, k))
        return torch.cat(outs, dim=0)

    def forward(self, ref, search):
        ref_feat = self._embed(ref)
        search_feat = self._embed(search)
        corr = self._xcorr(search_feat, ref_feat)
        heatmap = corr * self.corr_scale + self.corr_bias
        offset = self.offset_head(corr)
        return heatmap, offset