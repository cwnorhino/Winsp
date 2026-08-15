
from __future__ import annotations

BACKBONE_LAYERS = [(5, 1, 2), (3, 1, 1), (3, 2, 1), (3, 2, 1)]


def conv_out(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


def feat_size(input_size: int) -> int:
    n = input_size
    for k, s, p in BACKBONE_LAYERS:
        n = conv_out(n, k, s, p)
    return n


def receptive_field_geometry():

    start, jump = 0.5, 1
    for k, s, p in BACKBONE_LAYERS:
        start = start + ((k - 1) / 2 - p) * jump
        jump *= s
    return start, jump


FEAT_START, FEAT_STRIDE = receptive_field_geometry()


def px_to_feat(px: float) -> float:
    return (px - FEAT_START) / FEAT_STRIDE


def feat_to_px(feat_idx: float) -> float:
    return FEAT_START + FEAT_STRIDE * feat_idx


def valid_corr_size(search_px: int, ref_px: int) -> int:
   
    return feat_size(search_px) - feat_size(ref_px) + 1


def target_grid_coord(gt_px: float, ref_px: int) -> float:

    kh = feat_size(ref_px)
    return px_to_feat(gt_px) - (kh // 2)


def grid_coord_to_px(j: float, ref_px: int) -> float:
    kh = feat_size(ref_px)
    return feat_to_px(j + (kh // 2))