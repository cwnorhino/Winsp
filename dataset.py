"""
v2 change from the previous dataset: targets are now built on the VALID
correlation grid (226x226 for a 1000px search / 100px reference), using the
exact pixel<->feature mapping in geometry.py, instead of assuming the output
is a padded, same-size (250x250) grid at a flat "stride 4". That assumption
was never actually wrong in isolation (the padded model did produce a 250x250
map) -- the problem was the padding itself, which is what let the model cheat
(see model.py's docstring). Removing the padding means the target grid has to
shrink and shift to match, which is what target_grid_coord() computes.

Everything else here (per-image standardization, INTER_AREA downscale of the
reference, reading pre-generated pairs from disk) is unchanged from v1 and was
never implicated in the failure.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import geometry

REF_PX = 100     # reference image side length after the known 10x shrink
SEARCH_PX = 1000


class WinspPairDataset(Dataset):
    def __init__(self, root_dir, heatmap_sigma: float = 1.5):
        self.root = root_dir
        with open(os.path.join(root_dir, "ground_truth.json")) as f:
            self.records = json.load(f)
        self.sigma = heatmap_sigma
        self.grid_size = geometry.valid_corr_size(SEARCH_PX, REF_PX)  # 226

    def __len__(self):
        return len(self.records)

    def _load_pair(self, rec):
        pid = rec["pair_id"]
        ref = cv2.imread(os.path.join(self.root, "reference", f"pair_{pid:04d}_ref.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(self.root, "search", f"pair_{pid:04d}_search.png"), cv2.IMREAD_GRAYSCALE)
        return ref, search

    def __getitem__(self, idx):
        rec = self.records[idx]
        ref, search = self._load_pair(rec)

        ref_small = cv2.resize(ref, (REF_PX, REF_PX), interpolation=cv2.INTER_AREA)
        ref_t = self._to_tensor(ref_small)
        search_t = self._to_tensor(search)

        gt_x = rec["center_x_wide_px"]
        gt_y = rec["center_y_wide_px"]
        heatmap, offset, offset_mask, valid = self._make_targets(gt_x, gt_y)

        return {
            "reference": ref_t,
            "search": search_t,
            "heatmap": heatmap,
            "offset": offset,
            "offset_mask": offset_mask,
            "target_in_grid": valid,   # False for the rare example whose GT falls
                                        # outside the valid grid -- see note below
            "gt_x": gt_x,
            "gt_y": gt_y,
            "hard_case": rec.get("hard_case", False),
            "pair_id": rec["pair_id"],
        }

    def _to_tensor(self, img):
        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img).unsqueeze(0)

    def _make_targets(self, gt_x, gt_y):
        g = self.grid_size
        cx = geometry.target_grid_coord(gt_x, REF_PX)
        cy = geometry.target_grid_coord(gt_y, REF_PX)
        cx_int, cy_int = int(np.floor(cx)), int(np.floor(cy))

        yy, xx = np.mgrid[0:g, 0:g]
        heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2)).astype(np.float32)

        offset = np.zeros((2, g, g), dtype=np.float32)
        offset_mask = np.zeros((1, g, g), dtype=np.float32)
        valid = 0 <= cx_int < g and 0 <= cy_int < g
        if valid:
            offset[0, cy_int, cx_int] = cx - cx_int
            offset[1, cy_int, cx_int] = cy - cy_int
            offset_mask[0, cy_int, cx_int] = 1.0
        # NOTE: generate_dataset.py's choose_target() keeps every GT at least
        # 54px from the search-image border, comfortably inside the ~48.5px
        # margin the valid grid requires, so `valid` should be True for every
        # example the current generator produces. It's checked (not assumed)
        # here so a future generator change that violates that margin fails
        # loudly in training stats rather than silently corrupting targets.

        return (torch.from_numpy(heatmap).unsqueeze(0),
                torch.from_numpy(offset),
                torch.from_numpy(offset_mask),
                valid)