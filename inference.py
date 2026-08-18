from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import geometry
from model import SiamNCCLocalizer

REF_PX = 100          # size the reference crop is resized to before the backbone
                       # MUST match whatever WinspPairDataset resizes the reference to
                       # when it calls geometry.target_grid_coord() to build training targets.
TOL_PX = 8.0


def to_tensor(img):
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def find_local_maxima(hm, rel_thresh=0.85):
    t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)
    keep = (t == pooled).squeeze().numpy()

    peak_val = hm.max()
    floor_val = hm.min()
    margin = (peak_val - floor_val) * (1.0 - rel_thresh)
    thresh = peak_val - margin

    ys, xs = np.where(keep & (hm >= thresh))
    candidates = [(int(y), int(x), float(hm[y, x])) for y, x in zip(ys, xs)]

    if not candidates:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        candidates = [(int(y), int(x), float(hm[y, x]))]

    return candidates


def localize(model, ref_img, search_img, device, debug=False):
    ref_small = cv2.resize(ref_img, (REF_PX, REF_PX), interpolation=cv2.INTER_AREA)
    ref_t = to_tensor(ref_small).to(device)
    search_t = to_tensor(search_img).to(device)
    with torch.no_grad():
        heatmap, offset = model(ref_t, search_t)
    hm = heatmap[0, 0].cpu().numpy()
    off = offset[0].cpu().numpy()

    candidates = find_local_maxima(hm)
    h_img, w_img = search_img.shape
    center = np.array([w_img / 2.0, h_img / 2.0])

    scored = []
    for (cy, cx, score) in candidates:
        # Confirmed against WinspPairDataset._make_targets: channel 0 = dx
        # (offset[0] = cx - cx_int), channel 1 = dy (offset[1] = cy - cy_int).
        dx, dy = off[0, cy, cx], off[1, cy, cx]

        # geometry.grid_coord_to_px already adds the kh//2 window-center
        # shift and applies the correct receptive-field stride/offset for
        # THIS specific backbone (derived from BACKBONE_LAYERS) - don't
        # hand-roll a stride constant here, use the shared source of truth.
        x = geometry.grid_coord_to_px(cx + dx, REF_PX)
        y = geometry.grid_coord_to_px(cy + dy, REF_PX)
        d = float(np.hypot(x - center[0], y - center[1]))
        scored.append((x, y, score, d))

    # Rank by SCORE first to find the genuinely best match(es). Only among
    # candidates near-tied with the top score do we break the tie by
    # center-distance, per spec ("if more than one matching region is found,
    # return the one closest to the center"). Previously this sorted ALL
    # candidates - including weak/spurious ones that merely cleared the loose
    # rel_thresh cutoff - by center-distance first, which could hand back a
    # noise peak just for being geographically central, regardless of how
    # strong a match it actually was. On a noisy/ambiguous heatmap (like the
    # periodic-fin case) that's a real difference.
    scored.sort(key=lambda r: r[2], reverse=True)
    top_score = scored[0][2]
    TIE_TOLERANCE = 0.05
    tied = [s for s in scored if s[2] >= top_score - TIE_TOLERANCE]
    tied.sort(key=lambda r: r[3])
    best_x, best_y, best_score, _ = tied[0]

    if debug:
        n_out_of_bounds = sum(1 for (x, y, *_r) in scored if not (0 <= x <= w_img and 0 <= y <= h_img))
        print(f"[localize debug] n_candidates={len(scored)} n_out_of_bounds={n_out_of_bounds} "
              f"best=({best_x:.1f},{best_y:.1f}) img=({w_img}x{h_img})")
        if n_out_of_bounds > 0:
            print("  -> predictions landing outside the search image is a strong signal "
                  "of a coordinate-convention mismatch between training targets and this "
                  "inference code (check WinspPairDataset's target-building code).")

    near_ties = [s for s in scored if s[2] >= best_score - 0.05]
    distinct_ties = []
    for s in near_ties:
        if all(np.hypot(s[0] - t[0], s[1] - t[1]) > 20.0 for t in distinct_ties):
            distinct_ties.append(s)

    return {
        "x": best_x, "y": best_y, "score": best_score,
        "n_peaks_total": len(candidates),
        "n_distinct_ambiguous_matches": len(distinct_ties),
        "heatmap": hm,
    }


def evaluate(test_dir, checkpoint, device="cuda", debug=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = SiamNCCLocalizer().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    with open(os.path.join(test_dir, "ground_truth.json")) as f:
        records = json.load(f)

    results = []
    for rec in records:
        pid = rec["pair_id"]
        ref = cv2.imread(os.path.join(test_dir, "reference", f"pair_{pid:04d}_ref.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(test_dir, "search", f"pair_{pid:04d}_search.png"), cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()
        out = localize(model, ref, search, device, debug=debug)
        dt = time.perf_counter() - t0

        err = float(np.hypot(out["x"] - rec["center_x_wide_px"], out["y"] - rec["center_y_wide_px"]))
        results.append({
            "pair_id": pid, "error_px": err, "time_sec": dt,
            "hard_case": rec.get("hard_case", False),
            "n_distinct_ambiguous_matches": out["n_distinct_ambiguous_matches"],
        })

    errs = np.array([r["error_px"] for r in results])
    times = np.array([r["time_sec"] for r in results])
    hard_mask = np.array([r["hard_case"] for r in results])

    summary = {
        "n": len(results),
        "mean_err_px": float(errs.mean()),
        "median_err_px": float(np.median(errs)),
        "success_at_tol_pct": float(100 * np.mean(errs <= TOL_PX)),
        "success_normal_pct": float(100 * np.mean(errs[~hard_mask] <= TOL_PX)) if (~hard_mask).any() else None,
        "success_hard_pct": float(100 * np.mean(errs[hard_mask] <= TOL_PX)) if hard_mask.any() else None,
        "mean_time_ms": float(times.mean() * 1000),
        "ambiguous_frac": float(np.mean([r["n_distinct_ambiguous_matches"] > 1 for r in results])),
    }
    return results, summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--checkpoint", default="driftsense_v2_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug", action="store_true")
    args, _ = ap.parse_known_args()
    results, summary = evaluate(args.test_dir, args.checkpoint, args.device, debug=args.debug)
    print(json.dumps(summary, indent=2))