from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

REF_PX = 1000
SEARCH_PX = 1000
REF_NM_PER_PX = 1.0
SEARCH_NM_PER_PX = 10.0
REF_SPAN_NM = REF_PX * REF_NM_PER_PX
SEARCH_SPAN_NM = SEARCH_PX * SEARCH_NM_PER_PX
SEARCH_SS = 1
SEED_OFFSET = 7919


@dataclass
class Gate:
    y_nm: float
    width_nm: float
    soft_nm: float
    contrast: float
    bow_nm: float
    bow_period_nm: float


@dataclass
class Layout:
    pitch_nm: float
    fin_frac: float
    phase_nm: float
    edge_soft_nm: float
    ler_amp_nm: float
    lwr_amp_nm: float
    bg_val: float
    fin_val: float
    gate_val: float
    gates: list[Gate]
    field_seeds: tuple[int, int]


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lowfreq_field(rng, shape, sigma, grid=8):
    h, w = shape
    small = rng.normal(0.0, sigma, (grid, grid)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def make_layout(rng, hard_case=False):
    pitch = float(rng.uniform(27.0, 37.0))
    n_gates = 1 if rng.random() < 0.58 else 2
    gates = []

    if not hard_case:
        ys = np.sort(rng.uniform(900.0, SEARCH_SPAN_NM - 900.0, n_gates))
    else:
        # Gates still exist elsewhere in the search field; the target crop is
        # selected later so that it contains no gate landmark.
        ys = np.sort(rng.uniform(700.0, SEARCH_SPAN_NM - 700.0, n_gates))

    for y in ys:
        gates.append(Gate(
            y_nm=float(y),
            width_nm=float(rng.uniform(180.0, 290.0)),
            soft_nm=float(rng.uniform(5.0, 9.0)),
            contrast=float(rng.uniform(0.38, 0.58)),
            bow_nm=float(rng.uniform(0.0, 2.0)),
            bow_period_nm=float(rng.uniform(3500.0, 9000.0)),
        ))

    # Keep the fins high-contrast enough to survive 10 nm/px sampling.
    return Layout(
        pitch_nm=pitch,
        fin_frac=float(rng.uniform(0.42, 0.52)),
        phase_nm=float(rng.uniform(0.0, pitch)),
        edge_soft_nm=float(rng.uniform(1.0, 2.0)),
        ler_amp_nm=float(rng.uniform(0.5, 1.7)),
        lwr_amp_nm=float(rng.uniform(0.35, 1.25)),
        bg_val=float(rng.uniform(0.14, 0.20)),
        fin_val=float(rng.uniform(0.68, 0.82)),
        gate_val=float(rng.uniform(0.34, 0.46)),
        gates=gates,
        field_seeds=(int(rng.integers(0, 2**31 - 1)), int(rng.integers(0, 2**31 - 1))),
    )


def build_roughness(layout: Layout):
    # Per-fin edge roughness: each fin gets its own longitudinal stochastic
    # trajectory. This avoids the fake horizontal banding produced when every
    # fin shares one global y-dependent displacement field.
    n_fins = int(np.ceil(SEARCH_SPAN_NM / layout.pitch_nm)) + 6
    yk = np.linspace(-500.0, SEARCH_SPAN_NM + 500.0, 192, dtype=np.float32)
    r1 = np.random.default_rng(layout.field_seeds[0])
    r2 = np.random.default_rng(layout.field_seeds[1])
    ler = r1.normal(0.0, layout.ler_amp_nm, (yk.size, n_fins)).astype(np.float32)
    lwr = r2.normal(0.0, layout.lwr_amp_nm, (yk.size, n_fins)).astype(np.float32)
    # Low-pass the roughness along the fin direction while retaining stochastic
    # high-frequency variation.
    kernel = cv2.getGaussianKernel(9, 1.0)
    ler = cv2.filter2D(ler, -1, kernel)
    lwr = cv2.filter2D(lwr, -1, kernel)
    return yk, ler, lwr

def gates_intersect_window(layout: Layout, y_nm, half_span_nm, buffer_nm=30.0):
    y0 = y_nm - half_span_nm - buffer_nm
    y1 = y_nm + half_span_nm + buffer_nm
    return any(
        g.y_nm + g.width_nm * 0.5 >= y0 and
        g.y_nm - g.width_nm * 0.5 <= y1
        for g in layout.gates
    )


def render_physical(x_nm, y_nm, layout: Layout, roughness, pixel_nm):
    yk, ler_table, lwr_table = roughness
    x = np.asarray(x_nm, dtype=np.float32)
    y = np.asarray(y_nm, dtype=np.float32)

    # Absolute fin index makes the physical roughness identical in reference
    # and search views, including across the large tiled field.
    u = (x - layout.phase_nm) / layout.pitch_nm
    fin_idx = np.floor(u).astype(np.int32) + ler_table.shape[1] // 2
    fin_idx = np.clip(fin_idx, 1, ler_table.shape[1] - 2)
    yf = np.clip((y - yk[0]) / (yk[1] - yk[0]), 0.0, ler_table.shape[0] - 1.001)
    y0 = np.floor(yf).astype(np.int32)
    y1 = y0 + 1
    a = (yf - y0).astype(np.float32)

    ler0 = ler_table[y0, fin_idx]
    ler1 = ler_table[y1, fin_idx]
    lwr0 = lwr_table[y0, fin_idx]
    lwr1 = lwr_table[y1, fin_idx]
    ler = ler0 * (1.0 - a) + ler1 * a
    lwr = lwr0 * (1.0 - a) + lwr1 * a

    mod = np.mod(x - layout.phase_nm - ler, layout.pitch_nm)
    width = layout.pitch_nm * layout.fin_frac + lwr
    edge = max(layout.edge_soft_nm, pixel_nm * 0.18)
    d = width * 0.5 - np.abs(mod - layout.pitch_nm * 0.5)
    fin = smoothstep(d, -edge, edge)

    img = layout.bg_val + (layout.fin_val - layout.bg_val) * fin

    for g in layout.gates:
        bow = g.bow_nm * np.sin(2.0 * np.pi * x / g.bow_period_nm)
        gy0 = g.y_nm - g.width_nm * 0.5 + bow
        gy1 = g.y_nm + g.width_nm * 0.5 + bow
        soft = max(g.soft_nm, pixel_nm * 0.20)
        gate = smoothstep(y, gy0, gy0 + soft) * (1.0 - smoothstep(y, gy1 - soft, gy1))
        img = img * (1.0 - g.contrast * gate) + layout.gate_val * g.contrast * gate

    return np.clip(img, 0.0, 1.0).astype(np.float32)

def edge_response(img, rng, strength):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 0.7)
    p99 = np.percentile(mag, 99.5) + 1e-6
    angle = np.arctan2(gy, gx)
    detector = rng.uniform(0.0, 2.0 * np.pi)
    directional = 0.78 + 0.22 * np.cos(angle - detector)
    return np.clip(img + strength * np.clip(mag / p99, 0.0, 1.0) * directional, 0.0, 1.0), detector


def apply_acquisition(img, rng, reference):
    img = img.astype(np.float32)
    params = {}

    # Mild edge response followed by a feature-preserving optical/beam PSF.
    ew = rng.uniform(0.025, 0.055) if reference else rng.uniform(0.035, 0.075)
    img, az = edge_response(img, rng, ew)
    params["edge_weight"] = float(ew)
    params["detector_azimuth_rad"] = float(az)

    sigma = rng.uniform(0.30, 0.55) if reference else rng.uniform(0.45, 0.78)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma * rng.uniform(0.94, 1.06))
    params["psf_sigma"] = float(sigma)

    # Independent multiplicative illumination and detector gain fields.
    illum_sigma = rng.uniform(0.004, 0.010) if reference else rng.uniform(0.006, 0.014)
    gain_sigma = rng.uniform(0.002, 0.005) if reference else rng.uniform(0.003, 0.008)
    illum = np.exp(lowfreq_field(rng, img.shape, illum_sigma, 7) - 0.5 * illum_sigma**2)
    gain = np.exp(lowfreq_field(rng, img.shape, gain_sigma, 13) - 0.5 * gain_sigma**2)
    img = np.clip(img * illum * gain, 0.0, 1.0)
    params["illumination_sigma"] = float(illum_sigma)
    params["detector_gain_sigma"] = float(gain_sigma)

    # SEM-like correlated surface/detector texture at multiple spatial scales.
    hi_amp = rng.uniform(0.0015, 0.004) if reference else rng.uniform(0.0025, 0.006)
    mid_amp = rng.uniform(0.001, 0.003) if reference else rng.uniform(0.002, 0.005)
    hi = rng.normal(0.0, hi_amp, img.shape).astype(np.float32)
    mid = cv2.GaussianBlur(rng.normal(0.0, 1.0, img.shape).astype(np.float32), (0, 0), rng.uniform(1.2, 2.5))
    mid /= mid.std() + 1e-6
    img = np.clip(img + hi + mid_amp * mid, 0.0, 1.0)
    params["correlated_noise_amp"] = float(mid_amp)

    # Charging is a search-only acquisition effect.
    charging_count = 0
    if not reference and rng.random() < 0.28:
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
        charging_count = int(rng.integers(1, 3))
        for _ in range(charging_count):
            cx = rng.uniform(0, img.shape[1])
            cy = rng.uniform(0, img.shape[0])
            sx = rng.uniform(25, 75)
            sy = rng.uniform(35, 110)
            amp = rng.uniform(-0.012, 0.018)
            blob = amp * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
            img = np.clip(img + blob, 0.0, 1.0)
    params["charging_blobs"] = charging_count

    # Compound Gamma-Poisson count model. Gamma gives controlled overdispersion;
    # Poisson retains discrete electron-counting statistics. Reference has more
    # counts and less overdispersion than the lower-magnification search image.
    electrons = rng.uniform(800.0, 1200.0) if reference else rng.uniform(500.0, 800.0)
    gamma_shape = rng.uniform(220.0, 420.0) if reference else rng.uniform(120.0, 260.0)
    latent_gain = rng.gamma(gamma_shape, 1.0 / gamma_shape, img.shape).astype(np.float32)
    lam = np.clip(img * electrons * latent_gain, 0.0, None)
    counts = rng.poisson(lam).astype(np.float32)
    out = counts / electrons
    params["electron_scale"] = float(electrons)
    params["gamma_shape"] = float(gamma_shape)

    # Small electronics floor; intentionally not the dominant texture.
    read_sigma = rng.uniform(0.001, 0.003) if reference else rng.uniform(0.002, 0.005)
    out += rng.normal(0.0, read_sigma, out.shape).astype(np.float32)
    params["readout_sigma"] = float(read_sigma)

    # Tiny quantization-like perturbation before uint8 conversion.
    q = 1.0 / 255.0
    out = np.clip(np.round(np.clip(out, 0.0, 1.0) / q) * q, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8), params


def drift_field(shape, rng, amp_nm):
    h, w = shape
    gx = np.linspace(0, 1, 9, dtype=np.float32)
    gy = np.linspace(0, 1, 9, dtype=np.float32)
    dx_small = rng.normal(0, amp_nm * 0.35, (9, 9)).astype(np.float32)
    dy_small = rng.normal(0, amp_nm * 0.35, (9, 9)).astype(np.float32)
    dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return dx, dy


def choose_target(layout, rng, hard_case):
    half = REF_SPAN_NM * 0.5
    lo, hi = half + 40.0, SEARCH_SPAN_NM - half - 40.0
    x = float(rng.uniform(lo, hi))

    if not hard_case:
        # Most normal samples deliberately contain a gate crossing so the
        # reference has the distinctive FinFET landmark described by the task.
        # A minority remain gate-free to prevent the model from using gates as
        # the only correspondence cue.
        if layout.gates and rng.random() < 0.75:
            g = layout.gates[int(rng.integers(0, len(layout.gates)))]
            y = float(np.clip(g.y_nm + rng.uniform(-280.0, 280.0), lo, hi))
        else:
            for _ in range(1000):
                y = float(rng.uniform(lo, hi))
                if not gates_intersect_window(layout, y, half, 20.0):
                    return x, y
        return x, y

    # Hard cases are genuinely landmark-free: the search still contains gates,
    # but the 1000 nm reference region does not.
    for _ in range(1000):
        y = float(rng.uniform(lo, hi))
        if not gates_intersect_window(layout, y, half, 20.0):
            return x, y
    return x, float(rng.uniform(lo, hi))


def solve_observed_target(target_x, target_y, dx_field, dy_field):
    h, w = dx_field.shape
    x = target_x / SEARCH_NM_PER_PX
    y = target_y / SEARCH_NM_PER_PX
    for _ in range(5):
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        x = (target_x - float(dx_field[yi, xi])) / SEARCH_NM_PER_PX
        y = (target_y - float(dy_field[yi, xi])) / SEARCH_NM_PER_PX
    return x, y


def generate_pair(pair_id, seed, hard_case=False):
    root = np.random.SeedSequence([seed, pair_id, SEED_OFFSET])
    layout_ss, ref_ss, search_ss = root.spawn(3)
    layout_rng = np.random.default_rng(layout_ss)
    ref_rng = np.random.default_rng(ref_ss)
    search_rng = np.random.default_rng(search_ss)

    layout = make_layout(layout_rng, hard_case)
    roughness = build_roughness(layout)
    target_x, target_y = choose_target(layout, layout_rng, hard_case)

    # Reference: exact physical target region, with only acquisition-level
    # rotation/scale applied to the high-mag view.
    yy, xx = np.mgrid[0:REF_PX, 0:REF_PX].astype(np.float32)
    cx = cy = (REF_PX - 1) * 0.5
    theta = np.deg2rad(float(layout_rng.uniform(-2.0, 2.0)))
    scale = float(layout_rng.uniform(0.98, 1.02))
    ct, st = np.cos(theta), np.sin(theta)
    dx = xx - cx
    dy = yy - cy
    xr = (dx * ct - dy * st) / scale
    yr = (dx * st + dy * ct) / scale
    ref_x = target_x + xr * REF_NM_PER_PX
    ref_y = target_y + yr * REF_NM_PER_PX
    ref_geom = render_physical(ref_x, ref_y, layout, roughness, REF_NM_PER_PX)
    ref_img, ref_noise = apply_acquisition(ref_geom, ref_rng, True)

    # Search: render a larger tiled physical field. Drift changes the observed
    # sampling coordinates, but does not change the underlying canonical layout.
    ss = SEARCH_SS
    dim = SEARCH_PX * ss
    px_nm = SEARCH_NM_PER_PX / ss
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    drift_amp = float(search_rng.uniform(1.0, 3.5))
    drift_x, drift_y = drift_field((dim, dim), search_rng, drift_amp)
    obs_x = xx * px_nm
    obs_y = yy * px_nm
    phys_x = obs_x + drift_x
    phys_y = obs_y + drift_y
    search_geom_ss = render_physical(phys_x, phys_y, layout, roughness, px_nm)
    search_geom = cv2.resize(search_geom_ss, (SEARCH_PX, SEARCH_PX), interpolation=cv2.INTER_AREA)
    search_img, search_noise = apply_acquisition(search_geom, search_rng, False)

    # Ground truth is the observed location of the canonical physical target,
    # after the same drift field used to render the search image.
    gt_x, gt_y = solve_observed_target(target_x, target_y, drift_x, drift_y)
    half_px = REF_SPAN_NM / SEARCH_NM_PER_PX * 0.5
    bbox = [gt_x - half_px, gt_y - half_px, gt_x + half_px, gt_y + half_px]

    gt = {
        "pair_id": int(pair_id),
        "architecture": "finfet",
        "target_x_nm": float(target_x),
        "target_y_nm": float(target_y),
        "center_x_wide_px": float(gt_x),
        "center_y_wide_px": float(gt_y),
        "bbox_wide_px": [float(v) for v in bbox],
        "bbox_size_px": 100.0,
        "rotation_deg": float(np.rad2deg(theta)),
        "reference_scale": float(scale),
        "drift_amp_nm": drift_amp,
        "hard_case": bool(hard_case),
        "contains_gate_in_ref": bool(gates_intersect_window(layout, target_y, REF_SPAN_NM * 0.5)),
        "reference_noise_parameters": ref_noise,
        "search_noise_parameters": search_noise,
        "layout_parameters": asdict(layout),
        "random_seed": int(seed),
    }
    return ref_img, search_img, gt


def structure_score(img):
    # Vertical fins should dominate the x-gradient energy. Horizontal gates
    # contribute y-gradient energy, so this is a useful architecture sanity check.
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    x_energy = float(np.mean(np.abs(gx)))
    y_energy = float(np.mean(np.abs(gy)))
    return x_energy / (y_energy + 1e-6), float(f.std())


def scanline_stats(img):
    row_mean = img.astype(np.float32).mean(axis=1)
    diff = np.abs(row_mean - 0.5 * (np.roll(row_mean, 1) + np.roll(row_mean, -1)))
    diff[[0, -1]] = 0
    return float(diff.max()), float(np.percentile(diff, 99.5))


def validate(records, out_dir):
    report = {"n_pairs": len(records), "all_passed": True, "failures": [], "checks": {}}
    structure_scores = []
    stds = []
    for r in records:
        pid = r["pair_id"]
        rp = os.path.join(out_dir, "reference", f"pair_{pid:04d}_ref.png")
        sp = os.path.join(out_dir, "search", f"pair_{pid:04d}_search.png")
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if ref is None or search is None:
            report["failures"].append(f"pair {pid}: unreadable")
            continue
        if ref.shape != (1000, 1000) or search.shape != (1000, 1000):
            report["failures"].append(f"pair {pid}: bad dimensions")
        if not np.isfinite(ref).all() or not np.isfinite(search).all():
            report["failures"].append(f"pair {pid}: non-finite pixels")

        score, sd = structure_score(search)
        structure_scores.append(score)
        stds.append(sd)
        if score < 1.15 or sd < 7.0:
            report["failures"].append(f"pair {pid}: weak search structure score={score:.2f}, std={sd:.2f}")

        x, y = r["center_x_wide_px"], r["center_y_wide_px"]
        x0, y0, x1, y1 = r["bbox_wide_px"]
        if not (0 <= x0 < x1 <= SEARCH_PX and 0 <= y0 < y1 <= SEARCH_PX):
            report["failures"].append(f"pair {pid}: GT bbox out of bounds")
        if abs((x1 - x0) - 100.0) > 1e-5 or abs((y1 - y0) - 100.0) > 1e-5:
            report["failures"].append(f"pair {pid}: bbox not 100x100")
        if not (0 <= x <= SEARCH_PX and 0 <= y <= SEARCH_PX):
            report["failures"].append(f"pair {pid}: GT center out of bounds")

        for name, image in (("ref", ref), ("search", search)):
            mx, p995 = scanline_stats(image)
            if mx > 18.0 and p995 > 8.0:
                report["failures"].append(f"pair {pid} {name}: possible scanline artifact")

    report["checks"] = {
        "search_structure_score_mean": float(np.mean(structure_scores)) if structure_scores else None,
        "search_structure_score_min": float(np.min(structure_scores)) if structure_scores else None,
        "search_pixel_std_mean": float(np.mean(stds)) if stds else None,
        "all_files_valid": not any("unreadable" in x for x in report["failures"]),
        "gt_bbox_nominal_size_px": 100.0,
        "architecture": "finfet",
    }
    report["all_passed"] = len(report["failures"]) == 0
    return report


def save_preview(records, out_dir, n=6):
    pdir = os.path.join(out_dir, "previews")
    os.makedirs(pdir, exist_ok=True)
    ids = np.linspace(0, len(records) - 1, min(n, len(records)), dtype=int) if records else []
    for idx in ids:
        r = records[int(idx)]
        pid = r["pair_id"]
        ref = cv2.imread(os.path.join(out_dir, "reference", f"pair_{pid:04d}_ref.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(out_dir, "search", f"pair_{pid:04d}_search.png"), cv2.IMREAD_GRAYSCALE)
        sb = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        x0, y0, x1, y1 = [int(round(v)) for v in r["bbox_wide_px"]]
        cv2.rectangle(sb, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(sb, "GT", (x0 + 3, max(18, y0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        combo = np.full((1000, 2020, 3), 32, np.uint8)
        combo[:, :1000] = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        combo[:, 1020:] = sb
        cv2.imwrite(os.path.join(pdir, f"preview_{pid:04d}.png"), combo)


def write_citations(out_dir):
    dst = os.path.join(out_dir, "CITATIONS.md")
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citations.md")
    if os.path.exists(src):
        with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())
    else:
        with open(dst, "w", encoding="utf-8") as f:
            f.write("Add the team's literature-backed modeling references here.\n")


def main():
    ap = argparse.ArgumentParser(description="Generate paired synthetic FinFET SEM reference/search data")
    ap.add_argument("--architecture", choices=["finfet"], default="finfet")
    ap.add_argument("--num-pairs", "--num-samples", dest="num_pairs", type=int, default=1000)
    ap.add_argument("--num-hard-cases", type=int, default=100)
    ap.add_argument("--output-dir", type=str, default="./output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.num_pairs < 0 or args.num_hard_cases < 0:
        raise SystemExit("pair counts must be non-negative")

    out = args.output_dir
    os.makedirs(os.path.join(out, "reference"), exist_ok=True)
    os.makedirs(os.path.join(out, "search"), exist_ok=True)

    records = []
    t0 = time.perf_counter()
    total = args.num_pairs + args.num_hard_cases
    for pid in range(total):
        hard = pid >= args.num_pairs
        ref, search, gt = generate_pair(pid, args.seed, hard)
        cv2.imwrite(os.path.join(out, "reference", f"pair_{pid:04d}_ref.png"), ref, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(out, "search", f"pair_{pid:04d}_search.png"), search, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        records.append(gt)

    elapsed = time.perf_counter() - t0
    with open(os.path.join(out, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    stats = {
        "architecture": args.architecture,
        "n_pairs": total,
        "n_standard_pairs": args.num_pairs,
        "n_hard_pairs": args.num_hard_cases,
        "generation_time_sec": elapsed,
        "avg_time_per_pair_sec": elapsed / max(total, 1),
        "estimated_1000_pairs_sec": elapsed / max(total, 1) * 1000.0,
        "reference_size_px": [1000, 1000],
        "search_size_px": [1000, 1000],
        "reference_nm_per_px": REF_NM_PER_PX,
        "search_nm_per_px": SEARCH_NM_PER_PX,
    }
    with open(os.path.join(out, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    report = validate(records, out)
    with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    save_preview(records, out)
    write_citations(out)

    print(f"Generated {total} pairs in {elapsed:.2f}s ({elapsed / max(total, 1):.3f}s/pair)")
    print(f"Validation: {'PASS' if report['all_passed'] else 'FAIL'}")
    if report["failures"]:
        print("First failures:")
        for msg in report["failures"][:10]:
            print(" -", msg)


if __name__ == "__main__":
    main()