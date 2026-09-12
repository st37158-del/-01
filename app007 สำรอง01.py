"""
================================================================================
 Coastal Erosion / Accretion Analyzer v5.0 (Pro Research Edition)
 Virtual Transect Shoreline Dynamics & Robust Box-Counting Fractal Engine
================================================================================

An advanced Streamlit application for analyzing, measuring, and forecasting
coastal erosion and accretion trends from multi-temporal coastal imagery.

Key Upgrades in v5.0:
1. Automated Image Registration: ORB feature-based Homography Co-Registration
   anchored to the baseline image to eliminate false spatial drift.
2. Zero Border-Artifact Shoreline Extraction: Solid external morphology that
   prevents image frame boundaries and inland water bodies from distorting coastlines.
3. Virtual Transect Engine (DSAS Principle): Replaces flawed Area/Length metric
   with cross-shore transects, computing EPR (End-Point Rate), Mean Shift,
   Uncertainty (Std Dev), and spatial erosion/accretion distributions.
4. Robust Box-Counting Fractal Engine: Native scaling window filtering (removing
   discretization noise and sparse saturation) with R² validation.
5. Memory & Performance Optimized: Modular architecture with cached pipelines.

HOW TO RUN:
1. Install dependencies:
   pip install streamlit opencv-python numpy pandas scikit-learn plotly pillow

2. Run the app:
   streamlit run app.py
================================================================================
"""

import re
import warnings
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
APP_VERSION = "5.0 (Pro Research)"
MAX_DIM_DEFAULT = 1024          # Default maximum processing dimension (pixels)
DEFAULT_TRANSECTS = 30          # Number of cross-shore measurement transects
STABILITY_THRESHOLD_M = 0.05    # Below this absolute change (m), classify as Stable
MIN_IMAGES_FOR_TRENDS = 2       # Minimum images required for trend & forecast analysis
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]


# ==============================================================================
# HELPER & PREPROCESSING FUNCTIONS
# ==============================================================================

def parse_date_input(text: str) -> Optional[date]:
    """Parse YYYY-MM-DD or bare YYYY strings into a datetime.date object."""
    text = (text or "").strip()
    if not text:
        return None

    if text.isdigit() and 3 <= len(text) <= 4:
        try:
            year = int(text)
            if 1000 <= year <= 9999:
                return date(year, 1, 1)
        except ValueError:
            pass

    try:
        parsed = pd.to_datetime(text)
        return parsed.date()
    except Exception:
        return None


def extract_date_from_filename(filename: str) -> Optional[date]:
    """
    Best-effort parse of a leading YYYY-MM-DD date stamp from a filename, matching the
    Copernicus Browser export naming convention, e.g.
    '2025-12-21-00_00_2025-12-21-23_59_Sentinel-2_L2A_NDWI.jpg' -> 2025-12-21.
    """
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", filename)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def to_decimal_year(d: date) -> float:
    """Convert datetime.date to decimal year float (e.g. 2020-07-02 -> 2020.501)."""
    year_start = date(d.year, 1, 1)
    year_end = date(d.year + 1, 1, 1)
    year_length_days = (year_end - year_start).days
    days_into_year = (d - year_start).days
    return d.year + (days_into_year / year_length_days)


def load_image_bgr(uploaded_file) -> np.ndarray:
    """Robustly load an uploaded image into a BGR uint8 numpy array."""
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    file_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        uploaded_file.seek(0)
        pil_img = Image.open(uploaded_file).convert("RGB")
        rgb_array = np.array(pil_img)
        img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError("ไม่สามารถแปลงไฟล์ภาพได้")

    return img


def resize_for_processing(img_bgr: np.ndarray, original_scale: float, max_dim: int) -> Tuple[np.ndarray, float, bool, Tuple[int, int], Tuple[int, int]]:
    """Downscale image if max dimension is exceeded while keeping physical scale adjusted."""
    h, w = img_bgr.shape[:2]
    longest_side = max(h, w)
    original_shape = (h, w)

    if longest_side <= max_dim:
        return img_bgr, original_scale, False, original_shape, (h, w)

    resize_factor = max_dim / float(longest_side)
    new_w = max(1, int(round(w * resize_factor)))
    new_h = max(1, int(round(h * resize_factor)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    effective_scale = original_scale / resize_factor
    return resized, effective_scale, True, original_shape, (new_h, new_w)


def auto_crop_ui_chrome(img_bgr: np.ndarray, dominant_frac_thresh: float = 0.4,
                        banner_color_tol: float = 18.0, body_diff_thresh: float = 35.0,
                        max_crop_frac: float = 0.12) -> Tuple[np.ndarray, int, int]:
    """
    Detects and strips solid-color UI banner strips (e.g. Copernicus Browser's dark
    timestamp header / white credit-scale footer) from the top and bottom of a screenshot.

    A row only counts as a banner row if it is BOTH near-uniform in color AND clearly
    different from the image's own body color (sampled from the middle 30-70% band).
    The body-difference check is what stops legitimate uniform imagery (e.g. a large flat
    open-water region) from being mistaken for a banner and cropped away. Scanning inward
    from each edge stops once 3 consecutive rows break from the banner's reference color,
    capped at max_crop_frac of the image height per side.
    """
    h, w = img_bgr.shape[:2]
    max_crop = max(1, int(h * max_crop_frac))

    body_top = int(h * 0.3)
    body_bottom = max(body_top + 1, int(h * 0.7))
    body_region = img_bgr[body_top:body_bottom].reshape(-1, 3).astype(np.float64)
    body_ref = np.median(body_region, axis=0)

    def row_dominant(y: int):
        row = img_bgr[y].reshape(-1, 3).astype(np.int16)
        colors, counts = np.unique(row, axis=0, return_counts=True)
        dom_color = colors[np.argmax(counts)].astype(np.float64)
        dom_frac = counts.max() / row.shape[0]
        return dom_color, dom_frac

    def scan(row_indices) -> int:
        cut = 0
        ref_color = None
        consec_break = 0
        for i, y in enumerate(row_indices):
            dom_color, dom_frac = row_dominant(y)
            is_banner_like = (
                dom_frac >= dominant_frac_thresh
                and np.linalg.norm(dom_color - body_ref) >= body_diff_thresh
            )

            if ref_color is None:
                if not is_banner_like:
                    break
                ref_color = dom_color

            same_banner = is_banner_like and np.linalg.norm(dom_color - ref_color) <= banner_color_tol
            if same_banner:
                cut = i + 1
                consec_break = 0
            else:
                consec_break += 1
                if consec_break >= 6:
                    break
        return cut

    top_cut = scan(range(0, max_crop))
    bottom_cut = scan(range(h - 1, h - 1 - max_crop, -1))

    if top_cut == 0 and bottom_cut == 0:
        return img_bgr, 0, 0
    if top_cut + bottom_cut >= h:
        return img_bgr, 0, 0

    cropped = img_bgr[top_cut:h - bottom_cut, :]
    return cropped, top_cut, bottom_cut


# ==============================================================================
# IMAGE REGISTRATION & CO-REGISTRATION (ORB HOMOGRAPHY)
# ==============================================================================

def align_image_orb(target_bgr: np.ndarray, ref_bgr: np.ndarray, max_features: int = 1500,
                    max_warp_frac: float = 0.05) -> Tuple[np.ndarray, bool, str]:
    """
    Co-registers target_bgr to ref_bgr coordinate space using ORB feature matching and Homography.
    Prevents false shoreline shifts caused by slight camera offsets or varying crops.

    Two safeguards protect against repetitive/periodic scenes (e.g. grid-like aquaculture
    ponds or farmland) where ORB+RANSAC can lock onto a self-consistent but spatially WRONG
    correspondence -- matching one grid cell to a visually-identical neighboring one:
    1. A fixed RNG seed makes RANSAC's homography estimate reproducible across runs on the
       same inputs (OpenCV's RANSAC is otherwise randomized, so re-running the exact same
       alignment could silently produce a different -- possibly bad -- result each time).
    2. The resulting homography is sanity-checked by warping a dense grid of sample points
       and rejecting the fit if any point would be displaced by more than max_warp_frac of
       the image diagonal -- a plausible registration correction should stay small, not
       warp the image by a large fraction of its own size.
    """
    if target_bgr.shape[:2] == ref_bgr.shape[:2] and np.array_equal(target_bgr, ref_bgr):
        return target_bgr, True, "ภาพอ้างอิงฐาน (Reference Baseline)"

    cv2.setRNGSeed(42)

    orb = cv2.ORB_create(max_features)
    gray_tgt = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    kp_tgt, des_tgt = orb.detectAndCompute(gray_tgt, None)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

    if des_tgt is None or des_ref is None or len(kp_tgt) < 8 or len(kp_ref) < 8:
        return target_bgr, False, "ไม่พบจุด Feature เพียงพอสำหรับการ Alignment (ใช้ภาพเดิม)"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_tgt, des_ref)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:max(10, int(len(matches) * 0.4))]
    if len(good_matches) < 6:
        return target_bgr, False, "จุดจับคู่ (Matches) น้อยเกินไป (ใช้ภาพเดิม)"

    src_pts = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return target_bgr, False, "คำนวณ Homography Matrix ไม่สำเร็จ (ใช้ภาพเดิม)"

    h, w = target_bgr.shape[:2]
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 12), np.linspace(0, h - 1, 12))
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(grid_pts, H).reshape(-1, 2)
    displacement = np.linalg.norm(warped_pts - grid_pts.reshape(-1, 2), axis=1)
    diagonal = float(np.hypot(w, h))
    max_shift_frac = float(displacement.max() / diagonal)

    if max_shift_frac > max_warp_frac:
        return target_bgr, False, (
            f"Homography บิดภาพเกินสมเหตุสมผล ({max_shift_frac * 100:.0f}% ของภาพ) "
            f"อาจเกิดจากลวดลายซ้ำในภาพ (เช่น บ่อกุ้ง/นาเกลือ) หลอก Feature Matching — ใช้ภาพเดิม (ไม่ Alignment)"
        )

    aligned = cv2.warpPerspective(target_bgr, H, (ref_bgr.shape[1], ref_bgr.shape[0]))
    return aligned, True, f"จัดตำแหน่งภาพตรงกันเรียบร้อย ({len(good_matches)} จุดคู่สมนัย)"


# ==============================================================================
# WATER-LAND SEGMENTATION & CLEAN SHORELINE EXTRACTION
# ==============================================================================

def extract_binary_mask(img_bgr: np.ndarray, method: str, blur_kernel: int, land_is_bright: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    """Segment land from water using Otsu thresholding or HSV water color detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if method == "HSV Color Segmentation (แยกสีน้ำทะเลและแผ่นดิน)":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Water typically has hue in blue-cyan range (75-135 in OpenCV 0-180 scale)
        lower_water = np.array([75, 20, 20])
        upper_water = np.array([140, 255, 255])
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        binary = cv2.bitwise_not(water_mask)  # Land = 255, Water = 0
        thresh_val = 0.0
    else:  # Otsu Thresholding (Grayscale / NDWI)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        thresh_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if not land_is_bright:
            binary = cv2.bitwise_not(binary)

    return gray, binary, float(thresh_val)


def extract_clean_coastline(binary_mask: np.ndarray, denoise_kernel: int, keep_largest_only: bool,
                            channel_sever_kernel: int = 0) -> Optional[Dict[str, Any]]:
    """
    Clean binary mask using morphological filtering, sever inland water bodies (aquaculture
    ponds, irrigation canals, tidal creeks) that are hydrologically connected to the open sea
    through narrow channels, and extract the pure land-sea interface without outer image
    frame border artifacts.

    A pond linked to the sea by a thin canal is NOT an enclosed hole in the land mask — it is
    part of the same connected water region as the sea, so cv2.findContours(RETR_EXTERNAL)
    naturally traces the contour in and out around it, producing a false "shoreline" around
    every such pond. Fixing this requires severing those channels on the WATER side (an
    opening wide enough to break canals but narrower than the true open sea) before picking
    the largest remaining water component as the real sea; everything else (real land AND any
    inland ponds/canals) is then treated as land, matching the intended coastal-erosion use case.
    """
    h, w = binary_mask.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (denoise_kernel, denoise_kernel))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    land_source = cleaned
    if channel_sever_kernel and channel_sever_kernel >= 3:
        water_raw = cv2.bitwise_not(cleaned)
        sever_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (channel_sever_kernel, channel_sever_kernel))
        water_severed = cv2.morphologyEx(water_raw, cv2.MORPH_OPEN, sever_kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water_severed, connectivity=8)
        if n_labels > 1:
            sea_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            sea_mask = np.where(labels == sea_label, np.uint8(255), np.uint8(0))
            # Restore the true sea boundary eroded away by the opening, without
            # reconnecting the severed channels (clip back to the original water extent).
            sea_mask = cv2.dilate(sea_mask, sever_kernel)
            sea_mask = cv2.bitwise_and(sea_mask, water_raw)
            land_source = cv2.bitwise_not(sea_mask)

    contours, _ = cv2.findContours(land_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    if keep_largest_only:
        contours_used = [max(contours, key=cv2.contourArea)]
    else:
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas) if areas else 0.0
        if max_area <= 0:
            return None
        contours_used = [c for c, a in zip(contours, areas) if a >= 0.02 * max_area]

    # Create solid filled landmass (eliminates internal lakes, building edges, and noise)
    filled_land = np.zeros_like(cleaned)
    cv2.drawContours(filled_land, contours_used, -1, 255, thickness=-1)

    # Shoreline is where solid land touches water inside the image domain
    water_mask = (filled_land == 0).astype(np.uint8) * 255
    water_dilated = cv2.dilate(water_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    raw_shoreline = cv2.bitwise_and(filled_land, water_dilated)

    # Zero out outer 1-pixel frame border to eliminate rectangular frame edges completely
    border_mask = np.zeros_like(raw_shoreline)
    border_mask[1:-1, 1:-1] = 255
    true_shoreline = cv2.bitwise_and(raw_shoreline, border_mask)

    shore_y, shore_x = np.where(true_shoreline > 0)
    if len(shore_x) == 0:
        return None

    shore_pts = np.column_stack([shore_x, shore_y])
    land_area_px = float(np.count_nonzero(filled_land))
    coastline_length_px = float(len(shore_pts))

    # Split shoreline points into continuous polylines for smooth plotting
    shore_lines = split_into_polylines(shore_pts)

    return {
        "filled_land": filled_land,
        "shoreline_img": true_shoreline,
        "shore_pts": shore_pts,
        "shore_lines": shore_lines,
        "land_area_px": land_area_px,
        "coastline_length_px": coastline_length_px,
        "contours_used": contours_used,
    }


def split_into_polylines(pts: np.ndarray, max_jump: float = 3.5) -> List[np.ndarray]:
    """Split sorted or clustered point cloud into continuous polylines for clean visualization."""
    if len(pts) == 0:
        return []
    # Sort primarily along dominant variance axis
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eig(cov)
    proj = pts @ evecs[:, np.argmax(evals)]
    order = np.argsort(proj)
    ordered_pts = pts[order]

    lines = []
    current_line = [ordered_pts[0]]
    for i in range(1, len(ordered_pts)):
        p_prev = ordered_pts[i - 1]
        p_curr = ordered_pts[i]
        dist = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        if dist > max_jump:
            if len(current_line) >= 3:
                lines.append(np.array(current_line))
            current_line = [p_curr]
        else:
            current_line.append(p_curr)
    if len(current_line) >= 3:
        lines.append(np.array(current_line))
    return lines


def create_coastline_overlay(img_bgr: np.ndarray, shoreline_img: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Draw high-visibility green shoreline contour overlay onto original image."""
    overlay = img_bgr.copy()
    dilated_edge = cv2.dilate(shoreline_img, cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness)))
    overlay[dilated_edge > 0] = color
    return overlay


# ==============================================================================
# ROBUST BOX-COUNTING FRACTAL DIMENSION ENGINE
# ==============================================================================

def robust_box_counting(binary_edge_img: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Computes Box-Counting Fractal Dimension with rigorous scaling window filtering.
    Discards discretization noise (small r < 4) and sparse saturation (large r or box
    counts too small to regress on reliably).

    The scaling window is derived from the shoreline's own bounding-box extent, not the
    full image frame. A coastline occupying only a corner of a mostly-empty frame would
    otherwise let box sizes grow far past what the shoreline itself needs -- at that scale
    a handful of giant boxes already cover the whole curve, so N(r) barely shrinks as r
    grows and the regression slope (the FD estimate) gets biased low, sometimes below the
    topological minimum of 1.0 for a connected curve.
    """
    bool_img = binary_edge_img > 0
    if not np.any(bool_img):
        return None

    h, w = bool_img.shape
    ys, xs = np.where(bool_img)
    extent_h = int(ys.max() - ys.min()) + 1
    extent_w = int(xs.max() - xs.min()) + 1
    min_dim = min(extent_h, extent_w)

    # Scaling window: from r=4 to r=min_dim//4 (powers of 2)
    box_sizes = []
    size = 4
    max_size = max(8, min_dim // 4)
    while size <= max_size:
        box_sizes.append(size)
        size *= 2

    if len(box_sizes) < 3:
        # Fallback to geometric spacing if the shoreline's extent is small
        box_sizes = [int(s) for s in np.geomspace(4, max(8, min_dim // 3), num=5)]
        box_sizes = sorted(list(set(box_sizes)))

    counts = []
    for r in box_sizes:
        n_y = int(np.ceil(h / r))
        n_x = int(np.ceil(w / r))
        grid = np.zeros((n_y * r, n_x * r), dtype=bool)
        grid[:h, :w] = bool_img
        reshaped = grid.reshape(n_y, r, n_x, r)
        has_pixel = reshaped.any(axis=(1, 3))
        counts.append(int(has_pixel.sum()))

    box_sizes_arr = np.array(box_sizes, dtype=np.float64)
    counts_arr = np.array(counts, dtype=np.float64)

    # Drop box sizes whose count is too small to carry statistical weight in the fit --
    # these sit in the saturated tail where a curve gets covered by a handful of giant
    # boxes and would otherwise drag the slope down.
    reliable = counts_arr >= 4
    if reliable.sum() >= 3:
        box_sizes_arr = box_sizes_arr[reliable]
        counts_arr = counts_arr[reliable]

    log_inv_r = np.log(1.0 / box_sizes_arr)
    log_n = np.log(counts_arr)

    slope, intercept = np.polyfit(log_inv_r, log_n, 1)
    y_pred = slope * log_inv_r + intercept
    ss_res = np.sum((log_n - y_pred) ** 2)
    ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "box_sizes": box_sizes_arr.astype(int).tolist(),
        "counts": counts_arr.astype(int).tolist(),
        "log_inv_r": log_inv_r.tolist(),
        "log_n": log_n.tolist(),
        "fd": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


# ==============================================================================
# VIRTUAL TRANSECT SHORELINE CHANGE ENGINE (DSAS PRINCIPLE)
# ==============================================================================

def generate_virtual_transects(ref_shore_pts: np.ndarray, ref_filled_land: np.ndarray,
                               n_transects: int = 30) -> Optional[Dict[str, Any]]:
    """
    Generates cross-shore virtual transects automatically detecting shoreline orientation
    (North-South vs East-West) and the inland direction.
    """
    if len(ref_shore_pts) < 10:
        return None

    xs = ref_shore_pts[:, 0]
    ys = ref_shore_pts[:, 1]
    range_x = float(xs.max() - xs.min())
    range_y = float(ys.max() - ys.min())

    axis = 'y' if range_y >= range_x else 'x'

    # Determine inland side (centroid of land vs centroid of shoreline)
    land_y, land_x = np.where(ref_filled_land > 0)
    if len(land_x) == 0:
        return None

    if axis == 'y':
        land_on_min = float(np.mean(land_x)) < float(np.mean(xs))
        sample_min = float(ys.min() + 0.05 * range_y)
        sample_max = float(ys.max() - 0.05 * range_y)
    else:
        land_on_min = float(np.mean(land_y)) < float(np.mean(ys))
        sample_min = float(xs.min() + 0.05 * range_x)
        sample_max = float(xs.max() - 0.05 * range_x)

    samples = np.linspace(sample_min, sample_max, n_transects)

    return {
        "axis": axis,
        "land_on_min_side": land_on_min,
        "samples": samples.tolist(),
    }


def measure_transect_shifts(ref_pts: np.ndarray, tgt_pts: np.ndarray, transects: Dict[str, Any],
                            scale_m: float) -> Tuple[float, float, float, float, float, List[Dict[str, Any]]]:
    """
    Measures cross-shore displacement along each transect between reference (t0) and target (t).
    Returns (Mean Shift, Median Shift, Std Dev, Max Erosion, Max Accretion, Transect Details).
    """
    axis = transects["axis"]
    samples = transects["samples"]
    land_on_min = transects["land_on_min_side"]

    def build_lookup(points):
        lookup = {}
        for p in points:
            coord = int(round(p[1] if axis == 'y' else p[0]))
            val = float(p[0] if axis == 'y' else p[1])
            if coord not in lookup:
                lookup[coord] = []
            lookup[coord].append(val)
        return {k: float(np.median(v)) for k, v in lookup.items()}

    ref_map = build_lookup(ref_pts)
    tgt_map = build_lookup(tgt_pts)

    transect_details = []
    shifts = []

    for idx, s in enumerate(samples):
        s_int = int(round(s))
        # Search radius of +-4 pixels to ensure robust intersection
        r_c = [ref_map[k] for k in range(s_int - 4, s_int + 5) if k in ref_map]
        t_c = [tgt_map[k] for k in range(s_int - 4, s_int + 5) if k in tgt_map]

        if r_c and t_c:
            ref_val = float(np.mean(r_c))
            tgt_val = float(np.mean(t_c))
            raw_diff = tgt_val - ref_val
            # Positive shift = seaward movement (Accretion); Negative shift = landward movement (Erosion)
            shift_m = (raw_diff if land_on_min else -raw_diff) * scale_m

            shifts.append(shift_m)
            transect_details.append({
                "transect_id": idx + 1,
                "coord": s_int,
                "ref_val": ref_val,
                "tgt_val": tgt_val,
                "shift_m": shift_m,
                "status": classify_change(shift_m),
            })

    if not shifts:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    mean_shift = float(np.mean(shifts))
    median_shift = float(np.median(shifts))
    std_shift = float(np.std(shifts))
    max_erosion = float(min(shifts))
    max_accretion = float(max(shifts))

    return mean_shift, median_shift, std_shift, max_erosion, max_accretion, transect_details


def classify_change(distance_m: float) -> str:
    """Classify shoreline distance change into Erosion / Accretion / Stable."""
    if abs(distance_m) < STABILITY_THRESHOLD_M:
        return "คงที่ (Stable)"
    return "งอก (Accretion)" if distance_m > 0 else "กัดเซาะ (Erosion)"


# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================
st.set_page_config(
    page_title=f"เครื่องมือวิเคราะห์การกัดเซาะชายฝั่ง — v{APP_VERSION}",
    layout="wide",
)

st.title("เครื่องมือวิเคราะห์การกัดเซาะ/การงอกของชายฝั่ง")
st.caption("ระบบวัดแนวชายฝั่งด้วย Virtual Transects (DSAS Standard) ร่วมกับ Box-Counting Fractal Complexity Engine")


# ==============================================================================
# SIDEBAR — PROCESSING PARAMETERS
# ==============================================================================
st.sidebar.header("พารามิเตอร์การประมวลผล")

enable_alignment = st.sidebar.checkbox(
    "เปิดใช้ Auto-Alignment (ORB Feature Matching)",
    value=True,
    help="แนะนำให้เปิดไว้: จัดตำแหน่งและปรับมุมมองภาพทุกช่วงเวลาให้ตรงกับภาพฐานปีแรก ป้องกันความคลาดเคลื่อนเชิงตำแหน่ง "
         "(ปรับให้เข้มงวดขึ้นสำหรับภาพที่ถ่ายจากมุมมองเดิมทุกครั้ง เช่น ดาวน์โหลดจากเว็บวิวเดิมโดยเปลี่ยนแค่ปี — "
         "ถ้าพบว่าต้องบิดภาพมากเกินคาด ระบบจะปฏิเสธการจัดตำแหน่งนั้นและใช้ภาพเดิมแทน เพื่อกันไม่ให้เส้นแนวชายฝั่งเบี้ยวจากการจับคู่ผิด)"
)

auto_crop_banner = st.sidebar.checkbox(
    "ตัดแถบ UI (วันที่/เครดิต/มาตราส่วน) ออกอัตโนมัติ",
    value=True,
    help="สำหรับภาพที่แคปจาก Copernicus Browser ซึ่งมักติดแถบข้อความสีพื้นล้วนด้านบน-ล่างมาด้วย ระบบจะตรวจจับแถบสีทึบเหล่านี้และตัดออกก่อนประมวลผล เพื่อไม่ให้ขอบแถบกลายเป็นเส้นชายฝั่งปลอมที่บิดเบือนค่า Fractal Dimension"
)

segmentation_mode = st.sidebar.radio(
    "โหมดการจำแนกแผ่นดินและน้ำ (Segmentation)",
    ["Otsu Thresholding (Grayscale / NDWI)", "HSV Color Segmentation (แยกสีน้ำทะเลและแผ่นดิน)"],
    index=1,
    help="หากเป็นภาพดัชนี NDWI หรือภาพขาวดำ ให้เลือก Otsu หากเป็นภาพถ่ายสีจากดาวเทียมทั่วไปให้ลองใช้ HSV"
)

if segmentation_mode == "Otsu Thresholding (Grayscale / NDWI)":
    blur_kernel = st.sidebar.slider(
        "ขนาด Kernel สำหรับ Gaussian Blur (เลขคี่)",
        min_value=3, max_value=15, value=5, step=2,
        help="ลดจุดรบกวนก่อนทำ Segmentation"
    )
    land_choice = st.sidebar.radio(
        "พิกเซลสีไหนแทน 'แผ่นดิน' หลังทำ Threshold?",
        ["สว่าง (255) = แผ่นดิน", "มืด (0) = แผ่นดิน"],
        index=0,
    )
    land_is_bright = (land_choice == "สว่าง (255) = แผ่นดิน")
else:
    # HSV mode classifies land/water by hue directly -- it never touches the blurred
    # grayscale image or the brightness polarity, so neither control applies.
    blur_kernel = 5
    land_is_bright = True

denoise_kernel = st.sidebar.slider(
    "ระดับลด Noise และเติมเต็มแผ่นดิน (Denoise Kernel Size)",
    min_value=3, max_value=25, value=9, step=2,
)

channel_sever_kernel = st.sidebar.slider(
    "ระดับตัดขาดแหล่งน้ำภายใน (บ่อ/คลอง/นากุ้งที่เชื่อมกับทะเล)",
    min_value=0, max_value=101, value=21, step=2,
    help="บ่อเลี้ยงกุ้ง/นาเกลือ/คลองที่มีช่องทางเชื่อมกับทะเล (แม้เพียงไม่กี่พิกเซล) จะถูกนับเป็นส่วนเดียวกับทะเลและถูกลากเป็นแนวชายฝั่งปลอม ค่านี้ควรตั้งให้กว้างกว่าความกว้างของคลอง/ทางน้ำเชื่อมต่อในภาพ (พิกเซล) เพื่อตัดขาดออกจากทะเลจริงก่อนคำนวณ ตั้งเป็น 0 เพื่อปิดการทำงานนี้"
)

keep_largest_only = st.sidebar.checkbox(
    "เลือกเฉพาะแผ่นดินผืนหลัก (ตัดเกาะเล็ก/จุดกวนทิ้ง)",
    value=True,
    help="สกัดเฉพาะแนวชายฝั่งผืนแผ่นดินใหญ่ ป้องกันการคำนวณระยะหลุดกรอบ"
)

n_transects = st.sidebar.slider(
    "จำนวนเส้นตัดขวางสำหรับวัดระยะ (Virtual Transects)",
    min_value=10, max_value=60, value=DEFAULT_TRANSECTS, step=5,
    help="จำนวนเส้นสำรวจแนวตั้งฉากตามแนวชายฝั่งเพื่อเฉลี่ยการเคลื่อนที่และหาจุดวิกฤต"
)

max_dim = st.sidebar.slider(
    "ขนาดภาพสูงสุดสำหรับประมวลผล (พิกเซล)",
    min_value=256, max_value=4096, value=MAX_DIM_DEFAULT, step=128,
)


# ==============================================================================
# FILE UPLOAD & PER-IMAGE METADATA INPUT
# ==============================================================================
st.subheader("1. อัปโหลดภาพชายฝั่งหลายช่วงเวลา")

st.caption(
    "ยังไม่มีภาพ? ดาวน์โหลดภาพดาวเทียม Sentinel-2 (NDWI) ได้ฟรีจาก "
    "[Copernicus Browser](https://browser.dataspace.copernicus.eu/) — เลือกพื้นที่ชายฝั่งที่ต้องการ "
    "ตั้งค่า Visualization เป็น NDWI เลือกช่วงวันที่ แล้วกด Export Image เพื่อบันทึกไฟล์"
)

uploaded_files = st.file_uploader(
    "อัปโหลดภาพถ่ายดาวเทียม/ภาพถ่ายทางอากาศตามช่วงเวลา (PNG, JPG, TIFF)",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

image_configs = []

if uploaded_files:
    st.markdown("#### กำหนดวันที่และสเกลความละเอียดของแต่ละภาพ")
    n_files = len(uploaded_files)
    current_year = datetime.now().year

    for i, up_file in enumerate(uploaded_files):
        with st.expander(f"ภาพที่ {i + 1}: {up_file.name}", expanded=(n_files <= 3)):
            col1, col2 = st.columns(2)
            with col1:
                detected_date = extract_date_from_filename(up_file.name)
                if detected_date is not None:
                    default_date_str = detected_date.isoformat()
                    st.caption(f"ตรวจพบวันที่จากชื่อไฟล์: **{default_date_str}**")
                else:
                    default_date_str = str(current_year - (n_files - 1 - i))
                date_str = st.text_input(
                    "วันที่ / ปี (YYYY-MM-DD หรือ YYYY)",
                    value=default_date_str,
                    key=f"date_input_{i}_{up_file.name}",
                    help="ระบบดึงวันที่จากชื่อไฟล์ให้อัตโนมัติถ้าชื่อไฟล์ขึ้นต้นด้วย YYYY-MM-DD (รูปแบบไฟล์จาก Copernicus Browser) แก้ไขเองได้หากไม่ถูกต้อง",
                )
            with col2:
                scale_val = st.number_input(
                    "สเกลความละเอียดเชิงพื้นที่ (เมตร/พิกเซล)",
                    min_value=0.0001, value=0.1, step=0.1, format="%.4f",
                    key=f"scale_input_{i}_{up_file.name}",
                )
            image_configs.append({"file": up_file, "date_str": date_str, "scale": scale_val})
else:
    st.info("กรุณาอัปโหลดภาพชายฝั่งอย่างน้อย 2 ช่วงเวลาเพื่อเริ่มต้นการวิเคราะห์")


# ==============================================================================
# PROCESSING PIPELINE
# ==============================================================================
def run_pipeline(configs, blur_kernel, land_is_bright, segmentation_mode, max_dim,
                 denoise_kernel, keep_largest_only, enable_alignment, n_transects, auto_crop_banner,
                 channel_sever_kernel):
    parsed_configs = []
    errors = []

    for cfg in configs:
        up_file = cfg["file"]
        d = parse_date_input(cfg["date_str"])
        if d is None:
            errors.append(f"ไม่สามารถแปลงวันที่ '{cfg['date_str']}' ของ '{up_file.name}' ได้ — ข้ามภาพนี้")
            continue
        try:
            img_bgr = load_image_bgr(up_file)
        except Exception as exc:
            errors.append(f"อ่านไฟล์ภาพ '{up_file.name}' ไม่สำเร็จ ({exc}) — ข้ามภาพนี้")
            continue

        if auto_crop_banner:
            img_bgr, top_cut, bottom_cut = auto_crop_ui_chrome(img_bgr)
            if top_cut or bottom_cut:
                errors.append(
                    f"ตัดแถบ UI ของ '{up_file.name}' ออกอัตโนมัติ (บน {top_cut}px / ล่าง {bottom_cut}px)"
                )

        resized_img, eff_scale, was_resized, orig_shape, new_shape = resize_for_processing(
            img_bgr, cfg["scale"], max_dim
        )

        parsed_configs.append({
            "file_name": up_file.name,
            "date": d,
            "decimal_year": to_decimal_year(d),
            "scale": eff_scale,
            "original_scale": cfg["scale"],
            "was_resized": was_resized,
            "orig_shape": orig_shape,
            "new_shape": new_shape,
            "img_bgr": resized_img,
        })

    if not parsed_configs:
        return [], errors, None

    # Chronological sort (Earliest image becomes the Reference Baseline t0)
    parsed_configs.sort(key=lambda x: x["decimal_year"])
    ref_image_bgr = parsed_configs[0]["img_bgr"]
    ref_scale = parsed_configs[0]["scale"]

    results = []

    max_align_length_change_frac = 0.08

    for idx, item in enumerate(parsed_configs):
        current_bgr = item["img_bgr"]
        current_scale = item["scale"]
        align_msg = "ภาพฐานอ้างอิง (Baseline)"
        coast = None

        # Apply ORB Homography Alignment if enabled and not the reference image
        if enable_alignment and idx > 0:
            aligned_bgr, success, align_msg = align_image_orb(current_bgr, ref_image_bgr)
            if success:
                # Sanity-check the alignment against its actual effect on the extracted
                # shoreline, not just the raw geometric warp -- a repetitive scene (e.g.
                # grid-like ponds/farmland) can produce a homography that looks small and
                # plausible in pixel-displacement terms yet still shifts the boundary
                # relative to the pond network, ballooning the traced shoreline length.
                # Comparing against this same image's own unaligned extraction (rather
                # than the baseline's) isolates exactly what alignment changed.
                gray_u, bm_u, _ = extract_binary_mask(current_bgr, segmentation_mode, blur_kernel, land_is_bright)
                coast_u = extract_clean_coastline(bm_u, denoise_kernel, keep_largest_only, channel_sever_kernel)

                gray_a, bm_a, _ = extract_binary_mask(aligned_bgr, segmentation_mode, blur_kernel, land_is_bright)
                coast_a = extract_clean_coastline(bm_a, denoise_kernel, keep_largest_only, channel_sever_kernel)

                length_u = coast_u["coastline_length_px"] if coast_u else 0.0
                length_a = coast_a["coastline_length_px"] if coast_a else 0.0
                length_change_frac = (
                    abs(length_a - length_u) / length_u if length_u > 0 else 1.0
                )

                if coast_a is not None and length_change_frac <= max_align_length_change_frac:
                    current_bgr = aligned_bgr
                    # Warped onto the reference's pixel grid, so the reference's scale now
                    # applies -- the image's own pre-alignment scale no longer matches its pixels.
                    current_scale = ref_scale
                    coast = coast_a
                else:
                    align_msg = (
                        f"Alignment ทำให้ความยาวแนวชายฝั่งที่สกัดได้เปลี่ยนผิดปกติ "
                        f"({length_change_frac * 100:.0f}%) อาจเกิดจากลวดลายซ้ำในภาพ (เช่น บ่อกุ้ง/นาเกลือ) "
                        f"หลอก Feature Matching — ใช้ภาพเดิม (ไม่ Alignment)"
                    )
                    coast = coast_u

        if coast is None:
            gray, binary_mask, otsu_val = extract_binary_mask(
                current_bgr, segmentation_mode, blur_kernel, land_is_bright
            )
            coast = extract_clean_coastline(binary_mask, denoise_kernel, keep_largest_only, channel_sever_kernel)

        if coast is None:
            errors.append(f"ไม่พบแนวชายฝั่งในภาพ '{item['file_name']}' หลังการประมวลผล — ข้ามภาพนี้")
            continue

        fd_res = robust_box_counting(coast["shoreline_img"])
        if fd_res is None:
            errors.append(f"คำนวณ Box-Counting ภาพ '{item['file_name']}' ไม่สำเร็จ — ข้ามภาพนี้")
            continue

        overlay = create_coastline_overlay(current_bgr, coast["shoreline_img"])

        results.append({
            "file_name": item["file_name"],
            "date": item["date"],
            "decimal_year": item["decimal_year"],
            "scale_m_per_px": current_scale,
            "align_status": align_msg,
            "original_shape": item["orig_shape"],
            "processed_shape": item["new_shape"],
            "was_resized": item["was_resized"],
            "display_img": current_bgr,
            "overlay_img": overlay,
            "shoreline_img": coast["shoreline_img"],
            "shore_pts": coast["shore_pts"],
            "shore_lines": coast["shore_lines"],
            "filled_land": coast["filled_land"],
            "land_area_px": coast["land_area_px"],
            "coastline_length_px": coast["coastline_length_px"],
            "fd": fd_res["fd"],
            "fd_r2": fd_res["r_squared"],
            "box_sizes": fd_res["box_sizes"],
            "counts": fd_res["counts"],
            "log_inv_r": fd_res["log_inv_r"],
            "log_n": fd_res["log_n"],
            "intercept": fd_res["intercept"],
        })

    if not results:
        return [], errors, None

    # Generate Virtual Transects based on the baseline image (t0)
    baseline = results[0]
    transects = generate_virtual_transects(baseline["shore_pts"], baseline["filled_land"], n_transects)

    return results, errors, transects


analyze_clicked = st.button(
    "เริ่มการวิเคราะห์แนวชายฝั่ง (Run Analyzer v5.0)",
    type="primary",
    disabled=(len(image_configs) == 0),
)

if analyze_clicked:
    try:
        with st.spinner("กำลังจัดตำแหน่งภาพ (Co-Registration) • สกัดแนวชายฝั่ง • สร้าง Virtual Transects • คำนวณ Fractal Dimension..."):
            res, errs, trans = run_pipeline(
                image_configs, blur_kernel, land_is_bright, segmentation_mode, max_dim,
                denoise_kernel, keep_largest_only, enable_alignment, n_transects, auto_crop_banner,
                channel_sever_kernel
            )
        st.session_state["results"] = res
        st.session_state["errors"] = errs
        st.session_state["transects"] = trans
        st.session_state["processed"] = True
    except Exception as exc:
        st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {exc}")
        st.session_state["processed"] = False


# ==============================================================================
# RESULTS DISPLAY & REPORTING
# ==============================================================================
if st.session_state.get("processed"):
    results = st.session_state.get("results", [])
    errors = st.session_state.get("errors", [])
    transects = st.session_state.get("transects")

    for err in errors:
        st.warning(err)

    if len(results) == 0:
        st.error("ไม่มีภาพใดประมวลผลสำเร็จ กรุณาตรวจสอบการตั้งค่าแล้วลองใหม่")
        st.stop()

    # --------------------------------------------------------------------
    # 2. IMAGE PREVIEWS & OVERLAYS
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. ภาพผลลัพธ์การสกัดแนวชายฝั่ง (Clean Shoreline Overlay)")

    for r in results:
        resize_info = f" (ย่อจาก {r['original_shape'][1]}x{r['original_shape'][0]})" if r["was_resized"] else ""
        st.markdown(f"**{r['date'].isoformat()}** — {r['file_name']}  |  `{r['align_status']}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(r["display_img"], channels="BGR", caption=f"ภาพที่ประมวลผล{resize_info}", use_container_width=True)
        with c2:
            st.image(r["filled_land"], caption="มาสก์แผ่นดินผืนหลัก (Solid Land)", use_container_width=True)
        with c3:
            st.image(r["overlay_img"], channels="BGR", caption="แนวชายฝั่งไร้ขอบเฟรม (True Shoreline)", use_container_width=True)

        st.caption(
            f"ความยาวแนวชายฝั่งจริง: **{r['coastline_length_px']:,.0f} px** (ประมาณ {r['coastline_length_px'] * r['scale_m_per_px']:,.1f} ม.)  •  "
            f"Fractal Dimension (FD): **{r['fd']:.4f}** (R² = {r['fd_r2']:.4f})"
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # CALCULATE TRANSECT METRICS FOR ALL TIMESTEPS
    # --------------------------------------------------------------------
    baseline = results[0]
    summary_rows = []
    latest_transect_details = []

    for idx, r in enumerate(results):
        row = {
            "Date": r["date"].isoformat(),
            "Decimal_Year": r["decimal_year"],
            "Coastline_Length_m": r["coastline_length_px"] * r["scale_m_per_px"],
            "FD": r["fd"],
            "FD_R2": r["fd_r2"],
        }

        if idx == 0:
            row["Mean_Shift_m"] = 0.0
            row["Median_Shift_m"] = 0.0
            row["Std_Dev_m"] = 0.0
            row["Cumulative_Shift_m"] = 0.0
            row["EPR_m_per_year"] = 0.0
            row["Max_Erosion_m"] = 0.0
            row["Max_Accretion_m"] = 0.0
            row["Status"] = "ข้อมูลฐานอ้างอิง (Baseline)"
        else:
            prev = results[idx - 1]
            dt_step = r["decimal_year"] - prev["decimal_year"]
            dt_base = r["decimal_year"] - baseline["decimal_year"]

            # Compute shift relative to previous step
            m_s, med_s, std_s, min_e, max_a, t_details = measure_transect_shifts(
                prev["shore_pts"], r["shore_pts"], transects, r["scale_m_per_px"]
            )

            # Compute cumulative shift relative to baseline (t0)
            cum_m, _, _, _, _, cum_t_details = measure_transect_shifts(
                baseline["shore_pts"], r["shore_pts"], transects, r["scale_m_per_px"]
            )

            epr = cum_m / dt_base if dt_base > 0 else 0.0

            row["Mean_Shift_m"] = m_s
            row["Median_Shift_m"] = med_s
            row["Std_Dev_m"] = std_s
            row["Cumulative_Shift_m"] = cum_m
            row["EPR_m_per_year"] = epr
            row["Max_Erosion_m"] = min_e
            row["Max_Accretion_m"] = max_a
            row["Status"] = classify_change(m_s)

            if idx == len(results) - 1:
                latest_transect_details = cum_t_details

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # --------------------------------------------------------------------
    # 3. INTERACTIVE MULTI-TEMPORAL SHORELINE MAP WITH TRANSECTS
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. แผนที่เปรียบเทียบแนวชายฝั่งทุกช่วงเวลา (Multi-Temporal Shoreline Map)")

    show_transects = st.checkbox("แสดงเส้นสำรวจตัดขวาง (Virtual Transects)", value=True)

    fig_map = go.Figure()
    colors = px.colors.qualitative.Bold

    # Draw shorelines
    for idx, r in enumerate(results):
        color = colors[idx % len(colors)]
        for l_idx, line in enumerate(r["shore_lines"]):
            fig_map.add_trace(go.Scatter(
                x=line[:, 0], y=line[:, 1],
                mode="lines",
                name=f"{r['date'].isoformat()}" if l_idx == 0 else None,
                showlegend=(l_idx == 0),
                line=dict(color=color, width=2.5),
                hoverinfo="text",
                text=f"ปี: {r['date'].isoformat()} | FD: {r['fd']:.4f}",
            ))

    # Draw virtual transects connecting baseline and latest shoreline
    if show_transects and latest_transect_details and transects:
        axis = transects["axis"]
        for td in latest_transect_details:
            t_coord = td["coord"]
            ref_v = td["ref_val"]
            tgt_v = td["tgt_val"]
            shift = td["shift_m"]

            if axis == 'y':
                x_pts = [ref_v, tgt_v]
                y_pts = [t_coord, t_coord]
            else:
                x_pts = [t_coord, t_coord]
                y_pts = [ref_v, tgt_v]

            t_color = "#d62728" if shift < -STABILITY_THRESHOLD_M else ("#2ca02c" if shift > STABILITY_THRESHOLD_M else "#7f7f7f")

            fig_map.add_trace(go.Scatter(
                x=x_pts, y=y_pts,
                mode="lines+markers",
                line=dict(color=t_color, width=1.5, dash="dot"),
                marker=dict(size=4),
                hoverinfo="text",
                text=f"Transect #{td['transect_id']}: {shift:+.2f} ม. ({td['status']})",
                showlegend=False,
            ))

    fig_map.update_layout(
        title="การขยับตัวของแนวชายฝั่งตามเส้นตัดขวาง (Transect Lines: แดง = กัดเซาะ, เขียว = งอก)",
        xaxis_title="พิกัดพิกเซล X",
        yaxis_title="พิกัดพิกเซล Y",
        yaxis=dict(autorange="reversed"),
        height=580,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # --------------------------------------------------------------------
    # 4. TRANSECT SPATIAL DISTRIBUTION ANALYSIS
    # --------------------------------------------------------------------
    if latest_transect_details:
        st.markdown("---")
        st.subheader("4. การกระจายตัวของการเปลี่ยนแปลงตามแนวชายฝั่ง (Spatial Transect Profile)")
        st.caption(f"เปรียบเทียบการเคลื่อนที่ระหว่างปีฐาน ({baseline['date'].isoformat()}) กับปีล่าสุด ({results[-1]['date'].isoformat()})")

        t_df = pd.DataFrame(latest_transect_details)
        fig_bars = px.bar(
            t_df, x="transect_id", y="shift_m",
            color="status",
            color_discrete_map={
                "กัดเซาะ (Erosion)": "#d62728",
                "งอก (Accretion)": "#2ca02c",
                "คงที่ (Stable)": "#7f7f7f",
            },
            labels={"transect_id": "หมายเลขเส้นสำรวจ (Transect ID)", "shift_m": "ระยะขยับตัวสุทธิ (เมตร)"},
            title="ระยะการกัดเซาะ/การงอกแยกรายเส้นสำรวจ (บวก = งอก, ลบ = กัดเซาะ)",
        )
        fig_bars.update_layout(height=380)
        st.plotly_chart(fig_bars, use_container_width=True)

    # --------------------------------------------------------------------
    # 5. SCIENTIFIC SUMMARY TABLE
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("5. ตารางสรุปผลข้อมูลเชิงปริมาณ (Shoreline Dynamics Summary)")

    display_table = summary_df.rename(columns={
        "Date": "วันที่",
        "Decimal_Year": "ปี (ทศนิยม)",
        "Mean_Shift_m": "ระยะเฉลี่ย (ม.)",
        "Median_Shift_m": "ระยะมัธยฐาน (ม.)",
        "Std_Dev_m": "ความไม่แน่นอน (±σ ม.)",
        "Cumulative_Shift_m": "ระยะสะสมจากปีฐาน (ม.)",
        "EPR_m_per_year": "EPR (ม./ปี)",
        "Max_Erosion_m": "กัดเซาะสูงสุด (ม.)",
        "Max_Accretion_m": "งอกสูงสุด (ม.)",
        "FD": "FD",
        "Status": "สถานะภาพรวม",
    })

    st.dataframe(
        display_table[[
            "วันที่", "ปี (ทศนิยม)", "ระยะเฉลี่ย (ม.)", "ความไม่แน่นอน (±σ ม.)",
            "ระยะสะสมจากปีฐาน (ม.)", "EPR (ม./ปี)", "กัดเซาะสูงสุด (ม.)", "งอกสูงสุด (ม.)", "FD", "สถานะภาพรวม"
        ]].style.format({
            "ปี (ทศนิยม)": "{:.3f}",
            "ระยะเฉลี่ย (ม.)": "{:+.2f}",
            "ความไม่แน่นอน (±σ ม.)": "±{:.2f}",
            "ระยะสะสมจากปีฐาน (ม.)": "{:+.2f}",
            "EPR (ม./ปี)": "{:+.2f}",
            "กัดเซาะสูงสุด (ม.)": "{:+.2f}",
            "งอกสูงสุด (ม.)": "{:+.2f}",
            "FD": "{:.4f}",
        }),
        use_container_width=True,
    )

    # --------------------------------------------------------------------
    # 6. BOX-COUNTING DIAGNOSTIC
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("6. การตรวจสอบ Box-Counting (Morphological Complexity Diagnostic)")
    file_labels = [f"{r['date'].isoformat()} — {r['file_name']}" for r in results]
    selected_label = st.selectbox("เลือกภาพเพื่อตรวจสอบกราฟ Log-Log Regression:", file_labels)
    selected_idx = file_labels.index(selected_label)
    sel = results[selected_idx]

    fitted_y = np.array(sel["log_inv_r"]) * sel["fd"] + sel["intercept"]

    fig_diag = go.Figure()
    fig_diag.add_trace(go.Scatter(
        x=sel["log_inv_r"], y=sel["log_n"], mode="markers", name="จุดข้อมูลจริง log N(r)",
        marker=dict(size=10, color="#1f77b4"),
    ))
    fig_diag.add_trace(go.Scatter(
        x=sel["log_inv_r"], y=fitted_y, mode="lines",
        name=f"เส้นสมการถดถอย (FD = {sel['fd']:.4f})",
        line=dict(color="#d62728", dash="dash"),
    ))
    fig_diag.update_layout(
        title=f"Log-Log Plot ของ Box-Counting — {sel['file_name']} (R² = {sel['fd_r2']:.4f})",
        xaxis_title="log(1/r) [ความละเอียดของกล่องสำรวจ]",
        yaxis_title="log(N(r)) [จำนวนกล่องที่คลุมแนวชายฝั่ง]",
        height=400,
    )
    st.plotly_chart(fig_diag, use_container_width=True)

    st.info(
        "**คำแนะนำทางวิทยาศาสตร์เกี่ยวกับค่า FD:** Fractal Dimension สะท้อน **'ความขรุขระและความซับซ้อนเชิงเรขาคณิต'** ของแนวชายฝั่ง "
        "ค่า FD สูงหมายถึงชายฝั่งเว้าแหว่งหรือมีหัวแหลมซับซ้อน ค่า FD ต่ำเข้าใกล้ 1.0 หมายถึงชายหาดทอดตัวเป็นเส้นตรงเรียบ "
        "**ค่า FD ไม่ได้แปลว่าเกิดการกัดเซาะหรือการงอกโดยตรง** การประเมินการเคลื่อนที่ทางกายภาพต้องพิจารณาจากผลลัพธ์ของ Virtual Transects ในข้อ 4 และ 5"
    )

    # --------------------------------------------------------------------
    # 7. DIRECT SPATIAL & FD FORECASTING
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("7. การพยากรณ์การเคลื่อนตัวของแนวชายฝั่งล่วงหน้า (Linear Trend Forecasting)")

    if len(results) < MIN_IMAGES_FOR_TRENDS:
        st.warning(f"ต้องใช้ข้อมูลภาพอย่างน้อย {MIN_IMAGES_FOR_TRENDS} ช่วงเวลาเพื่อสร้างแบบจำลองแนวโน้ม")
    else:
        years = summary_df["Decimal_Year"].values.reshape(-1, 1)
        cum_shifts = summary_df["Cumulative_Shift_m"].values
        fds = summary_df["FD"].values

        # 1. Spatial Trend Model (Cumulative Transect Shift vs Year)
        shift_model = LinearRegression()
        shift_model.fit(years, cum_shifts)
        r2_shift = shift_model.score(years, cum_shifts)

        # 2. FD Trend Model
        fd_model = LinearRegression()
        fd_model.fit(years, fds)
        r2_fd = fd_model.score(years, fds)

        last_year = float(years.max())

        col_base, col_horizon = st.columns(2)
        with col_base:
            base_year = st.number_input(
                "ปีฐานสำหรับเริ่มพยากรณ์",
                min_value=1900, max_value=2200, value=int(round(last_year)), step=1,
            )
        with col_horizon:
            horizon = st.number_input(
                "ระยะเวลาที่ต้องการพยากรณ์ล่วงหน้า (ปี)",
                min_value=1, max_value=30, value=5, step=1,
            )

        forecast_rows = []
        base_f = float(base_year)
        prev_cum = float(shift_model.predict([[base_f]])[0])

        for n in range(1, int(horizon) + 1):
            future_y = base_f + n
            pred_cum = float(shift_model.predict([[future_y]])[0])
            pred_fd = float(fd_model.predict([[future_y]])[0])
            inc_shift = pred_cum - prev_cum
            prev_cum = pred_cum

            forecast_rows.append({
                "Year": int(base_year + n),
                "Years_Ahead": f"+{n} ปี",
                "Cumulative_Distance_m": pred_cum,
                "Distance_Change_m": inc_shift,
                "Predicted_FD": pred_fd,
                "Status": classify_change(inc_shift),
            })

        forecast_df = pd.DataFrame(forecast_rows)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            first_f = forecast_df.iloc[0]
            st.metric(
                f"ปีถัดไป (+1 ปี, {first_f['Year']})",
                f"{first_f['Distance_Change_m']:+.2f} ม./ปี",
                delta=f"{first_f['Status']}",
            )
        with col_m2:
            last_f = forecast_df.iloc[-1]
            st.metric(
                f"อีก {horizon} ปี ({last_f['Year']})",
                f"ระยะสะสม {last_f['Cumulative_Distance_m']:+.2f} ม.",
                delta=f"{last_f['Status']}",
            )

        st.caption(
            f"สมการอัตราการเปลี่ยนแปลงเฉลี่ย: Shift = {shift_model.coef_[0]:+.3f} ม./ปี × Year + ({shift_model.intercept_:.2f}) (R² = {r2_shift:.4f})"
        )
