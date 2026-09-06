"""
================================================================================
 Coastal Erosion / Accretion Analyzer — Box-Counting Fractal Dimension Engine
================================================================================

A Streamlit application for analyzing and forecasting coastal erosion or
accretion trends from multi-temporal coastal imagery (e.g., Landsat /
Sentinel-2 NDWI composites) using the Box-Counting Fractal Dimension (FD)
method combined with linear-regression based forecasting.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
1. Install dependencies:
   pip install streamlit opencv-python numpy pandas scikit-learn plotly pillow

2. Run the app:
   streamlit run app.py

--------------------------------------------------------------------------------
METHODOLOGY (SUMMARY)
--------------------------------------------------------------------------------
1. Each uploaded image is converted to grayscale, Gaussian-blurred, and
   binarized with Otsu's automatic thresholding to separate Land vs. Water.
2. The coastline boundary (1-pixel wide) is extracted either via contour
   tracing of the binary mask or via Canny edge detection.
3. The Box-Counting algorithm covers the boundary image with square grids of
   decreasing size r = 2^k and counts the number of boxes N(r) that contain at
   least one boundary pixel. The Fractal Dimension (FD) is the slope of the
   linear regression of log(N(r)) versus log(1/r).
4. Physical (metric) shoreline change between consecutive dates is estimated
   from the change in land area, normalized by coastline length, then scaled
   by the user-supplied spatial resolution (meters/pixel).
5. A linear regression of FD vs. time is used to forecast FD values 5 and 10
   years into the future. The historical relationship between FD change and
   physical shoreline displacement is used to translate the forecast FD
   values into estimated future retreat / advance distances (in meters).

NOTE: This tool provides a simplified geomorphological *estimate* based on
image-processing heuristics. It is intended for exploratory/educational
analysis and should be validated against ground-truth surveys before being
used for engineering or policy decisions.
================================================================================
"""

import warnings
from datetime import date, datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTS
# ==============================================================================
MAX_DIM_DEFAULT = 1024          # Default max processing dimension (pixels)
STABILITY_THRESHOLD_M = 0.01    # Below this |distance| (m), classify as "Stable"
MIN_IMAGES_FOR_TRENDS = 2       # Minimum images required for change/forecast analysis
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_date_input(text: str):
    """
    Parse a user-supplied date string that may be a full date (YYYY-MM-DD, or
    many other common formats via pandas) or just a bare year (YYYY).

    Returns a `datetime.date` object, or None if parsing fails.
    """
    text = (text or "").strip()
    if not text:
        return None

    # First, try a bare year (e.g. "2020").
    if text.isdigit() and 3 <= len(text) <= 4:
        try:
            year = int(text)
            if 1000 <= year <= 9999:
                return date(year, 1, 1)
        except ValueError:
            pass

    # Otherwise, let pandas try to parse a full date string.
    try:
        parsed = pd.to_datetime(text)
        return parsed.date()
    except Exception:
        return None


def to_decimal_year(d: date) -> float:
    """Convert a `datetime.date` into a decimal-year float, e.g. 2020-07-02 -> 2020.501"""
    year_start = date(d.year, 1, 1)
    year_end = date(d.year + 1, 1, 1)
    year_length_days = (year_end - year_start).days
    days_into_year = (d - year_start).days
    return d.year + (days_into_year / year_length_days)


def load_image_bgr(uploaded_file) -> np.ndarray:
    """
    Robustly load an uploaded file into a BGR uint8 numpy array.
    Tries OpenCV's decoder first (fast, handles most formats), and falls
    back to PIL for formats OpenCV cannot decode (e.g. some TIFF variants).
    """
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    file_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        # Fallback: use PIL, then convert RGB -> BGR for OpenCV consistency.
        uploaded_file.seek(0)
        pil_img = Image.open(uploaded_file).convert("RGB")
        rgb_array = np.array(pil_img)
        img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError("Could not decode image file.")

    return img


def resize_for_processing(img_bgr: np.ndarray, original_scale: float, max_dim: int):
    """
    Downscale an image if it exceeds `max_dim` on its longest side, to keep
    processing fast and memory-bounded. The scale factor (meters/pixel) is
    adjusted proportionally so all downstream physical-distance math remains
    correct in real-world units.

    Returns: (resized_img_bgr, effective_scale_m_per_px, was_resized, original_shape, new_shape)
    """
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


def extract_binary_mask(img_bgr: np.ndarray, blur_kernel: int, land_is_bright: bool):
    """
    Convert to grayscale, apply Gaussian blur, and binarize using Otsu's method.
    Ensures the returned mask always encodes LAND = 255, WATER = 0, regardless
    of which side of the Otsu split was naturally brighter in the source image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    otsu_thresh_val, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if not land_is_bright:
        binary = cv2.bitwise_not(binary)

    return gray, blurred, binary, otsu_thresh_val


def extract_clean_coastline(binary_mask: np.ndarray, denoise_kernel: int, keep_largest_only: bool,
                             method: str, canny_low: int, canny_high: int):
    """
    Clean the LAND/WATER binary mask (morphological opening + closing) to remove
    speckle noise and small holes, then trace the coastline contour(s) on the
    cleaned mask.

    To keep physical measurements (land area, coastline length) robust to noise,
    they are ALWAYS computed geometrically from the contour(s) — cv2.contourArea /
    cv2.arcLength — rather than by counting raw pixels. This avoids wildly
    inflated erosion/accretion distances caused by stray noise pixels or small
    unrelated islands elsewhere in the frame.

    If `keep_largest_only` is True, only the single largest contour (assumed to
    be the main coastline) is kept, discarding small islands/artifacts entirely.
    If False, any contour with area >= 1% of the largest contour's area is kept.

    Returns a dict with: cleaned_mask, edge_img, land_area_px, coastline_length_px,
    contours_used. Returns None if no usable contour is found.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (denoise_kernel, denoise_kernel))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    if keep_largest_only:
        contours_used = [max(contours, key=cv2.contourArea)]
    else:
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas) if areas else 0.0
        if max_area <= 0:
            return None
        contours_used = [c for c, a in zip(contours, areas) if a >= 0.01 * max_area]

    land_area_px = float(sum(cv2.contourArea(c) for c in contours_used))
    coastline_length_px = float(sum(cv2.arcLength(c, True) for c in contours_used))

    if method == "Canny Edge Detection":
        # Run Canny on a filled version of the SAME clean contour(s), so the
        # visualization method never reintroduces the noise we just removed.
        filled_mask = np.zeros_like(cleaned)
        cv2.drawContours(filled_mask, contours_used, -1, 255, thickness=-1)
        edge_img = cv2.Canny(filled_mask, canny_low, canny_high)
    else:  # "Contour Boundary"
        edge_img = np.zeros_like(cleaned)
        cv2.drawContours(edge_img, contours_used, -1, 255, thickness=1)

    return {
        "cleaned_mask": cleaned,
        "edge_img": edge_img,
        "land_area_px": land_area_px,
        "coastline_length_px": coastline_length_px,
        "contours_used": contours_used,
    }


def box_counting_fd(binary_edge_img: np.ndarray):
    """
    Compute the Box-Counting Fractal Dimension of a binary boundary image.

    Algorithm:
      1. Pad the image to a square whose side is the next power of two.
      2. For box sizes r = 2^1, 2^2, ..., 2^(p-1) (p = padded exponent),
         tile the image into non-overlapping r x r boxes and count how many
         boxes contain at least one boundary (True) pixel -> N(r).
      3. FD = slope of the least-squares linear fit of log(N(r)) vs log(1/r).

    Returns a dict with box_sizes, counts, fd (slope), intercept, r_squared,
    log_inv_r, log_n. Returns None if the image has no boundary pixels or is
    too small for a meaningful multi-scale fit.
    """
    bool_img = binary_edge_img > 0
    if not np.any(bool_img):
        return None

    h, w = bool_img.shape
    longest_side = max(h, w)
    p = int(np.ceil(np.log2(max(longest_side, 2))))
    padded_size = 2 ** p

    padded = np.zeros((padded_size, padded_size), dtype=bool)
    padded[:h, :w] = bool_img

    box_sizes = [2 ** k for k in range(1, p)]  # r = 2, 4, 8, ..., padded_size/2
    if len(box_sizes) < 2:
        return None

    counts = []
    for r in box_sizes:
        n_boxes_axis = padded_size // r
        cropped = padded[: n_boxes_axis * r, : n_boxes_axis * r]
        reshaped = cropped.reshape(n_boxes_axis, r, n_boxes_axis, r)
        box_has_pixel = reshaped.any(axis=(1, 3))
        counts.append(int(box_has_pixel.sum()))

    counts_arr = np.array(counts, dtype=np.float64)
    box_sizes_arr = np.array(box_sizes, dtype=np.float64)

    log_inv_r = np.log(1.0 / box_sizes_arr)
    log_n = np.log(counts_arr)

    slope, intercept = np.polyfit(log_inv_r, log_n, 1)

    y_pred = slope * log_inv_r + intercept
    ss_res = np.sum((log_n - y_pred) ** 2)
    ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "box_sizes": box_sizes,
        "counts": counts_arr.tolist(),
        "log_inv_r": log_inv_r,
        "log_n": log_n,
        "fd": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def classify_change(distance_m: float) -> str:
    """Classify a physical distance change into Erosion / Accretion / Stable."""
    if abs(distance_m) < STABILITY_THRESHOLD_M:
        return "Stable"
    return "Accretion" if distance_m > 0 else "Erosion"


# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================
st.set_page_config(
    page_title="Coastal Erosion Analyzer — Fractal Dimension",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Coastal Erosion / Accretion Analyzer")
st.caption("Box-Counting Fractal Dimension Method — Multi-Temporal Coastline Analysis & Forecasting")

with st.expander("ℹ️ About this tool / Methodology", expanded=False):
    st.markdown(
        """
This tool analyzes a **time series of coastal images** (e.g., NDWI composites
from Landsat or Sentinel-2) to:

1. Extract the **Land/Water boundary (coastline)** from each image using
   Otsu thresholding plus contour or Canny edge extraction.
2. Quantify the **geometric complexity** of each coastline using the
   **Box-Counting Fractal Dimension (FD)** — a higher FD generally indicates a
   more complex / irregular coastline shape.
3. Estimate **physical erosion/accretion (in meters)** between consecutive
   dates from the change in land area, normalized by coastline length and
   scaled by your image resolution.
4. **Forecast** future FD values and translate the trend into projected
   shoreline retreat/advance distances at +5 and +10 years.

*This is an educational/exploratory estimate, not a substitute for a certified
coastal survey.*
        """
    )

# ==============================================================================
# SIDEBAR — PROCESSING PARAMETERS
# ==============================================================================
st.sidebar.header("⚙️ Processing Parameters")

blur_kernel = st.sidebar.slider(
    "Gaussian Blur Kernel Size (odd)", min_value=3, max_value=15, value=5, step=2,
    help="Larger values smooth out more noise before thresholding, at the cost of fine detail."
)

land_choice = st.sidebar.radio(
    "Which pixel intensity represents LAND after thresholding?",
    ["Bright (255) = Land", "Dark (0) = Land"],
    index=0,
    help="Depends on your NDWI/band convention. If results look inverted, switch this."
)
land_is_bright = (land_choice == "Bright (255) = Land")

denoise_kernel = st.sidebar.slider(
    "ระดับลด Noise / เกาะเล็ก (Denoise Kernel Size, odd)", min_value=3, max_value=25, value=9, step=2,
    help="ค่ายิ่งสูง ยิ่งลบจุดรบกวนเล็กๆ (speckle noise) และรูเล็กๆ ออกจากมาสก์มากขึ้น "
         "(ใช้ Morphological Opening ตามด้วย Closing) ช่วยให้ค่าพื้นที่/ความยาวชายฝั่งแม่นยำขึ้น"
)

keep_largest_only = st.sidebar.checkbox(
    "ใช้เฉพาะแนวชายฝั่งหลัก (ตัด noise/เกาะเล็กทิ้ง)", value=True,
    help="แนะนำให้เปิดไว้ — คำนวณพื้นที่และความยาวชายฝั่งจากรูปร่างที่ใหญ่ที่สุดเพียงรูปเดียวเท่านั้น "
         "ลดค่าระยะกัดเซาะ/งอกที่คลาดเคลื่อนสูงเกินจริงจาก noise หรือเกาะเล็กที่ไม่เกี่ยวข้อง"
)

edge_method = st.sidebar.radio(
    "Coastline Boundary Extraction Method",
    ["Contour Boundary", "Canny Edge Detection"],
    index=0,
)

if edge_method == "Canny Edge Detection":
    canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
    canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
else:
    canny_low, canny_high = 50, 150  # unused defaults when contour method is selected

max_dim = st.sidebar.slider(
    "Max Processing Dimension (px)", min_value=256, max_value=4096, value=MAX_DIM_DEFAULT, step=128,
    help="Images larger than this are downscaled for performance. The spatial scale is "
         "automatically adjusted so meter-based results remain accurate."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Upload images in roughly chronological order — the app will auto-suggest "
    "sequential default years, which you can edit."
)

# ==============================================================================
# FILE UPLOAD & PER-IMAGE METADATA INPUT
# ==============================================================================
st.subheader("📤 1. Upload Coastal Images")

uploaded_files = st.file_uploader(
    "Upload multiple coastal images (NDWI / Landsat / Sentinel-2 or any Land-Water image)",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

image_configs = []

if uploaded_files:
    st.markdown("#### 📝 Configure Date & Scale for Each Image")
    n_files = len(uploaded_files)
    current_year = datetime.now().year

    for i, up_file in enumerate(uploaded_files):
        with st.expander(f"🖼️ Image {i + 1}: {up_file.name}", expanded=(n_files <= 3)):
            col1, col2 = st.columns(2)
            with col1:
                default_year = current_year - (n_files - 1 - i)
                date_str = st.text_input(
                    "Date / Year (YYYY-MM-DD or YYYY)",
                    value=str(default_year),
                    key=f"date_input_{i}_{up_file.name}",
                )
            with col2:
                scale_val = st.number_input(
                    "Spatial Resolution Scale (meters/pixel)",
                    min_value=0.0001, value=1.0, step=0.1, format="%.4f",
                    key=f"scale_input_{i}_{up_file.name}",
                )
            image_configs.append({"file": up_file, "date_str": date_str, "scale": scale_val})
else:
    st.info("👆 Upload two or more coastal images to begin (chronological order recommended).")


# ==============================================================================
# ANALYSIS PIPELINE
# ==============================================================================
def run_pipeline(configs, blur_kernel, land_is_bright, edge_method, canny_low, canny_high, max_dim,
                  denoise_kernel, keep_largest_only):
    """Run the full image-processing + FD pipeline over all configured images."""
    results = []
    errors = []

    for cfg in configs:
        up_file = cfg["file"]
        d = parse_date_input(cfg["date_str"])
        if d is None:
            errors.append(f"⚠️ Could not parse date '{cfg['date_str']}' for '{up_file.name}' — skipped.")
            continue

        try:
            img_bgr = load_image_bgr(up_file)
        except Exception as exc:
            errors.append(f"⚠️ Could not read image '{up_file.name}' ({exc}) — skipped.")
            continue

        resized_img, eff_scale, was_resized, orig_shape, new_shape = resize_for_processing(
            img_bgr, cfg["scale"], max_dim
        )

        gray, blurred, binary_mask, otsu_val = extract_binary_mask(resized_img, blur_kernel, land_is_bright)

        coast_result = extract_clean_coastline(
            binary_mask, denoise_kernel, keep_largest_only, edge_method, canny_low, canny_high
        )
        if coast_result is None:
            errors.append(
                f"⚠️ No coastline boundary detected in '{up_file.name}'. "
                f"Try adjusting blur/threshold/denoise parameters — skipped."
            )
            continue

        cleaned_mask = coast_result["cleaned_mask"]
        edge_img = coast_result["edge_img"]
        land_area_px = coast_result["land_area_px"]
        coastline_length_px = coast_result["coastline_length_px"]

        if coastline_length_px <= 0 or int(np.count_nonzero(edge_img)) == 0:
            errors.append(
                f"⚠️ Coastline boundary in '{up_file.name}' collapsed after cleanup. "
                f"Try lowering the Denoise Kernel Size — skipped."
            )
            continue

        fd_result = box_counting_fd(edge_img)
        if fd_result is None:
            errors.append(f"⚠️ Image '{up_file.name}' too small/uniform for box-counting — skipped.")
            continue

        results.append({
            "file_name": up_file.name,
            "date": d,
            "decimal_year": to_decimal_year(d),
            "scale_m_per_px": eff_scale,
            "original_scale_m_per_px": cfg["scale"],
            "was_resized": was_resized,
            "original_shape": orig_shape,
            "processed_shape": new_shape,
            "original_img_bgr": resized_img,
            "gray_img": gray,
            "binary_mask": binary_mask,
            "cleaned_mask": cleaned_mask,
            "edge_img": edge_img,
            "otsu_val": float(otsu_val),
            "land_area_px": land_area_px,
            "coastline_length_px": coastline_length_px,
            "fd": fd_result["fd"],
            "intercept": fd_result["intercept"],
            "fd_r_squared": fd_result["r_squared"],
            "box_sizes": fd_result["box_sizes"],
            "counts": fd_result["counts"],
            "log_inv_r": fd_result["log_inv_r"],
            "log_n": fd_result["log_n"],
        })

    # Sort chronologically — essential for correct interval-based change detection.
    results.sort(key=lambda r: r["decimal_year"])
    return results, errors


analyze_clicked = st.button(
    "🔍 Analyze Coastlines & Compute Fractal Dimensions",
    type="primary",
    disabled=(len(image_configs) == 0),
)

if analyze_clicked:
    try:
        with st.spinner("Processing images — extracting coastlines & computing fractal dimensions..."):
            results, errors = run_pipeline(
                image_configs, blur_kernel, land_is_bright, edge_method, canny_low, canny_high, max_dim,
                denoise_kernel, keep_largest_only,
            )
        st.session_state["results"] = results
        st.session_state["errors"] = errors
        st.session_state["processed"] = True
    except Exception as exc:
        st.error(f"An unexpected error occurred during processing: {exc}")
        st.session_state["processed"] = False


# ==============================================================================
# RESULTS DISPLAY
# ==============================================================================
if st.session_state.get("processed"):
    results = st.session_state.get("results", [])
    errors = st.session_state.get("errors", [])

    for err in errors:
        st.warning(err)

    if len(results) == 0:
        st.error("No images were successfully processed. Please check your inputs and try again.")
        st.stop()

    # --------------------------------------------------------------------
    # 2. IMAGE PROCESSING PREVIEWS
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🖼️ 2. Image Processing Previews")

    for r in results:
        resize_note = ""
        if r["was_resized"]:
            resize_note = (
                f", auto-resized from {r['original_shape'][1]}x{r['original_shape'][0]} "
                f"to {r['processed_shape'][1]}x{r['processed_shape'][0]}"
            )
        st.markdown(
            f"**{r['file_name']}** — {r['date'].isoformat()} "
            f"(Scale: {r['scale_m_per_px']:.4f} m/px{resize_note})"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(r["original_img_bgr"], channels="BGR", caption="Original Image", use_container_width=True)
        with c2:
            st.image(
                r["cleaned_mask"], caption=f"Binary Mask — Cleaned (Otsu={r['otsu_val']:.1f})",
                use_container_width=True,
            )
        with c3:
            st.image(r["edge_img"], caption="Extracted Coastline (Main Contour Only)", use_container_width=True)
        st.caption(
            f"Land Area: {r['land_area_px']:,.0f} px  •  "
            f"Coastline Length: {r['coastline_length_px']:,.0f} px  •  "
            f"Fractal Dimension (FD): **{r['fd']:.4f}** (R² = {r['fd_r_squared']:.4f})"
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # BUILD SUMMARY DATAFRAME + INTERVAL-BASED PHYSICAL DISTANCE CALC
    # --------------------------------------------------------------------
    rows = []
    for idx, r in enumerate(results):
        row = {
            "Date": r["date"].isoformat(),
            "Decimal_Year": r["decimal_year"],
            "Land_Area_px": r["land_area_px"],
            "Coastline_Length_px": r["coastline_length_px"],
            "Scale_m_per_px": r["scale_m_per_px"],
            "FD": r["fd"],
        }
        if idx == 0:
            row["Distance_Change_m"] = np.nan
            row["Status"] = "Baseline (Reference)"
        else:
            prev = results[idx - 1]
            area_diff_px = r["land_area_px"] - prev["land_area_px"]
            avg_coastline_len = float(np.mean([r["coastline_length_px"], prev["coastline_length_px"]]))
            avg_scale = float(np.mean([r["scale_m_per_px"], prev["scale_m_per_px"]]))
            shift_px = area_diff_px / avg_coastline_len if avg_coastline_len > 0 else 0.0
            shift_m = shift_px * avg_scale
            row["Distance_Change_m"] = shift_m
            row["Status"] = classify_change(shift_m)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # --------------------------------------------------------------------
    # 3. STRUCTURED SUMMARY TABLE
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 3. Structured Summary Table")
    st.dataframe(
        summary_df.style.format(
            {
                "Decimal_Year": "{:.3f}",
                "Scale_m_per_px": "{:.4f}",
                "FD": "{:.4f}",
                "Distance_Change_m": "{:+.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------------------
    # 4. BOX-COUNTING DIAGNOSTIC (LOG-LOG REGRESSION)
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📐 4. Box-Counting Diagnostic (Log-Log Regression)")
    file_labels = [f"{r['date'].isoformat()} — {r['file_name']}" for r in results]
    selected_label = st.selectbox("Select an image to inspect its box-counting fit:", file_labels)
    selected_idx = file_labels.index(selected_label)
    sel = results[selected_idx]

    fitted_y = sel["fd"] * sel["log_inv_r"] + sel["intercept"]

    fig_diag = go.Figure()
    fig_diag.add_trace(go.Scatter(
        x=sel["log_inv_r"], y=sel["log_n"], mode="markers", name="log N(r) data points",
        marker=dict(size=10, color="#1f77b4"),
    ))
    fig_diag.add_trace(go.Scatter(
        x=sel["log_inv_r"], y=fitted_y, mode="lines",
        name=f"Linear Fit (FD = slope = {sel['fd']:.4f})",
        line=dict(color="#d62728", dash="dash"),
    ))
    fig_diag.update_layout(
        title=f"Box-Counting Log-Log Plot — {sel['file_name']} (R² = {sel['fd_r_squared']:.4f})",
        xaxis_title="log(1/r)", yaxis_title="log(N(r))",
        height=450,
    )
    st.plotly_chart(fig_diag, use_container_width=True)

    st.dataframe(
        pd.DataFrame({
            "Box Size r (px)": sel["box_sizes"],
            "Box Count N(r)": [int(c) for c in sel["counts"]],
        }),
        use_container_width=True,
    )

    # --------------------------------------------------------------------
    # 5. EROSION VS. ACCRETION — INTERVAL CHANGES
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 5. Erosion vs. Accretion — Interval Changes")

    if len(results) < MIN_IMAGES_FOR_TRENDS:
        st.info(f"Upload at least {MIN_IMAGES_FOR_TRENDS} images with valid dates to compute erosion/accretion trends.")
    else:
        interval_labels = []
        interval_distances = []
        for idx in range(1, len(results)):
            prev, curr = results[idx - 1], results[idx]
            interval_labels.append(f"{prev['date'].isoformat()} → {curr['date'].isoformat()}")
            interval_distances.append(float(summary_df.iloc[idx]["Distance_Change_m"]))

        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in interval_distances]
        fig_bar = go.Figure(go.Bar(
            x=interval_labels, y=interval_distances, marker_color=bar_colors,
            text=[f"{v:+.2f} m" for v in interval_distances], textposition="auto",
        ))
        fig_bar.update_layout(
            title="Interval-wise Coastline Change (Green = Accretion, Red = Erosion)",
            xaxis_title="Period", yaxis_title="Distance Change (m)", height=450,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        for idx in range(1, len(results)):
            v = float(summary_df.iloc[idx]["Distance_Change_m"])
            status = summary_df.iloc[idx]["Status"]
            d_label = results[idx]["date"].isoformat()
            if status == "Erosion":
                st.error(f"📉 {d_label}: **Erosion of {abs(v):.3f} m**")
            elif status == "Accretion":
                st.success(f"📈 {d_label}: **Accretion of {abs(v):.3f} m**")
            else:
                st.info(f"➖ {d_label}: **Stable** (Δ = {v:+.3f} m)")

    # --------------------------------------------------------------------
    # 6. FRACTAL DIMENSION TREND & YEAR-BY-YEAR FORECAST
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 6. Fractal Dimension Trend & Forecast (Year-by-Year)")

    if len(results) < MIN_IMAGES_FOR_TRENDS:
        st.info(f"Upload at least {MIN_IMAGES_FOR_TRENDS} images to enable forecasting.")
        forecast_df = pd.DataFrame()
    else:
        years_arr = np.array([r["decimal_year"] for r in results]).reshape(-1, 1)
        fd_arr = np.array([r["fd"] for r in results])

        model = LinearRegression()
        model.fit(years_arr, fd_arr)
        r2_trend = model.score(years_arr, fd_arr)

        last_year_detected = float(years_arr.max())

        col_base, col_horizon = st.columns(2)
        with col_base:
            base_year = st.number_input(
                "ปีฐานสำหรับเริ่มพยากรณ์ (Base Year to Forecast From)",
                min_value=1900, max_value=2200,
                value=int(round(last_year_detected)),
                step=1,
                help="ค่าเริ่มต้นคือปีของภาพล่าสุดที่อัปโหลด สามารถแก้ไขได้หากต้องการพยากรณ์จากปีอื่น "
                     "(เช่น ปีปัจจุบัน) โดยยังใช้แนวโน้ม FD ที่คำนวณจากข้อมูลย้อนหลังทั้งหมด",
            )
        with col_horizon:
            horizon = st.number_input(
                "พยากรณ์ล่วงหน้ากี่ปี (Forecast Horizon, years)",
                min_value=1, max_value=30, value=5, step=1,
            )

        base_year_f = float(base_year)
        baseline_fd = float(model.predict([[base_year_f]])[0])

        # ---- Empirical FD-change -> physical-distance scaling factor ----
        # Derived from historical intervals: average (distance_m / delta_fd) ratio.
        interval_delta_fd = np.diff(fd_arr)
        interval_distance_m = summary_df["Distance_Change_m"].values[1:].astype(float)
        valid_mask = np.abs(interval_delta_fd) > 1e-9

        if np.any(valid_mask):
            k_ratio = float(np.mean(interval_distance_m[valid_mask] / interval_delta_fd[valid_mask]))
        else:
            k_ratio = 0.0

        if len(results) < 3:
            st.caption("⚠️ Forecast is based on limited data (fewer than 3 images) — interpret with caution.")

        # ---- Build year-by-year forecast rows: base_year+1 ... base_year+horizon ----
        forecast_rows = []
        prev_cumulative = 0.0
        for n in range(1, int(horizon) + 1):
            forecast_year = base_year_f + n
            predicted_fd_n = float(model.predict([[forecast_year]])[0])
            delta_fd_n = predicted_fd_n - baseline_fd
            cumulative_distance_n = k_ratio * delta_fd_n
            incremental_distance_n = cumulative_distance_n - prev_cumulative
            prev_cumulative = cumulative_distance_n

            forecast_rows.append({
                "Date": f"Forecast +{n}y ({int(base_year + n)})",
                "Years_Ahead": n,
                "Year": base_year + n,
                "FD": predicted_fd_n,
                "Cumulative_Distance_m": cumulative_distance_n,
                "Distance_Change_m": incremental_distance_n,
                "Status": classify_change(incremental_distance_n),
            })

        forecast_df = pd.DataFrame(forecast_rows)

        # ---- Quick-glance metrics: next year vs. final forecast year ----
        first_row = forecast_df.iloc[0]
        last_row = forecast_df.iloc[-1]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                f"ปีถัดไป (+1 ปี, {int(first_row['Year'])})",
                f"{first_row['FD']:.4f}",
                delta=f"{first_row['Distance_Change_m']:+.3f} m ({first_row['Status']})",
            )
        with col_b:
            st.metric(
                f"อีก {int(horizon)} ปี ({int(last_row['Year'])})",
                f"{last_row['FD']:.4f}",
                delta=f"รวม {last_row['Cumulative_Distance_m']:+.3f} m ({last_row['Status']})",
            )

        st.caption(
            f"Trend model: FD = {model.coef_[0]:.6f} × Year + {model.intercept_:.6f}   (R² = {r2_trend:.4f})"
        )

        # ---- Table: year-by-year forecast ----
        st.markdown("##### พยากรณ์รายปี (Year-by-Year Forecast)")
        display_forecast = forecast_df[
            ["Years_Ahead", "Year", "FD", "Cumulative_Distance_m", "Distance_Change_m", "Status"]
        ].rename(columns={
            "Years_Ahead": "+ปี",
            "Year": "ปี ค.ศ.",
            "FD": "FD ที่คาดการณ์",
            "Cumulative_Distance_m": "ระยะสะสมจากปีฐาน (ม.)",
            "Distance_Change_m": "เปลี่ยนแปลงจากปีก่อนหน้า (ม.)",
            "Status": "สถานะ",
        })
        st.dataframe(
            display_forecast.style.format({
                "FD ที่คาดการณ์": "{:.4f}",
                "ระยะสะสมจากปีฐาน (ม.)": "{:+.3f}",
                "เปลี่ยนแปลงจากปีก่อนหน้า (ม.)": "{:+.3f}",
            }),
            use_container_width=True,
        )

        # ---- Chart 1: FD historical trend + forecast curve ----
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=[r["decimal_year"] for r in results], y=fd_arr, mode="markers+lines",
            name="Historical FD", line=dict(color="#1f77b4"), marker=dict(size=9),
        ))
        trend_x = [float(years_arr.min()), float(years_arr.max())]
        fig_forecast.add_trace(go.Scatter(
            x=trend_x, y=model.predict(np.array(trend_x).reshape(-1, 1)),
            mode="lines", name="Fitted Trend", line=dict(color="gray", dash="dot"),
        ))
        forecast_years_axis = [base_year_f] + [base_year_f + n for n in range(1, int(horizon) + 1)]
        forecast_fd_axis = [baseline_fd] + forecast_df["FD"].tolist()
        fig_forecast.add_trace(go.Scatter(
            x=forecast_years_axis, y=forecast_fd_axis,
            mode="lines+markers", name=f"Forecast (+1 to +{int(horizon)}y)",
            line=dict(color="#d62728", dash="dash"), marker=dict(size=9, symbol="diamond"),
        ))
        fig_forecast.update_layout(
            title="Fractal Dimension Trend & Forecast",
            xaxis_title="Year", yaxis_title="Fractal Dimension (FD)", height=460,
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # ---- Chart 2: year-over-year erosion/accretion change ----
        bar_colors_fc = ["#2ca02c" if v >= 0 else "#d62728" for v in forecast_df["Distance_Change_m"]]
        fig_fc_bar = go.Figure(go.Bar(
            x=[f"+{n}ปี ({int(y)})" for n, y in zip(forecast_df["Years_Ahead"], forecast_df["Year"])],
            y=forecast_df["Distance_Change_m"],
            marker_color=bar_colors_fc,
            text=[f"{v:+.2f} m" for v in forecast_df["Distance_Change_m"]],
            textposition="auto",
        ))
        fig_fc_bar.update_layout(
            title="ค่าการกัดเซาะ/งอกที่คาดการณ์ — เปลี่ยนแปลงต่อปี (Year-over-Year)",
            xaxis_title="ปีที่พยากรณ์ล่วงหน้า", yaxis_title="เปลี่ยนแปลงจากปีก่อนหน้า (ม.)", height=420,
        )
        st.plotly_chart(fig_fc_bar, use_container_width=True)

        # ---- Chart 3: cumulative shift from the base year ----
        fig_fc_cum = go.Figure(go.Scatter(
            x=forecast_df["Year"], y=forecast_df["Cumulative_Distance_m"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#9467bd"), marker=dict(size=9),
        ))
        fig_fc_cum.update_layout(
            title=f"ระยะกัดเซาะ/งอกสะสม นับจากปี {int(base_year)} (Cumulative Change from Base Year)",
            xaxis_title="ปี", yaxis_title="ระยะสะสม (ม.)", height=420,
        )
        st.plotly_chart(fig_fc_cum, use_container_width=True)

    # --------------------------------------------------------------------
    # 7. DOWNLOADABLE CSV REPORT
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 7. Download Full Report")

    export_df = summary_df[["Date", "FD", "Distance_Change_m", "Status"]].copy()
    export_df["Cumulative_Distance_m"] = np.nan
    export_df["Type"] = "Historical"

    if not forecast_df.empty:
        forecast_export = forecast_df[["Date", "FD", "Distance_Change_m", "Cumulative_Distance_m", "Status"]].copy()
        forecast_export["Type"] = "Forecast"
        export_df = pd.concat([export_df, forecast_export], ignore_index=True)

    export_df = export_df.rename(columns={"FD": "FD_Value", "Distance_Change_m": "Erosion_Accretion_Distance_m"})
    export_df = export_df[
        ["Date", "FD_Value", "Erosion_Accretion_Distance_m", "Cumulative_Distance_m", "Status", "Type"]
    ]

    csv_data = export_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_data,
        file_name="coastal_erosion_accretion_report.csv",
        mime="text/csv",
    )

    st.dataframe(export_df, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Built with Streamlit, OpenCV, NumPy, scikit-learn & Plotly. "
        "Estimates are heuristic and intended for exploratory analysis only."
    )
