"""
================================================================================
 Coastal Erosion / Accretion Analyzer v4.0
 Box-Counting Fractal Dimension & Direct Spatial Trend Engine
================================================================================

A Streamlit application for analyzing and forecasting coastal erosion or
accretion trends from multi-temporal coastal imagery using the Box-Counting
Fractal Dimension (FD) method combined with Direct Linear Regression forecasting.

HOW TO RUN:
1. Install dependencies:
   pip install streamlit opencv-python numpy pandas scikit-learn plotly pillow

2. Run the app:
   streamlit run app.py
================================================================================
"""

import warnings
from datetime import date, datetime

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
# CONSTANTS
# ==============================================================================
MAX_DIM_DEFAULT = 1024          # Default max processing dimension (pixels)
STABILITY_THRESHOLD_M = 0.01    # Below this |distance| (m), classify as "Stable"
MIN_IMAGES_FOR_TRENDS = 2       # Minimum images required for change/forecast analysis
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
BORDER_CLIP_PX = 2              # Border pixels to zero-out to remove frame artifacts


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_date_input(text: str):
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


def resize_for_processing(img_bgr: np.ndarray, original_scale: float, max_dim: int):
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


def extract_binary_mask(img_bgr: np.ndarray, blur_kernel: int, land_is_bright: bool):
    """Convert to grayscale, apply Gaussian blur, and binarize using Otsu thresholding."""
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
    """Clean binary mask, remove border artifacts, and trace clean coastline contours."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (denoise_kernel, denoise_kernel))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Remove outer image frame artifacts by zeroing out the border pixels
    if BORDER_CLIP_PX > 0:
        cleaned[:BORDER_CLIP_PX, :] = 0
        cleaned[-BORDER_CLIP_PX:, :] = 0
        cleaned[:, :BORDER_CLIP_PX] = 0
        cleaned[:, -BORDER_CLIP_PX:] = 0

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

    if method == "ตรวจจับขอบแบบ Canny":
        filled_mask = np.zeros_like(cleaned)
        cv2.drawContours(filled_mask, contours_used, -1, 255, thickness=-1)
        edge_img = cv2.Canny(filled_mask, canny_low, canny_high)
    else:  # Contour Method
        edge_img = np.zeros_like(cleaned)
        cv2.drawContours(edge_img, contours_used, -1, 255, thickness=1)

    return {
        "cleaned_mask": cleaned,
        "edge_img": edge_img,
        "land_area_px": land_area_px,
        "coastline_length_px": coastline_length_px,
        "contours_used": contours_used,
    }


def create_coastline_overlay(img_bgr: np.ndarray, contours, color=(0, 255, 0), thickness=2):
    """Draw extracted coastline contours directly onto original BGR image for visual verification."""
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def box_counting_fd(binary_edge_img: np.ndarray):
    """Compute the Box-Counting Fractal Dimension of a binary boundary image."""
    bool_img = binary_edge_img > 0
    if not np.any(bool_img):
        return None

    h, w = bool_img.shape
    longest_side = max(h, w)
    p = int(np.ceil(np.log2(max(longest_side, 2))))
    padded_size = 2 ** p

    padded = np.zeros((padded_size, padded_size), dtype=bool)
    padded[:h, :w] = bool_img

    box_sizes = [2 ** k for k in range(1, p)]
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
    """Classify physical distance change into กัดเซาะ (Erosion) / งอก (Accretion) / คงที่ (Stable)."""
    if abs(distance_m) < STABILITY_THRESHOLD_M:
        return "คงที่"
    return "งอก" if distance_m > 0 else "กัดเซาะ"


# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================
st.set_page_config(
    page_title="เครื่องมือวิเคราะห์การกัดเซาะชายฝั่ง — Fractal Dimension v4",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 เครื่องมือวิเคราะห์การกัดเซาะ/การงอกของชายฝั่ง (v4.0)")
st.caption("วิธี Box-Counting Fractal Dimension & Direct Spatial Trend — วิเคราะห์และพยากรณ์แนวชายฝั่งจากภาพหลายช่วงเวลา")

with st.expander("ℹ️ เกี่ยวกับเครื่องมือนี้ / หลักการทำงาน", expanded=False):
    st.markdown(
        """
เครื่องมือนี้วิเคราะห์ **ชุดภาพชายฝั่งตามช่วงเวลา** (เช่น ภาพ NDWI จากดาวเทียม Landsat/Sentinel-2) เพื่อ:

1. **สกัดแนวชายฝั่งแบบปราศจาก Noise ขอบภาพ:** ด้วย Otsu Thresholding, Denoise Morphological Filtering และขจัด กรอบภาพ (Frame Artifacts)
2. **วาดเส้นชายฝั่งซ้อนทับภาพจริง (Overlay):** ตรวจสอบความถูกต้องของการสกัดแนวชายฝั่งด้วยตาเปล่าได้ทันที
3. **คำนวณ Fractal Dimension (FD):** วัดความซับซ้อนเรขาคณิตของแนวชายฝั่งด้วย Box-Counting Method
4. **พยากรณ์การเปลี่ยนแปลงทางกายภาพโดยตรง (Direct Spatial Forecasting):** พยากรณ์ระยะถอยร่น/รุกล้ำด้วย Linear Regression บนระยะทางสะสม และแยกวิเคราะห์แนวโน้ม FD อย่างสมเหตุสมผลทางวิทยาศาสตร์

*หมายเหตุ: ผลลัพธ์เป็นการประมาณค่าเชิงสำรวจ/การศึกษา ควรได้รับการตรวจสอบกับข้อมูลสำรวจภาคสนามก่อนนำไปใช้งานจริง*
        """
    )

# ==============================================================================
# SIDEBAR — PROCESSING PARAMETERS
# ==============================================================================
st.sidebar.header("⚙️ พารามิเตอร์การประมวลผล")

blur_kernel = st.sidebar.slider(
    "ขนาด Kernel สำหรับ Gaussian Blur (เลขคี่)", min_value=3, max_value=15, value=5, step=2,
    help="ค่ายิ่งสูง ยิ่งลดจุดรบกวนก่อนทำ Threshold แต่จะสูญเสียรายละเอียดปลีกย่อยมากขึ้น"
)

land_choice = st.sidebar.radio(
    "พิกเซลสีไหนแทน 'แผ่นดิน' หลังทำ Threshold?",
    ["สว่าง (255) = แผ่นดิน", "มืด (0) = แผ่นดิน"],
    index=0,
    help="หากผลการจำแนกแผ่นดินกับน้ำสลับกัน ให้สลับตัวเลือกนี้"
)
land_is_bright = (land_choice == "สว่าง (255) = แผ่นดิน")

denoise_kernel = st.sidebar.slider(
    "ระดับลด Noise / เกาะเล็ก (Denoise Kernel Size, odd)", min_value=3, max_value=25, value=9, step=2,
    help="ค่ายิ่งสูง ยิ่งลบจุดรบกวนเล็กๆ และรูเล็กๆ ออกจากมาสก์มากขึ้น"
)

keep_largest_only = st.sidebar.checkbox(
    "ใช้เฉพาะแนวชายฝั่งหลัก (ตัด noise/เกาะเล็กทิ้ง)", value=True,
    help="แนะนำให้เปิดไว้ — คำนวณเฉพาะแนวชายฝั่งที่ใหญ่ที่สุด ป้องกันค่าระยะทางเพี้ยน"
)

edge_method = st.sidebar.radio(
    "วิธีสกัดแนวขอบชายฝั่ง",
    ["ขอบเขตจาก Contour", "ตรวจจับขอบแบบ Canny"],
    index=0,
)

if edge_method == "ตรวจจับขอบแบบ Canny":
    canny_low = st.sidebar.slider("ค่าขีดเริ่ม Canny ต่ำ (Low Threshold)", 0, 255, 50)
    canny_high = st.sidebar.slider("ค่าขีดเริ่ม Canny สูง (High Threshold)", 0, 255, 150)
else:
    canny_low, canny_high = 50, 150

max_dim = st.sidebar.slider(
    "ขนาดภาพสูงสุดสำหรับประมวลผล (พิกเซล)", min_value=256, max_value=4096, value=MAX_DIM_DEFAULT, step=128,
    help="ย่อขนาดภาพขนาดใหญ่เพื่อความรวดเร็ว สเกลความละเอียดจะถูกปรับตามสัดส่วนอัตโนมัติ"
)

current_params = {
    "blur": blur_kernel, "land": land_is_bright, "denoise": denoise_kernel,
    "keep_largest": keep_largest_only, "method": edge_method,
    "canny_low": canny_low, "canny_high": canny_high, "max_dim": max_dim
}

if "last_params" in st.session_state and st.session_state["last_params"] != current_params:
    st.session_state["params_changed"] = True

# ==============================================================================
# FILE UPLOAD & PER-IMAGE METADATA INPUT
# ==============================================================================
st.subheader("📤 1. อัปโหลดภาพชายฝั่ง")

uploaded_files = st.file_uploader(
    "อัปโหลดภาพชายฝั่งหลายช่วงเวลา (NDWI / Landsat / Sentinel-2 หรือภาพแผ่นดิน-น้ำ)",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

image_configs = []

if uploaded_files:
    st.markdown("#### 📝 กำหนดวันที่และค่า Scale ของแต่ละภาพ")
    n_files = len(uploaded_files)
    current_year = datetime.now().year

    for i, up_file in enumerate(uploaded_files):
        with st.expander(f"🖼️ ภาพที่ {i + 1}: {up_file.name}", expanded=(n_files <= 3)):
            col1, col2 = st.columns(2)
            with col1:
                default_year = current_year - (n_files - 1 - i)
                date_str = st.text_input(
                    "วันที่ / ปี (YYYY-MM-DD หรือ YYYY)",
                    value=str(default_year),
                    key=f"date_input_{i}_{up_file.name}",
                )
            with col2:
                scale_val = st.number_input(
                    "ค่า Scale ความละเอียดเชิงพื้นที่ (เมตร/พิกเซล)",
                    min_value=0.0001, value=1.0, step=0.1, format="%.4f",
                    key=f"scale_input_{i}_{up_file.name}",
                )
            image_configs.append({"file": up_file, "date_str": date_str, "scale": scale_val})
else:
    st.info("👆 อัปโหลดภาพชายฝั่งอย่างน้อย 2 ภาพเพื่อเริ่มต้น")

# ==============================================================================
# ANALYSIS PIPELINE
# ==============================================================================
def run_pipeline(configs, blur_kernel, land_is_bright, edge_method, canny_low, canny_high, max_dim,
                  denoise_kernel, keep_largest_only):
    results = []
    errors = []

    for cfg in configs:
        up_file = cfg["file"]
        d = parse_date_input(cfg["date_str"])
        if d is None:
            errors.append(f"⚠️ ไม่สามารถแปลงวันที่ '{cfg['date_str']}' ของ '{up_file.name}' ได้ — ข้ามภาพนี้")
            continue

        try:
            img_bgr = load_image_bgr(up_file)
        except Exception as exc:
            errors.append(f"⚠️ อ่านไฟล์ภาพ '{up_file.name}' ไม่สำเร็จ ({exc}) — ข้ามภาพนี้")
            continue

        resized_img, eff_scale, was_resized, orig_shape, new_shape = resize_for_processing(
            img_bgr, cfg["scale"], max_dim
        )

        gray, blurred, binary_mask, otsu_val = extract_binary_mask(resized_img, blur_kernel, land_is_bright)

        coast_result = extract_clean_coastline(
            binary_mask, denoise_kernel, keep_largest_only, edge_method, canny_low, canny_high
        )
        if coast_result is None:
            errors.append(f"⚠️ ตรวจไม่พบแนวชายฝั่งในภาพ '{up_file.name}' — ข้ามภาพนี้")
            continue

        cleaned_mask = coast_result["cleaned_mask"]
        edge_img = coast_result["edge_img"]
        land_area_px = coast_result["land_area_px"]
        coastline_length_px = coast_result["coastline_length_px"]
        contours_used = coast_result["contours_used"]

        if coastline_length_px <= 0 or int(np.count_nonzero(edge_img)) == 0:
            errors.append(f"⚠️ แนวชายฝั่งในภาพ '{up_file.name}' หายไปหลังลด Noise — ข้ามภาพนี้")
            continue

        fd_result = box_counting_fd(edge_img)
        if fd_result is None:
            errors.append(f"⚠️ คำนวณ Box-Counting ภาพ '{up_file.name}' ไม่สำเร็จ — ข้ามภาพนี้")
            continue

        overlay_img = create_coastline_overlay(resized_img, contours_used)

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
            "overlay_img": overlay_img,
            "gray_img": gray,
            "binary_mask": binary_mask,
            "cleaned_mask": cleaned_mask,
            "edge_img": edge_img,
            "contours_used": contours_used,
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

    results.sort(key=lambda r: r["decimal_year"])
    return results, errors


analyze_clicked = st.button(
    "🔍 วิเคราะห์แนวชายฝั่งและคำนวณ Fractal Dimension",
    type="primary",
    disabled=(len(image_configs) == 0),
)

if analyze_clicked:
    try:
        with st.spinner("กำลังประมวลผลภาพ — สกัดแนวชายฝั่ง สร้าง Overlay และคำนวณ Fractal Dimension..."):
            results, errors = run_pipeline(
                image_configs, blur_kernel, land_is_bright, edge_method, canny_low, canny_high, max_dim,
                denoise_kernel, keep_largest_only,
            )
        st.session_state["results"] = results
        st.session_state["errors"] = errors
        st.session_state["processed"] = True
        st.session_state["last_params"] = current_params
        st.session_state["params_changed"] = False
    except Exception as exc:
        st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {exc}")
        st.session_state["processed"] = False


# ==============================================================================
# RESULTS DISPLAY
# ==============================================================================
if st.session_state.get("processed"):
    if st.session_state.get("params_changed"):
        st.warning("⚠️ มีการปรับเปลี่ยนพารามิเตอร์ใน Sidebar — กรุณากดปุ่ม '🔍 วิเคราะห์...' อีกครั้งเพื่ออัปเดตผลลัพธ์")

    results = st.session_state.get("results", [])
    errors = st.session_state.get("errors", [])

    for err in errors:
        st.warning(err)

    if len(results) == 0:
        st.error("ไม่มีภาพใดประมวลผลสำเร็จ กรุณาตรวจสอบไฟล์หรือพารามิเตอร์แล้วลองใหม่")
        st.stop()

    # --------------------------------------------------------------------
    # 2. IMAGE PREVIEWS WITH COASTLINE OVERLAY
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🖼️ 2. ผลการสกัดแนวชายฝั่งและ Overlay")

    for r in results:
        resize_note = f", ย่อขนาดจาก {r['original_shape'][1]}x{r['original_shape'][0]}" if r["was_resized"] else ""
        st.markdown(f"**{r['file_name']}** — {r['date'].isoformat()} (สเกล: {r['scale_m_per_px']:.4f} ม./px{resize_note})")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(r["original_img_bgr"], channels="BGR", caption="ภาพต้นฉบับ", use_container_width=True)
        with c2:
            st.image(r["cleaned_mask"], caption=f"มาสก์แผ่นดิน (Otsu={r['otsu_val']:.1f})", use_container_width=True)
        with c3:
            st.image(r["edge_img"], caption="แนวขอบชายฝั่ง", use_container_width=True)
        with c4:
            st.image(r["overlay_img"], channels="BGR", caption="🟩 แนวชายฝั่งซ้อนทับภาพจริง", use_container_width=True)
            
        st.caption(
            f"พื้นที่แผ่นดิน: {r['land_area_px']:,.0f} px  •  "
            f"ความยาวชายฝั่ง: {r['coastline_length_px']:,.0f} px  •  "
            f"Fractal Dimension (FD): **{r['fd']:.4f}** (R² = {r['fd_r_squared']:.4f})"
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # BUILD SUMMARY DATAFRAME + CUMULATIVE SPATIAL DISTANCE
    # --------------------------------------------------------------------
    rows = []
    cum_shift = 0.0
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
            row["Distance_Change_m"] = 0.0
            row["Cumulative_Shift_m"] = 0.0
            row["Status"] = "ข้อมูลฐาน (เริ่มต้น)"
        else:
            prev = results[idx - 1]
            area_diff_px = r["land_area_px"] - prev["land_area_px"]
            avg_coastline_len = float(np.mean([r["coastline_length_px"], prev["coastline_length_px"]]))
            avg_scale = float(np.mean([r["scale_m_per_px"], prev["scale_m_per_px"]]))
            shift_px = area_diff_px / avg_coastline_len if avg_coastline_len > 0 else 0.0
            shift_m = shift_px * avg_scale
            cum_shift += shift_m
            
            row["Distance_Change_m"] = shift_m
            row["Cumulative_Shift_m"] = cum_shift
            row["Status"] = classify_change(shift_m)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # --------------------------------------------------------------------
    # 3. MULTI-TEMPORAL SHORELINE MAP
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🗺️ 3. แผนที่เปรียบเทียบการเปลี่ยนแปลงแนวชายฝั่งรายปี (Multi-Temporal Map)")
    
    fig_multi = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, r in enumerate(results):
        color = colors[idx % len(colors)]
        for c_idx, contour in enumerate(r["contours_used"]):
            pts = contour.reshape(-1, 2)
            fig_multi.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode="lines",
                name=f"{r['date'].isoformat()}" if c_idx == 0 else None,
                showlegend=(c_idx == 0),
                line=dict(color=color, width=2),
                hoverinfo="text",
                text=f"ปี: {r['date'].isoformat()} | FD: {r['fd']:.4f}"
            ))

    fig_multi.update_layout(
        title="เปรียบเทียบเส้นแนวชายฝั่งทุกช่วงเวลา (พิกเซลพิกัดภาพ)",
        xaxis_title="พิกัดพิกเซล X",
        yaxis_title="พิกัดพิกเซล Y",
        yaxis=dict(autorange="reversed"),  # Match image orientation (top-left origin)
        height=550,
    )
    st.plotly_chart(fig_multi, use_container_width=True)

    # --------------------------------------------------------------------
    # 4. STRUCTURED SUMMARY TABLE
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 4. ตารางสรุปผลข้อมูล")
    display_summary = summary_df.rename(columns={
        "Date": "วันที่",
        "Decimal_Year": "ปี (ทศนิยม)",
        "Land_Area_px": "พื้นที่แผ่นดิน (px)",
        "Coastline_Length_px": "ความยาวชายฝั่ง (px)",
        "Scale_m_per_px": "สเกล (ม./px)",
        "FD": "FD",
        "Distance_Change_m": "เปลี่ยนแปลงจากช่วงก่อน (ม.)",
        "Cumulative_Shift_m": "ระยะสะสมจากจุดเริ่ม (ม.)",
        "Status": "สถานะ",
    })
    st.dataframe(
        display_summary.style.format(
            {
                "ปี (ทศนิยม)": "{:.3f}",
                "สเกล (ม./px)": "{:.4f}",
                "FD": "{:.4f}",
                "เปลี่ยนแปลงจากช่วงก่อน (ม.)": "{:+.3f}",
                "ระยะสะสมจากจุดเริ่ม (ม.)": "{:+.3f}",
            }
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------------------
    # 5. BOX-COUNTING DIAGNOSTIC
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📐 5. การตรวจสอบ Box-Counting (Log-Log Regression)")
    file_labels = [f"{r['date'].isoformat()} — {r['file_name']}" for r in results]
    selected_label = st.selectbox("เลือกภาพเพื่อตรวจสอบผลการคำนวณ Box-Counting:", file_labels)
    selected_idx = file_labels.index(selected_label)
    sel = results[selected_idx]

    fitted_y = sel["fd"] * sel["log_inv_r"] + sel["intercept"]

    fig_diag = go.Figure()
    fig_diag.add_trace(go.Scatter(
        x=sel["log_inv_r"], y=sel["log_n"], mode="markers", name="ข้อมูล log N(r)",
        marker=dict(size=10, color="#1f77b4"),
    ))
    fig_diag.add_trace(go.Scatter(
        x=sel["log_inv_r"], y=fitted_y, mode="lines",
        name=f"เส้น Regression (FD = {sel['fd']:.4f})",
        line=dict(color="#d62728", dash="dash"),
    ))
    fig_diag.update_layout(
        title=f"Log-Log Plot ของ Box-Counting — {sel['file_name']} (R² = {sel['fd_r_squared']:.4f})",
        xaxis_title="log(1/r)", yaxis_title="log(N(r))", height=420,
    )
    st.plotly_chart(fig_diag, use_container_width=True)

    # --------------------------------------------------------------------
    # 6. DIRECT SPATIAL & FD FORECASTING
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 6. การพยากรณ์แนวชายฝั่งล่วงหน้า (Direct Spatial Forecasting)")

    if len(results) < MIN_IMAGES_FOR_TRENDS:
        st.info(f"ต้องใช้อย่างน้อย {MIN_IMAGES_FOR_TRENDS} ภาพเพื่อทำการพยากรณ์")
        forecast_df = pd.DataFrame()
    else:
        years_arr = np.array([r["decimal_year"] for r in results]).reshape(-1, 1)
        fd_arr = np.array([r["fd"] for r in results])
        cum_shifts = summary_df["Cumulative_Shift_m"].values

        # 1. FD Trend Model
        fd_model = LinearRegression()
        fd_model.fit(years_arr, fd_arr)
        r2_fd = fd_model.score(years_arr, fd_arr)

        # 2. Direct Distance Trend Model (Predicting Cumulative Shift over Time)
        dist_model = LinearRegression()
        dist_model.fit(years_arr, cum_shifts)
        r2_dist = dist_model.score(years_arr, cum_shifts)

        last_year_detected = float(years_arr.max())

        col_base, col_horizon = st.columns(2)
        with col_base:
            base_year = st.number_input(
                "ปีฐานสำหรับเริ่มพยากรณ์",
                min_value=1900, max_value=2200,
                value=int(round(last_year_detected)), step=1,
            )
        with col_horizon:
            horizon = st.number_input(
                "พยากรณ์ล่วงหน้ากี่ปี", min_value=1, max_value=30, value=5, step=1,
            )

        base_year_f = float(base_year)
        baseline_cum_shift = float(dist_model.predict([[base_year_f]])[0])

        forecast_rows = []
        prev_cum = baseline_cum_shift

        for n in range(1, int(horizon) + 1):
            future_y = base_year_f + n
            pred_fd = float(fd_model.predict([[future_y]])[0])
            pred_cum_shift = float(dist_model.predict([[future_y]])[0])
            inc_shift = pred_cum_shift - prev_cum
            prev_cum = pred_cum_shift

            forecast_rows.append({
                "Date": f"พยากรณ์ +{n}ปี ({int(base_year + n)})",
                "Years_Ahead": n,
                "Year": base_year + n,
                "FD": pred_fd,
                "Cumulative_Distance_m": pred_cum_shift,
                "Distance_Change_m": inc_shift,
                "Status": classify_change(inc_shift),
            })

        forecast_df = pd.DataFrame(forecast_rows)

        first_row = forecast_df.iloc[0]
        last_row = forecast_df.iloc[-1]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                f"ปีถัดไป (+1 ปี, {int(first_row['Year'])})",
                f"FD: {first_row['FD']:.4f}",
                delta=f"{first_row['Distance_Change_m']:+.3f} ม. ({first_row['Status']})",
            )
        with col_b:
            st.metric(
                f"อีก {int(horizon)} ปี ({int(last_row['Year'])})",
                f"FD: {last_row['FD']:.4f}",
                delta=f"สะสม {last_row['Cumulative_Distance_m']:+.3f} ม. ({last_row['Status']})",
            )

        st.caption(
            f"📊 แบบจำลองระยะทางทางกายภาพ: Distance = {dist_model.coef_[0]:.4f} × Year + {dist_model.intercept_:.2f} (R² = {r2_dist:.4f})  |  "
            f"Complexity Model: FD = {fd_model.coef_[0]:.6f} × Year + {fd_model.intercept_:.4f} (R² = {r2_fd:.4f})"
        )

        st.markdown("##### ตารางพยากรณ์รายปี (Year-by-Year Forecast)")
        display_forecast = forecast_df[
            ["Years_Ahead", "Year", "FD", "Cumulative_Distance_m", "Distance_Change_m", "Status"]
        ].rename(columns={
            "Years_Ahead": "+ปี",
            "Year": "ปี ค.ศ.",
            "FD": "FD ที่คาดการณ์",
            "Cumulative_Distance_m": "ระยะสะสมจากปีฐาน (ม.)",
            "Distance_Change_m": "เปลี่ยนแปลงต่อปี (ม.)",
            "Status": "สถานะ",
        })
        st.dataframe(
            display_forecast.style.format({
                "FD ที่คาดการณ์": "{:.4f}",
                "ระยะสะสมจากปีฐาน (ม.)": "{:+.3f}",
                "เปลี่ยนแปลงต่อปี (ม.)": "{:+.3f}",
            }),
            use_container_width=True,
        )

        # Chart: Direct Physical Shoreline Shift Prediction
        fig_shift = go.Figure()
        fig_shift.add_trace(go.Scatter(
            x=[r["decimal_year"] for r in results], y=cum_shifts,
            mode="markers+lines", name="ระยะจริงย้อนหลัง",
            line=dict(color="#1f77b4"), marker=dict(size=8),
        ))
        
        future_years = [base_year_f + n for n in range(0, int(horizon) + 1)]
        future_shifts = [baseline_cum_shift] + forecast_df["Cumulative_Distance_m"].tolist()
        
        fig_shift.add_trace(go.Scatter(
            x=future_years, y=future_shifts,
            mode="lines+markers", name=f"พยากรณ์ระยะสะสม (+{int(horizon)} ปี)",
            line=dict(color="#d62728", dash="dash"), marker=dict(size=8, symbol="diamond"),
        ))
        fig_shift.update_layout(
            title="แนวโน้มและการพยากรณ์การขยับตัวทางกายภาพของแนวชายฝั่ง (Spatial Shift Forecast)",
            xaxis_title="ปี", yaxis_title="ระยะสะสม (เมตร)", height=450,
        )
        st.plotly_chart(fig_shift, use_container_width=True)

    # --------------------------------------------------------------------
    # 7. DOWNLOAD REPORT
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 7. ดาวน์โหลดรายงาน")

    export_df = summary_df[["Date", "FD", "Distance_Change_m", "Cumulative_Shift_m", "Status"]].copy()
    export_df["Type"] = "ข้อมูลย้อนหลัง"

    if not forecast_df.empty:
        forecast_export = forecast_df[["Date", "FD", "Distance_Change_m", "Cumulative_Distance_m", "Status"]].copy()
        forecast_export["Type"] = "พยากรณ์"
        forecast_export = forecast_export.rename(columns={"Cumulative_Distance_m": "Cumulative_Shift_m"})
        export_df = pd.concat([export_df, forecast_export], ignore_index=True)

    export_df = export_df.rename(columns={
        "Date": "วันที่",
        "FD": "ค่า_FD",
        "Distance_Change_m": "เปลี่ยนแปลง_เมตร",
        "Cumulative_Shift_m": "ระยะสะสม_เมตร",
        "Status": "สถานะ",
        "Type": "ประเภทข้อมูล",
    })

    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ ดาวน์โหลดรายงาน CSV",
        data=csv_data,
        file_name="coastal_erosion_report_v4.csv",
        mime="text/csv",
    )