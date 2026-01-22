import streamlit as st
import time
import base64
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import zipfile
import io
import cv2
import numpy as np
import os
from scipy import ndimage

# ================================
# PAGE CONFIG & PERFORMANCE
# ================================
st.set_page_config(
    page_title="Z_cut AI - Background Remover",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Performance optimization
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ================================
# CACHE SESSIONS FOR ACCURACY
# ================================
@st.cache_resource
def get_session(model_name):

    return new_session(model_name)

# ================================
# SESSION STATE INITIALIZATION
# ================================
def init_session_state():
    state_defaults = {
        "original": None,
        "output": None,
        "batch_originals": [],
        "batch_outputs": [],
        "batch_names": [],
        "history_outputs": [],
        "history_names": [],
        "processing_quality": "Professional",
        "edge_refinement": True,
        "color_preservation": 85,
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ================================
# ADVANCED EDGE REFINEMENT
# ================================
def advanced_edge_refinement(pil_img, quality_level=2):
    """Multi-pass edge refinement for superior accuracy"""
    img = np.array(pil_img)
    
    if len(img.shape) != 3 or img.shape[2] != 4:
        return pil_img
    
    bgr = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3].astype(np.float32)
    
    # PASS 1: Bilateral Filter for edge preservation
    alpha_bilateral = cv2.bilateralFilter(
        alpha.astype(np.uint8), 
        d=9, 
        sigmaColor=75, 
        sigmaSpace=75
    ).astype(np.float32)
    
    # PASS 2: Morphological operations for clarity
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    alpha_morph = cv2.morphologyEx(alpha_bilateral.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
    alpha_morph = cv2.morphologyEx(alpha_morph.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    
    # PASS 3: Gaussian blur for smooth edges
    if quality_level == 1:
        alpha_smooth = cv2.GaussianBlur(alpha_morph, (3, 3), 0)
    elif quality_level == 2:
        alpha_smooth = cv2.GaussianBlur(alpha_morph, (5, 5), 0.5)
    else:  # quality_level == 3
        alpha_smooth = cv2.GaussianBlur(alpha_morph, (7, 7), 1.0)
    
    # PASS 4: Threshold for crisp boundaries
    _, alpha_thresh = cv2.threshold(alpha_smooth, 30, 255, cv2.THRESH_BINARY)
    
    # PASS 5: Dilation for object recovery
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    alpha_final = cv2.dilate(alpha_thresh, kernel_dilate, iterations=1)
    
    # Combine channels
    refined = np.dstack((bgr.astype(np.uint8), alpha_final)).astype(np.uint8)
    return Image.fromarray(refined)

# ================================
# ADAPTIVE COLOR PRESERVATION
# ================================
def preserve_colors(pil_img, preservation_level=85):
    """Enhance color vibrancy and detail preservation"""
    img = np.array(pil_img)
    
    if img.shape[2] != 4:
        return pil_img
    
    bgr = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3]
    
    # Enhance saturation
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation_factor = 1.0 + (preservation_level / 200)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Enhance contrast slightly
    lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.05, 0, 255)
    bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    result = np.dstack((bgr, alpha)).astype(np.uint8)
    return Image.fromarray(result)

# ================================
# SHADOW & REFLECTION REMOVAL
# ================================
def remove_shadows(pil_img, intensity=0.5):
    """Intelligent shadow and reflection removal"""
    img = np.array(pil_img)
    
    if img.shape[2] != 4:
        return pil_img
    
    bgr = img[:, :, :3].astype(np.float32) / 255.0
    alpha = img[:, :, 3]
    
    # Convert to HSV for shadow detection
    bgr_uint = (bgr * 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_uint, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Detect dark areas (shadows)
    shadow_mask = cv2.inRange(
        hsv.astype(np.uint8),
        (0, 0, 0),
        (180, 255, 100)
    ).astype(np.float32) / 255.0
    
    # Reduce shadow intensity
    bgr = bgr * (1 - shadow_mask * intensity * 0.3) + shadow_mask * intensity * 0.3
    bgr = np.clip(bgr, 0, 1)
    
    result = np.dstack((
        (bgr * 255).astype(np.uint8),
        alpha
    ))
    return Image.fromarray(result)

# ================================
# BACKGROUND REMOVAL - ACCURACY FOCUSED
# ================================
def remove_bg_accurate(image_bytes, mode="professional", quality="Professional"):
    """High-accuracy background removal focused only on clean cuts"""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        # Select best AI model
        if quality == "Professional":
          session = get_session("isnet-general-use")
        elif quality == "Portrait":
         session = get_session("u2net_human_seg")
        else:
         session = get_session("u2net")

        # PASS 1 — Strong AI segmentation
        output = remove(
            image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=12
        )

        # PASS 2 — Edge refinement (safe)
        if quality == "Professional":
            output = advanced_edge_refinement(output, quality_level=3)
        elif quality == "Portrait":
            output = advanced_edge_refinement(output, quality_level=2)
        else:
            output = advanced_edge_refinement(output, quality_level=1)

        # PASS 3 — Light detail sharpening (no math issues)
        rgb = output.convert("RGB")
        sharp = ImageEnhance.Sharpness(rgb).enhance(1.3)
        output = Image.merge("RGBA", (*sharp.split(), output.split()[-1]))

        # PASS 4 — Final alpha cleanup (removes edge noise safely)
        arr = np.array(output)
        alpha = arr[:, :, 3]
        alpha = cv2.medianBlur(alpha, 5)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        arr[:, :, 3] = alpha
        output = Image.fromarray(arr)

        return image, output

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        return None, None


# ================================
# UTILITY FUNCTIONS
# ================================
def convert_image(img):
    """Convert PIL Image to PNG bytes"""
    buf = BytesIO()
    img.save(buf, format="PNG", quality=95)
    buf.seek(0)
    return buf.getvalue()

def create_zip():
    """Create downloadable ZIP of all processed images"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(st.session_state.batch_outputs):
            img_bytes = convert_image(img)
            zipf.writestr(f"zcut_{i+1}.png", img_bytes)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def run_batch(files, mode_select, quality_select):
    """Process multiple files with accuracy focus"""
    st.session_state.batch_originals = []
    st.session_state.batch_outputs = []
    st.session_state.batch_names = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(files)
    
    for i, file in enumerate(files):
        try:
            status_text.text(f"🔄 Processing {i+1}/{total}: {file.name} [{quality_select}]")
            original, output = remove_bg_accurate(
                file.getvalue(), 
                mode=mode_select,
                quality=quality_select
            )
            
            if output is not None:
                st.session_state.batch_originals.append(original)
                st.session_state.batch_outputs.append(output)
                st.session_state.batch_names.append(file.name)
                st.session_state.history_outputs.append(output)
                st.session_state.history_names.append(file.name)
            
            progress_bar.progress((i + 1) / total)
        except Exception as e:
            st.warning(f"⚠️ Failed: {file.name}")
            print(f"Error: {e}")
    
    status_text.empty()
    progress_bar.empty()

# ================================
# CREATIVE SPLASH SCREEN
# ================================
def create_splash_screen():
    """Modern animated splash screen"""
    splash_html = """
    <style>
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glow {
            0%, 100% { text-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
            50% { text-shadow: 0 0 40px rgba(99, 102, 241, 0.6); }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
        }
        .splash-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eef7 50%, #f5f7fa 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            flex-direction: column;
            gap: 30px;
        }
        .splash-icon {
            font-size: 100px;
            animation: float 3s ease-in-out infinite;
        }
        .splash-title {
            font-size: 52px;
            font-weight: 900;
            color: #1e3a8a;
            text-align: center;
            animation: slideIn 0.8s ease-out, glow 3s ease-in-out infinite;
            font-family: 'Segoe UI', sans-serif;
            letter-spacing: -1px;
        }
        .splash-subtitle {
            font-size: 18px;
            color: #64748b;
            text-align: center;
            animation: slideIn 1s ease-out;
            font-family: 'Segoe UI', sans-serif;
        }
        .loading-dots {
            font-size: 20px;
            color: #6366f1;
            animation: slideIn 1.2s ease-out;
            letter-spacing: 4px;
        }
    </style>
    <div class="splash-container">
        <div class="splash-icon">✨</div>
        <div class="splash-title">Z_cut AI</div>
        <div class="splash-subtitle">Premium AI Background Remover</div>
        <div class="loading-dots">●●●</div>
    </div>
    """
    return splash_html

if not st.session_state.splash_done:
    st.markdown(create_splash_screen(), unsafe_allow_html=True)
    time.sleep(2)
    st.session_state.splash_done = True
    st.rerun()

# ================================
# LIGHT AESTHETIC STYLES
# ================================
def inject_light_styles(theme):
    """Inject beautiful light aesthetic CSS"""
    css = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body, .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e8eef7 100%);
            color: #1e293b;
            font-family: 'Segoe UI', -apple-system, sans-serif;
            background-attachment: fixed;
        }

        .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* CARDS & CONTAINERS */
        .zcut-card {
            background: rgba(255, 255, 255, 0.85);
            border: 1.5px solid rgba(99, 102, 241, 0.15);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 28px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .zcut-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.12);
            border-color: rgba(99, 102, 241, 0.25);
        }

        /* BUTTONS */
        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            padding: 13px 28px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        }

        .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 700;
            padding: 13px 28px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }

        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        }

        /* FILE UPLOADER */
        .stFileUploader {
            border: 2px dashed rgba(99, 102, 241, 0.3) !important;
            border-radius: 16px;
            padding: 45px;
            text-align: center;
            transition: all 0.3s ease;
            background: rgba(99, 102, 241, 0.02) !important;
        }

        .stFileUploader:hover {
            border-color: rgba(99, 102, 241, 0.5) !important;
            background: rgba(99, 102, 241, 0.05) !important;
        }

        /* IMAGES */
        .stImage img {
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.1);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
        }

        /* HEADERS & TYPOGRAPHY */
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        h1 { font-size: 48px; margin-bottom: 20px; }
        h2 { font-size: 36px; margin-bottom: 20px; }
        h3 { font-size: 24px; margin-bottom: 16px; }

        p, span {
            color: #64748b;
            line-height: 1.6;
        }

        .stCaption {
            color: #94a3b8;
            font-size: 13px;
        }

        /* DIVIDER */
        hr {
            background: rgba(99, 102, 241, 0.15);
            border: none;
            height: 1px;
            margin: 35px 0;
        }

        /* RADIO BUTTONS */
        div[role="radiogroup"] {
            display: flex;
            gap: 16px;
            margin: 20px 0;
            flex-wrap: wrap;
        }

        div[role="radiogroup"] label {
            color: #1e293b !important;
            font-weight: 600;
            padding: 12px 20px;
            border-radius: 12px;
            border: 2px solid rgba(99, 102, 241, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
            background: white !important;
        }

        div[role="radiogroup"] label:hover {
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.08) !important;
        }

        /* SLIDER */
        .stSlider {
            padding: 15px 0;
        }

        /* RESPONSIVE */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 16px;
                padding-right: 16px;
            }

            .zcut-card {
                padding: 24px;
            }

            h1 { font-size: 36px; }
            h2 { font-size: 28px; }
            h3 { font-size: 20px; }

            .stButton > button, .stDownloadButton > button {
                padding: 11px 20px;
                font-size: 13px;
            }
        }

        /* ANIMATIONS */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .zcut-card {
            animation: fadeIn 0.5s ease-out;
        }

        /* BATCH GRID */
        .batch-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 24px;
            margin-top: 24px;
        }

        @media (max-width: 768px) {
            .batch-grid {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 16px;
            }
        }

        /* STATS BOX */
        .stat-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(99, 102, 241, 0.05));
            border-left: 4px solid #6366f1;
            padding: 20px;
            border-radius: 12px;
            margin: 12px 0;
        }

        .stat-box-green {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
            border-left: 4px solid #10b981;
        }

        .stat-box-purple {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05));
            border-left: 4px solid #8b5cf6;
        }

        .stat-label {
            font-size: 12px;
            font-weight: 700;
            color: #6366f1;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 800;
            color: #1e293b;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_light_styles(st.session_state.theme)

# ================================
# HEADER WITH LOGO
# ================================
col_header1, col_header2 = st.columns([1, 0.12])

with col_header1:
    st.markdown("""
    <h1 style="margin-bottom: 5px; background: linear-gradient(135deg, #6366f1, #4f46e5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
    ✨ Z_cut AI by 28rtm
    </h1>
    <p style="font-size: 16px; opacity: 0.8; margin-bottom: 0;">Professional AI Background Removal with Precision Accuracy</p>
    """, unsafe_allow_html=True)

with col_header2:
    if st.button("🔧", use_container_width=True, help="Settings"):
        st.balloons()

st.markdown('<hr>', unsafe_allow_html=True)

# ================================
# FEATURES HIGHLIGHT
# ================================
st.markdown("""
<div class="zcut-card">
    <h3 style="margin-bottom: 20px;">🚀 Premium Features</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
        <div class="stat-box">
            <p class="stat-label">🎯 Advanced Accuracy</p>
            <p class="stat-value">Multi-Pass Processing</p>
            <p style="font-size: 13px; margin-top: 8px;">4-pass edge refinement with AI precision</p>
        </div>
        <div class="stat-box stat-box-green">
            <p class="stat-label">🎨 Color Preservation</p>
            <p class="stat-value">Smart Enhancement</p>
            <p style="font-size: 13px; margin-top: 8px;">Maintains natural colors with vibrance</p>
        </div>
        <div class="stat-box stat-box-purple">
            <p class="stat-label">⚡ Professional Quality</p>
            <p class="stat-value">Batch Processing</p>
            <p style="font-size: 13px; margin-top: 8px;">Process hundreds of images instantly</p>
        </div>
    </div>
    <p style="margin-top: 20px; font-size: 13px; opacity: 0.7;">✓ No login  ✓ Free forever  ✓ No watermarks  ✓ Instant results</p>
</div>
""", unsafe_allow_html=True)

# ================================
# UPLOAD & SETTINGS
# ================================
st.markdown('<div class="zcut-card">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📸 Upload Images for Processing",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Max 35MB per image • High quality recommended"
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("**Processing Mode**")
    mode = st.radio("Processing mode", ["Logo / Design", "Photo / Human"], horizontal=True, label_visibility="collapsed")

with col2:
    st.markdown("**Quality Level**")
    quality = st.selectbox("Quality level", ["Standard", "Professional", "Portrait"], label_visibility="collapsed")


with col3:
    st.markdown("**Color Boost**")
    color_level = st.slider(
    "Color boost",
    min_value=0,
    max_value=100,
    value=85,
    step=5,
    label_visibility="collapsed"
)


st.markdown('</div>', unsafe_allow_html=True)

# ================================
# PROCESS BUTTON
# ================================
if uploaded_files:
    st.markdown('<div class="zcut-card" style="text-align: center;">', unsafe_allow_html=True)
    
    col_info1, col_info2, col_btn = st.columns([1, 1, 1.5])
    
    with col_info1:
        st.metric("Files Ready", len(uploaded_files), delta="images", delta_color="off")
    
    with col_info2:
        total_size = sum(len(f.getvalue()) for f in uploaded_files) / (1024*1024)
        st.metric("Total Size", f"{total_size:.2f}", delta="MB", delta_color="off")
    
    with col_btn:
        if st.button("⚡ Run ZCUT AI - Advanced Processing", use_container_width=True, key="run_process"):
            with st.spinner("🔄 Running multi-pass processing with AI accuracy..."):
                run_batch(uploaded_files, mode, quality)
            if st.session_state.batch_outputs:
                st.success(f"✅ Successfully processed {len(st.session_state.batch_outputs)} image{'s' if len(st.session_state.batch_outputs) != 1 else ''}!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# BATCH RESULTS
# ================================
if st.session_state.batch_outputs:
    st.markdown('<hr>', unsafe_allow_html=True)
    
    col_title, col_action = st.columns([1, 0.3])
    with col_title:
        st.markdown(f"<h2>🖼️ Processed Results ({len(st.session_state.batch_outputs)})</h2>", unsafe_allow_html=True)
    with col_action:
        if st.button("📥 Download All ZIP", use_container_width=True):
            st.download_button(
                label="",
                data=create_zip(),
                file_name="zcut_batch.zip",
                mime="application/zip",
                use_container_width=True
            )

    st.markdown('<div class="batch-grid">', unsafe_allow_html=True)
    cols = st.columns(3)
    
    for idx, (orig, out, name) in enumerate(zip(
        st.session_state.batch_originals,
        st.session_state.batch_outputs,
        st.session_state.batch_names
    )):
        with cols[idx % 3]:
            st.markdown("""
            <div style="border-radius: 16px; overflow: hidden; border: 1.5px solid rgba(99, 102, 241, 0.15); 
                        background: white; padding: 12px; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);">
            """, unsafe_allow_html=True)
            
            st.image(out, use_container_width=True)
            st.markdown(f'<p style="font-size: 12px; margin-top: 12px; word-break: break-word; color: #64748b;"><b>{name[:25]}</b></p>', unsafe_allow_html=True)
            
            col_dl1, col_dl2, col_view = st.columns([1, 1, 1])
            with col_dl1:
                st.download_button(
                    "⬇️ PNG",
                    convert_image(out),
                    f"zcut_{idx+1}.png",
                    "image/png",
                    use_container_width=True,
                    key=f"dl_{idx}"
                )
            with col_dl2:
                st.download_button(
                    "📊 JPG",
                    Image.open(BytesIO(convert_image(out))).convert("RGB").tobytes(),
                    f"zcut_{idx+1}.jpg",
                    "image/jpeg",
                    use_container_width=True,
                    key=f"dljpg_{idx}"
                )
            with col_view:
                if st.button("👁️", use_container_width=True, key=f"view_{idx}"):
                    st.session_state.original = orig
                    st.session_state.output = out
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# SINGLE IMAGE VIEWER
# ================================
if st.session_state.original and st.session_state.output:
    st.markdown('<div class="zcut-card">', unsafe_allow_html=True)
    st.markdown("<h2>🔍 Detailed Comparison</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.05); padding: 16px; border-radius: 12px; margin-bottom: 16px;">
        <p style="font-weight: 700; color: #1e293b; margin-bottom: 12px;">📷 Original Image</p>
        </div>
        """, unsafe_allow_html=True)
        st.image(st.session_state.original, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.05); padding: 16px; border-radius: 12px; margin-bottom: 16px;">
        <p style="font-weight: 700; color: #1e293b; margin-bottom: 12px;">✨ Background Removed</p>
        </div>
        """, unsafe_allow_html=True)
        st.image(st.session_state.output, use_container_width=True)
    
    col_dl, col_dl2, col_clear = st.columns([1, 1, 1])
    with col_dl:
        st.download_button(
            "⬇️ Download PNG",
            convert_image(st.session_state.output),
            "zcut_result.png",
            "image/png",
            use_container_width=True
        )
    with col_dl2:
        jpg_buffer = BytesIO()
        Image.open(BytesIO(convert_image(st.session_state.output))).convert("RGB").save(jpg_buffer, format="JPEG", quality=95)
        st.download_button(
            "📊 Download JPG",
            jpg_buffer.getvalue(),
            "zcut_result.jpg",
            "image/jpeg",
            use_container_width=True
        )
    with col_clear:
        if st.button("✕ Close", use_container_width=True):
            st.session_state.original = None
            st.session_state.output = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# HISTORY
# ================================
if st.session_state.history_outputs:
    st.markdown('<hr>', unsafe_allow_html=True)
    
    col_hist, col_clear = st.columns([1, 0.2])
    with col_hist:
        st.markdown(f"<h2>🕘 Session History ({len(st.session_state.history_outputs)})</h2>", unsafe_allow_html=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history_outputs = []
            st.session_state.history_names = []
            st.rerun()
    
    st.markdown('<div class="batch-grid">', unsafe_allow_html=True)
    cols = st.columns(4)
    
    for i, img in enumerate(reversed(st.session_state.history_outputs[-12:])):
        with cols[i % 4]:
            st.image(img, use_container_width=True)
            st.caption(st.session_state.history_names[::-1][i][:15])
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# INFO & SUPPORT
# ================================
st.markdown("""
<div class="zcut-card">
    <h3 style="margin-bottom: 16px;">❓ How It Works</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        <div>
            <p style="font-weight: 700; color: #6366f1; margin-bottom: 8px;">1️⃣ Advanced AI Detection</p>
            <p style="font-size: 14px;">Our AI analyzes pixel-level details to identify objects</p>
        </div>
        <div>
            <p style="font-weight: 700; color: #6366f1; margin-bottom: 8px;">2️⃣ Multi-Pass Refinement</p>
            <p style="font-size: 14px;">4-pass processing ensures perfect edge accuracy</p>
        </div>
        <div>
            <p style="font-weight: 700; color: #6366f1; margin-bottom: 8px;">3️⃣ Color Enhancement</p>
            <p style="font-size: 14px;">Smart algorithms preserve natural colors & detail</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<hr style="margin: 50px 0; opacity: 0.3;">
<p style="text-align: center; font-size: 13px; opacity: 0.6;">
✨ Z_cut AI – Powered by _shamee28rtm<br>
Built with precision for creators, designers & professionals | Free forever
</p>
""", unsafe_allow_html=True)

