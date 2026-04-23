import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(
    page_title="스플래시 가이드 검증기",
    page_icon="🧭",
    layout="wide",
)

# [삭제] Gemini API 설정 및 AI 모델 로드 섹션 제거

def evaluate_quality(pil_image):
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 분석 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    p_raw = np.mean(20 * np.log(np.abs(fshift) + 1))
    
    # 화질 점수 계산 로직 (수정된 기준 반영)
    purity_score = max(0, min(100, 100 - (p_raw - 175.0) * 30)) 
    
    # 선명도 분석 (Laplacian)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = max(0, min(100, lap_var / 8)) 

    # 판정 문턱값 (수정된 기준 반영)
    is_blurry = clarity_score < 12
    is_pixelated = purity_score < 35 
    
    quality_score = (purity_score * 0.7) + (clarity_score * 0.3)
    
    return is_blurry, is_pixelated, quality_score

# 2. OS별 규격 정의
OS_SPECS = {
    "iOS": {
        "size": (1580, 2795), 
        "crop_side": 217, 
        "notch_height": 328,
        "padding_width": 100 
    },
    "Android": {
        "size": (1536, 2152), 
        "crop_side": 328, 
        "notch_height": 211,
        "padding_width": 100 
    },
}

def apply_guide_overlay(image, os_name):
    config = OS_SPECS[os_name]
    width, height = image.size
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    purple = (128, 0, 128, 76)
    red = (255, 0, 0, 76)
    emerald = (50, 255, 170, 76) 
    
    crop_side = config["crop_side"]
    notch_height = config["notch_height"]
    pad_w = config["padding_width"]

    # 1. 자주색 영역 (크롭)
    draw.rectangle([(0, 0), (crop_side, height)], fill=purple)
    draw.rectangle([(width - crop_side, 0), (width, height)], fill=purple)
    
    # 2. 에메랄드 영역 (안전 여백)
    draw.rectangle([(crop_side, notch_height), (crop_side + pad_w, height)], fill=emerald)
    draw.rectangle([(width - crop_side - pad_w, notch_height), (width - crop_side, height)], fill=emerald)
    
    # 3. 빨간색 영역 (노치)
    draw.rectangle([(0, 0), (width, notch_height)], fill=red)
    
    return Image.alpha_composite(canvas, overlay).convert("RGB")

# 3. 디자인 스타일 (CSS)
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: #F2F2F2; }
    h1, h2, h3, h4 { color: #FFFFFF !important; } 
    
    .check-pass { font-size: 1.5rem; font-weight: 800; color: #00E676; }
    .check-fail { font-size: 1.5rem; font-weight: 800; color: #FF5252; }
    .status-text { font-size: 0.9rem; color: #AAAAAA; }

    .guide-container {
        background-color: #1E1E1E;
        padding: 15px 25px;
        border-radius: 12px;
        border: 1px solid #333333;
        margin-bottom: 25px;
        display: flex;
        flex-direction: column; 
        align-items: flex-start;
        gap: 10px;
        width: 100%;
    }

    .guide-row {
        display: flex;
        flex-wrap: wrap;            
        justify-content: flex-start;
        align-items: center;
        gap: 20px;
        width: 100%;
    }

    .guide-item {
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        color: #DDDDDD;
    }

    .color-box {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        margin-right: 8px;
        flex-shrink: 0;
    }

    .warning-text {
        font-size: 0.85rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 5px;
        line-height: 1.2;
    }

    .stImage { display: flex; justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

# 4. 메인 UI
st.title("스플래시 가이드 검증기")
st.caption("UX/UI 디자인 품질 및 규격 수동 검수 도구")

with st.sidebar:
    st.header("검수 옵션")
    selected_os = st.radio("OS 선택", options=["Android", "iOS"], index=0)
    
    st.divider()
    st.markdown(f"### 📱 {selected_os} 상세 규격")
    spec = OS_SPECS[selected_os]
    st.write(f"- **권장 사이즈:** {spec['size'][0]}x{spec['size'][1]}px")
    st.write(f"- **용량 제한:** 500KB 미만")

uploaded_file = st.file_uploader("시안 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

# 가이드 안내 UI (AD 마크 관련 문구 삭제)
st.markdown("""
<div class="guide-container">
    <div class="guide-row">
        <div class="guide-item"><div class="color-box" style="background-color: rgba(255, 0, 0, 0.8);"></div>상단 노치 영역</div>
        <div class="guide-item"><div class="color-box" style="background-color: rgba(128, 0, 128, 0.8);"></div>좌우 크롭 영역</div>
        <div class="guide-item"><div class="color-box" style="background-color: rgba(50, 255, 170, 0.8);"></div>텍스트 안전 여백</div>
    </div>
    <div class="guide-row">
        <div class="warning-text" style="color: #DDDDDD;">
            ⚠️ 하단 프리뷰 이미지를 보고 주요 이미지와 텍스트가 컬러 영역을 침범하지 않았는지 직접 확인해 주세요!
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    actual_w, actual_h = image.size
    expected_w, expected_h = OS_SPECS[selected_os]["size"]
    file_size_kb = uploaded_file.size / 1024

    is_dim_valid = (actual_w, actual_h) == (expected_w, expected_h)
    is_size_valid = file_size_kb <= 500 
    
    with st.spinner('이미지 품질을 분석 중입니다...'):
        is_blurry, is_pixelated, quality_score = evaluate_quality(image)

    # [수정] 결과를 3개 컬럼으로 재배치 (AI 분석 컬럼 삭제)
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "check-pass" if is_dim_valid else "check-fail"
        st.markdown(f'<div class="{status}">{"✅ 규격 통과" if is_dim_valid else "❌ 규격 오류"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text">{actual_w}x{actual_h}px</div>', unsafe_allow_html=True)
    
    with col2:
        status = "check-pass" if is_size_valid else "check-fail"
        st.markdown(f'<div class="{status}">{"✅ 용량 적합" if is_size_valid else "❌ 용량 초과"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text">{file_size_kb:.1f} KB</div>', unsafe_allow_html=True)
    
    with col3:
        if not is_blurry and not is_pixelated and quality_score >= 60:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">디자인 품질: {quality_score:.0f}점</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">품질 점수: {quality_score:.0f}점</div>', unsafe_allow_html=True)
            st.warning("화질 점수가 기준(60점)에 미달되었습니다. 원본 이미지를 확인해 주세요.")

    st.divider()
    
    # 실제 사이즈 프리뷰 (사용자가 직접 침범 여부 판단)
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 가이드라인 적용 프리뷰 (침범 여부를 직접 확인하세요)",
        width=actual_w
    )
