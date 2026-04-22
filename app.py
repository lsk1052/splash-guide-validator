import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai
import cv2
import numpy as np
import io

# 1. 페이지 설정
st.set_page_config(
    page_title="스플래시 가이드 검증기",
    page_icon="🧭",
    layout="wide",
)

# 2. Gemini API 설정
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    # 사용자의 요청에 따라 안정적인 모델 설정 (기존 2.0 유지하되 에러 처리 강화)
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

def check_ad_text(image):
    try:
        prompt = """
        이 이미지는 모바일 앱의 스플래시 화면 시안입니다. 
        이미지 내부에 '광고', 'AD', '협찬', '할인', '구매'와 같은 광고성 텍스트가 포함되어 있는지 확인해주세요.
        만약 있다면 해당 단어들만 콤마(,)로 구분해서 답변해주고, 없다면 '없음'이라고만 답변하세요.
        """
        response = model.generate_content([prompt, image])
        result_text = response.text.strip()
        
        if "없음" in result_text or not result_text:
            return []
        
        found_words = [word.strip() for word in result_text.split(',')]
        return [{"text": word, "prob": 1.0} for word in found_words]
    except Exception as e:
        return []

def evaluate_quality(pil_image):
    # PIL 이미지를 OpenCV 형식(BGR)으로 변환
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. 흐릿함(Blur) 분석: Laplacian 분산값
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 픽셀 깨짐(Pixelation) 분석: 주파수 분석(FFT) 활용
    # 저해상도 이미지를 억지로 늘리면 특정 고주파 영역이 비정상적으로 왜곡됩니다.
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    pixel_score = np.mean(magnitude_spectrum)
    
    # 임계값 설정 (실제 시안 데이터로 튜닝 필요)
    is_blurry = blur_score < 80.0
    is_pixelated = pixel_score > 160.0  # 인위적 엣지가 과도하게 많음
    
    return is_blurry, is_pixelated, blur_score, pixel_score

# 3. OS별 규격 정의
OS_SPECS = {
    "iOS": {"size": (1580, 2795), "crop_side": 217, "notch_height": 328},
    "Android": {"size": (1536, 2152), "crop_side": 328, "notch_height": 211},
}

# 4. 가이드 레이어 그리기 함수
def apply_guide_overlay(image, os_name):
    config = OS_SPECS[os_name]
    width, height = image.size
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    purple, red = (128, 0, 128, 76), (255, 0, 0, 76)
    crop_side, notch_height = config["crop_side"], config["notch_height"]

    draw.rectangle([(0, 0), (crop_side, height)], fill=purple)
    draw.rectangle([(width - crop_side, 0), (width, height)], fill=purple)
    draw.rectangle([(0, 0), (width, notch_height)], fill=red)

    return Image.alpha_composite(canvas, overlay).convert("RGB")

# 5. 디자인 스타일 (CSS)
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: #F2F2F2; }
    h1, h2, h3, h4 { color: #E60012 !important; }
    .check-pass { font-size: 1.5rem; font-weight: 800; color: #00E676; }
    .check-fail { font-size: 1.5rem; font-weight: 800; color: #FF5252; }
    .status-text { font-size: 0.9rem; color: #AAAAAA; }
    </style>
    """, unsafe_allow_html=True)

# 6. 메인 UI
st.title("스플래시 가이드 검증기")
st.caption("UX/UI 디자인 품질 및 규격 자동 검수 도구")

with st.sidebar:
    st.header("검수 옵션")
    selected_os = st.radio("OS 선택", options=["Android", "iOS"], index=0)

uploaded_file = st.file_uploader("시안 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    actual_w, actual_h = image.size
    expected_w, expected_h = OS_SPECS[selected_os]["size"]
    file_size_kb = uploaded_file.size / 1024

    is_dim_valid = (actual_w, actual_h) == (expected_w, expected_h)
    is_size_valid = file_size_kb <= 1024 # 가이드 기준에 따라 조정
    
    with st.spinner('AI 분석 및 품질 검토 중...'):
        detected_ad_list = check_ad_text(image)
        is_blurry, is_pixelated, b_score, p_score = evaluate_quality(image)

    # UI 결과 표시 (4개 컬럼)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "check-pass" if is_dim_valid else "check-fail"
        st.markdown(f'<div class="{status}">{"✅ 규격 통과" if is_dim_valid else "❌ 규격 오류"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text">{actual_w}x{actual_h}px</div>', unsafe_allow_html=True)
    
    with col2:
        status = "check-pass" if is_size_valid else "check-fail"
        st.markdown(f'<div class="{status}">{"✅ 용량 적합" if is_size_valid else "❌ 용량 초과"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text">{file_size_kb:.1f} KB</div>', unsafe_allow_html=True)

    with col3:
        if not is_blurry and not is_pixelated:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">깨끗한 시안입니다.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            msg = "흐림" if is_blurry else "픽셀 깨짐"
            st.markdown(f'<div class="status-text">{msg} 현상 감지</div>', unsafe_allow_html=True)

    with col4:
        if not detected_ad_list:
            st.markdown('<div class="check-pass">✅ 광고 없음</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 광고 감지</div>', unsafe_allow_html=True)
            for ad in detected_ad_list: st.write(f"- `{ad['text']}`")

    st.divider()
    st.image(apply_guide_overlay(image, selected_os), use_container_width=True, caption=f"{selected_os} 안전 영역 가이드 적용 화면")
