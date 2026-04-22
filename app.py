import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(
    page_title="스플래시 가이드 검증기",
    page_icon="🧭",
    layout="wide",
)

# 2. Gemini API 설정 (유지)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
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
        if "없음" in result_text or not result_text: return []
        found_words = [word.strip() for word in result_text.split(',')]
        return [{"text": word, "prob": 1.0} for word in found_words]
    except Exception: return []

def evaluate_quality(pil_image):
    # 이미지를 분석하기 좋게 변환 및 리사이즈
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. 선명도 분석 (Sobel 연산자 사용)
    # Laplacian보다 픽셀의 변화율을 더 정밀하게 측정합니다.
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_score = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # 2. 픽셀 노이즈/거칠기 분석 (입자감 체크)
    # 저화질 이미지에서 나타나는 특유의 자글자글한 노이즈를 측정합니다.
    noise_score = cv2.meanStdDev(gray)[1][0][0] 
    
    # 3. FFT (주파수 분석) - 픽셀 깨짐(계단 현상) 감지
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    pixel_score = np.mean(magnitude_spectrum)

    # --- [이선경 님의 이미지 맞춤형 임계값] ---
    # 원본(250점대)과 저화질(150점대 이하)을 가르는 기준입니다.
    # edge_score가 낮거나 pixel_score가 비정상적으로 높으면 필터링합니다.
    
    is_blurry = edge_score < 12.0    # 선명도 기준 (값이 낮을수록 흐림)
    is_pixelated = pixel_score > 185.0 # 픽셀 노이즈 기준 (값이 높을수록 깨짐)
    
    # 종합 점수 (UI 표시용)
    quality_score = edge_score 
    
    return is_blurry, is_pixelated, quality_score, pixel_score

# 3. OS별 규격 정의
OS_SPECS = {
    "iOS": {"size": (1580, 2795), "crop_side": 217, "notch_height": 328},
    "Android": {"size": (1536, 2152), "crop_side": 328, "notch_height": 211},
}

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
    /* 실제 사이즈 이미지가 중앙에 오도록 설정 */
    .stImage { display: flex; justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

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
    is_size_valid = file_size_kb <= 1024 
    
    with st.spinner('AI 분석 및 품질 검토 중...'):
        detected_ad_list = check_ad_text(image)
        # 중요: 세 번째 변수 이름을 quality_score로 받습니다.
        is_blurry, is_pixelated, quality_score, p_score = evaluate_quality(image)

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
        # 이 부분이 quality_score를 사용하는 출력 영역입니다.
        if not is_blurry and not is_pixelated:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">선명도: {quality_score:.1f} / 노이즈: {p_score:.1f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            reason = "픽셀 깨짐 및 노이즈" if is_pixelated else "이미지 흐림"
            st.markdown(f'<div class="status-text">{reason} (점수: {quality_score:.1f})</div>', unsafe_allow_html=True)
            st.warning("원본 파일(70% 이상 품질)을 사용해 주세요.")
            
    with col4:
        if not detected_ad_list: 
            st.markdown('<div class="check-pass">✅ 광고 없음</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 광고 감지</div>', unsafe_allow_html=True)
            for ad in detected_ad_list: st.write(f"- `{ad['text']}`")

    st.divider()
    
    # 실제 사이즈 프리뷰
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 실제 사이즈 프리뷰 ({actual_w}x{actual_h})",
        width=actual_w // 2 
    )
    
    # [수정] 실제 사이즈로 표시하되, 너무 크면 브라우저 너비에 맞춤
    # width=actual_w를 명시하면 Streamlit이 해당 픽셀 너비로 렌더링을 시도합니다.
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 실제 사이즈 프리뷰 ({actual_w}x{actual_h})",
        width=actual_w # 너무 클 수 있어 절반(50%) 사이즈로 제안하거나, actual_w 그대로 사용하세요.
    )
