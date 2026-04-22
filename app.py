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

def check_design_violations(image, os_name):
    config = OS_SPECS[os_name]
    w, h = image.size
    
    prompt = f"""
    이 이미지는 {os_name}용 앱 스플래시 화면입니다. 
    1. 광고성 문구('광고', 'AD', '할인' 등)가 있는지 확인하세요.
    2. 모든 텍스트의 위치를 확인하고, 다음 영역을 침범하는지 확인하세요:
       - 좌우 여백: {config['crop_side']}px 이내
       - 상단 노치: {config['notch_height']}px 이내
    결과를 '광고: [단어들], 넘침: [예/아니오]' 형태로 알려주세요.
    """
    # 이후 Gemini의 답변을 파싱하여 UI에 빨간색 경고를 띄움
        response = model.generate_content([prompt, image])
        result_text = response.text.strip()
        if "없음" in result_text or not result_text: return []
        found_words = [word.strip() for word in result_text.split(',')]
        return [{"text": word, "prob": 1.0} for word in found_words]
    except Exception: return []

def evaluate_quality(pil_image):
    # 이미지를 분석하기 좋게 변환
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. 선명도 분석 (Sobel)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_score = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # 2. 주파수 분석 (FFT) - 픽셀 깨짐 감지
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    p_score = np.mean(magnitude_spectrum)

    # --- [이선경 님 데이터 맞춤형 정밀 튜닝] ---
    # 직접 확인하신 수치(175.5 vs 177.1)를 바탕으로 176.5를 기준점으로 잡습니다. 
    is_blurry = edge_score < 15.0      # 선명도 기준 (낮으면 흐림) 
    is_pixelated = p_score > 176.5     # 노이즈 기준 (높으면 깨짐) 
    
    # [유지] 종합 점수 (UI 표시용)
    # 선명도(edge_score)를 화면에 대표 점수로 보여줍니다. 
    quality_score = edge_score 
    
    return is_blurry, is_pixelated, quality_score, p_score

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
    
    # [수정] 실제 사이즈로 표시하되, 너무 크면 브라우저 너비에 맞춤
    # width=actual_w를 명시하면 Streamlit이 해당 픽셀 너비로 렌더링을 시도합니다.
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 실제 사이즈 프리뷰 ({actual_w}x{actual_h})",
        width=actual_w # 너무 클 수 있어 절반(50%) 사이즈로 제안하거나, actual_w 그대로 사용하세요.
    )
