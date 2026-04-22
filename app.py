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

def check_design_compliance(image, os_name):
    config = OS_SPECS[os_name]
    notch = config["notch_height"]
    side = config["crop_side"]
    
    try:
        # AI에게 광고 여부와 영역 침범 여부를 동시에 묻습니다.
        prompt = f"""
        당신은 UX/UI 디자인 검수 전문가입니다. 이 {os_name} 스플래시 시안을 분석하세요.
        
        1. 광고 확인: '광고', 'AD', '이벤트', '할인' 등 홍보성 문구가 있나요?
        2. 영역 침범 확인: 텍스트가 다음 '위험 영역'에 침범했나요?
           - 상단 {notch}px 영역 (빨간색 가이드 부분)
           - 좌우 각 {side}px 사이드 영역 (보라색 가이드 부분)
        
        결과는 반드시 다음 JSON 형식으로만 답변하세요:
        {{"ad_found": ["단어1", "단어2"], "overflow": true/false, "reason": "이유 요약"}}
        """
        
        response = model.generate_content([prompt, image])
        # JSON 응답을 안전하게 파싱하기 위한 처리
        import json
        result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        return result
    except Exception as e:
        # 에러 발생 시 기본값 반환
        return {"ad_found": [], "overflow": False, "reason": "분석 불가"}

def evaluate_quality(pil_image):
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. 원본 선명도/노이즈 분석
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_raw = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    p_raw = np.mean(20 * np.log(np.abs(fshift) + 1))

    # 2. [핵심] 디자이너용 점수 변환 로직
    # 정밀도(Purity): 175.5(최상) ~ 177.5(최악) 사이의 값을 100~0점으로 매핑
    # 1.6점의 차이를 60점 이상의 차이로 증폭시킵니다.
    purity_score = max(0, min(100, 100 - (p_raw - 175.0) * 40))
    
    # 선명도(Clarity): 텍스트에 속지 않도록 정밀도가 낮으면 선명도 점수도 깎습니다.
    clarity_score = max(0, min(100, edge_raw * 4))
    if purity_score < 50:
        clarity_score *= 0.6 # 픽셀이 깨졌다면 선명함은 '가짜'이므로 감점

    # 3. 판정 기준
    is_blurry = clarity_score < 40
    is_pixelated = purity_score < 55 # 55점 미만은 무조건 불합격
    
    # UI에 표시할 최종 품질 점수
    final_design_score = (purity_score * 0.7) + (clarity_score * 0.3)
    
    return is_blurry, is_pixelated, final_design_score, purity_score

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
    
    with st.spinner('AI가 광고 및 안전 영역을 정밀 검수 중입니다...'):
        compliance_result = check_design_compliance(image, selected_os)
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
        if not is_blurry and not is_pixelated:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">디자인 품질: {quality_score:.0f}점</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">품질 점수: {quality_score:.0f}점 (정밀도 부족)</div>', unsafe_allow_html=True)
            st.warning("픽셀 깨짐이 감지되었습니다. 고화질 원본을 사용하세요.")
            
    with col4:
        # 광고 및 영역 침범 결과 통합 표시
        ad_list = compliance_result.get("ad_found", [])
        has_overflow = compliance_result.get("overflow", False)
        
        if not ad_list and not has_overflow:
            st.markdown('<div class="check-pass">✅ 가이드 준수</div>', unsafe_allow_html=True)
            st.markdown('<div class="status-text">광고 및 영역 침범 없음</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 가이드 위반</div>', unsafe_allow_html=True)
            if ad_list:
                st.write(f"🚫 광고 감지: `{', '.join(ad_list)}`")
            if has_overflow:
                st.write(f"🚫 안전 영역 침범")
                st.caption(f"이유: {compliance_result.get('reason', '')}")

    st.divider()
    
    # [수정] 실제 사이즈로 표시하되, 너무 크면 브라우저 너비에 맞춤
    # width=actual_w를 명시하면 Streamlit이 해당 픽셀 너비로 렌더링을 시도합니다.
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 실제 사이즈 프리뷰 ({actual_w}x{actual_h})",
        width=actual_w # 너무 클 수 있어 절반(50%) 사이즈로 제안하거나, actual_w 그대로 사용하세요.
    )
