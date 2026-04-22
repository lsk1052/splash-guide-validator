import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(
    page_title="스플래시 가이드 검증기",
    page_icon="🧭",
    layout="wide",
)

# 2. Gemini API 설정 (Secrets에서 키를 가져옵니다)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')

def check_ad_text(image):
    try:
        # AI에게 전달할 명확한 프롬프트
        prompt = """
        이 이미지는 모바일 앱의 스플래시 화면 시안입니다. 
        이미지 내부에 '광고', 'AD', '협찬', '할인', '구매'와 같은 광고성 텍스트가 포함되어 있는지 확인해주세요.
        만약 있다면 해당 단어들만 콤마(,)로 구분해서 답변해주고, 없다면 '없음'이라고만 답변하세요.
        """
        # 이미지 분석 실행
        response = model.generate_content([prompt, image])
        result_text = response.text.strip()
        
        if "없음" in result_text or not result_text:
            return []
        
        found_words = [word.strip() for word in result_text.split(',')]
        return [{"text": word, "prob": 1.0} for word in found_words]
    except Exception as e:
        st.error(f"AI 분석 중 오류 발생: {e}")
        return []

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
    
    purple, red, yellow, white = (128, 0, 128, 76), (255, 0, 0, 76), (255, 215, 0, 255), (255, 255, 255, 255)
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
    .check-pass { font-size: 2rem; font-weight: 800; color: #00E676; }
    .check-fail { font-size: 2rem; font-weight: 800; color: #FF5252; }
    </style>
    """, unsafe_allow_html=True)

# 6. 메인 UI
st.title("스플래시 가이드 검증기")
st.caption("브랜드사 시안 규격 및 안전영역 셀프 검수 도구")

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
    is_size_valid = file_size_kb <= 500
    
    with st.spinner('AI가 광고 텍스트를 정밀 분석 중입니다...'):
        detected_ad_list = check_ad_text(image)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="{"check-pass" if is_dim_valid else "check-fail"}">{"✅ 규격 통과" if is_dim_valid else "❌ 규격 불일치"}</div>', unsafe_allow_html=True)
        st.write(f"규격: {actual_w}x{actual_h}px")
    with col2:
        st.markdown(f'<div class="{"check-pass" if is_size_valid else "check-fail"}">{"✅ 용량 적합" if is_size_valid else "❌ 용량 초과"}</div>', unsafe_allow_html=True)
        st.write(f"용량: {file_size_kb:.1f} KB")
    with col3:
        if not detected_ad_list:
            st.markdown('<div class="check-pass">✅ 광고 없음</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 광고 감지</div>', unsafe_allow_html=True)
            for ad in detected_ad_list: st.write(f"- `{ad['text']}`")

    st.divider()
    st.image(apply_guide_overlay(image, selected_os), use_container_width=True)