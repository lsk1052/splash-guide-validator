import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import easyocr

# 1. 페이지 설정
st.set_page_config(
    page_title="스플래시 가이드 검증기",
    page_icon="🧭",
    layout="wide",
)

# 2. OCR 모델 캐싱 및 로드
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['ko', 'en'])

reader = get_ocr_reader()

def check_ad_text(image):
    img_np = np.array(image)
    results = reader.readtext(img_np)
    ad_keywords = ['광고', 'AD', '협찬', '할인', '구매']
    detected_ads = []
    for (bbox, text, prob) in results:
        for keyword in ad_keywords:
            if keyword in text:
                detected_ads.append({"text": text, "prob": prob})
    return detected_ads

# 3. OS별 규격 정의
OS_SPECS = {
    "iOS": {
        "size": (1580, 2795),
        "crop_side": 217,
        "notch_height": 328,
    },
    "Android": {
        "size": (1536, 2152),
        "crop_side": 328,
        "notch_height": 211,
    },
}

# 4. 그리기 헬퍼 함수
def draw_dashed_rectangle(draw, box, color, width=5, dash=24, gap=14):
    x1, y1, x2, y2 = box
    x = x1
    while x < x2:
        x_end = min(x + dash, x2)
        draw.line([(x, y1), (x_end, y1)], fill=color, width=width)
        draw.line([(x, y2), (x_end, y2)], fill=color, width=width)
        x += dash + gap
    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        draw.line([(x1, y), (x1, y_end)], fill=color, width=width)
        draw.line([(x2, y), (x2, y_end)], fill=color, width=width)
        y += dash + gap

def apply_guide_overlay(image, os_name):
    config = OS_SPECS[os_name]
    width, height = image.size
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    purple = (128, 0, 128, int(255 * 0.30))
    red = (255, 0, 0, int(255 * 0.30))
    yellow = (255, 215, 0, 255)
    white = (255, 255, 255, 255)

    crop_side = config["crop_side"]
    notch_height = config["notch_height"]

    draw.rectangle([(0, 0), (crop_side, height)], fill=purple)
    draw.rectangle([(width - crop_side, 0), (width, height)], fill=purple)
    draw.rectangle([(0, 0), (width, notch_height)], fill=red)

    ad_x2, ad_y1 = width - 40, 40
    ad_x1, ad_y2 = ad_x2 - 170, ad_y1 + 90
    draw_dashed_rectangle(draw, (ad_x1, ad_y1, ad_x2, ad_y2), yellow, width=4)
    draw.text((ad_x1 + 16, ad_y1 + 26), "AD", fill=yellow)

    label_x, label_y = 28, min(notch_height - 44, 24)
    draw.rectangle([(label_x - 10, label_y - 6), (label_x + 220, label_y + 34)], fill=(0, 0, 0, 130))
    draw.text((label_x, label_y), "상단 노치 영역", fill=white)

    return Image.alpha_composite(canvas, overlay).convert("RGB")

# 5. 디자인 스타일 (CSS)
st.markdown(
    """
    <style>
    .stApp { background-color: #111111; color: #F2F2F2; }
    h1, h2, h3, h4 { color: #E60012 !important; }
    [data-testid="stSidebar"] { background-color: #191919; }
    [data-testid="stSidebar"] * { color: #F2F2F2 !important; }
    [data-testid="stSidebar"] code { background-color: #333333 !important; color: #FFFFFF !important; }
    [data-testid="stFileUploader"] section { background-color: #1E1E1E !important; border: 1px dashed #444444 !important; }
    [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small { color: #BBBBBB !important; }
    [data-testid="stFileUploader"] button { background-color: #333333 !important; color: #F2F2F2 !important; }
    .check-pass { font-size: 2rem; font-weight: 800; color: #00E676; margin: 0.4rem 0 1rem 0; }
    .check-fail { font-size: 2rem; font-weight: 800; color: #FF5252; margin: 0.4rem 0 1rem 0; }
    .block-container { padding-top: 1.4rem; max-width: 1400px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 6. 메인 UI 레이아웃
st.title("스플래시 가이드 검증기")
st.caption("브랜드사 시안 규격 및 안전영역(크롭/노치/AD 마크) 셀프 검수 도구")

with st.sidebar:
    st.header("검수 옵션")
    selected_os = st.radio("OS 선택", options=["Android", "iOS"], index=0)
    st.write("---")
    st.write("**기준 규격**")
    st.write(f"- iOS: `{OS_SPECS['iOS']['size'][0]} x {OS_SPECS['iOS']['size'][1]} px`")
    st.write(f"- Android: `{OS_SPECS['Android']['size'][0]} x {OS_SPECS['Android']['size'][1]} px`")

uploaded_file = st.file_uploader("시안 이미지를 업로드하세요 (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("좌측에서 OS를 선택한 뒤 이미지를 업로드하면 자동으로 검수가 진행됩니다.")
else:
    # 1. 파일 정보 로드
    image = Image.open(uploaded_file).convert("RGB")
    actual_w, actual_h = image.size
    expected_w, expected_h = OS_SPECS[selected_os]["size"]
    file_size_kb = uploaded_file.size / 1024

    # 2. 검증 데이터 생성 (중요: UI 출력 전 수행)
    is_dim_valid = (actual_w, actual_h) == (expected_w, expected_h)
    is_size_valid = file_size_kb <= 500
    
    with st.spinner('이미지 내 텍스트 분석 중...'):
        detected_ad_list = check_ad_text(image)

    # 3. 결과 상단 요약 UI 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        if is_dim_valid:
            st.markdown('<div class="check-pass">✅ 규격 통과</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">❌ 규격 불일치</div>', unsafe_allow_html=True)
        st.markdown(f"**규격:** {actual_w}x{actual_h}px (기준: {expected_w}x{expected_h}px)")

    with col2:
        if is_size_valid:
            st.markdown('<div class="check-pass">✅ 용량 적합</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">❌ 용량 초과</div>', unsafe_allow_html=True)
        st.markdown(f"**용량:** {file_size_kb:.1f} KB (기준: 500 KB 이하)")

    with col3:
        if not detected_ad_list:
            st.markdown('<div class="check-pass">✅ 광고 텍스트 클린</div>', unsafe_allow_html=True)
            st.markdown("**결과:** 감지된 광고성 문구 없음")
        else:
            st.markdown('<div class="check-fail">⚠️ 광고 텍스트 감지</div>', unsafe_allow_html=True)
            for ad in detected_ad_list:
                st.write(f"- `{ad['text']}` ({int(ad['prob']*100)}%)")

    st.divider()

    # 4. 가이드 오버레이 이미지 출력
    result_image = apply_guide_overlay(image, selected_os)
    st.image(result_image, caption=f"{selected_os} 가이드 오버레이 결과", use_container_width=True)