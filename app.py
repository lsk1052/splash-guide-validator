import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(
    page_title="스플래시 가이드 검증기",
    page_icon="🧭",
    layout="wide",
)


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


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    width: int = 5,
    dash: int = 24,
    gap: int = 14,
) -> None:
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


def apply_guide_overlay(image: Image.Image, os_name: str) -> Image.Image:
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

    # AD 마크 가이드 (우측 상단)
    ad_margin_top = 40
    ad_margin_right = 40
    ad_box_w = 170
    ad_box_h = 90
    ad_x2 = width - ad_margin_right
    ad_y1 = ad_margin_top
    ad_x1 = ad_x2 - ad_box_w
    ad_y2 = ad_y1 + ad_box_h
    draw_dashed_rectangle(draw, (ad_x1, ad_y1, ad_x2, ad_y2), yellow, width=4)
    draw.text((ad_x1 + 16, ad_y1 + 26), "AD", fill=yellow)

    # 노치 라벨
    label = "상단 노치 영역"
    label_x = 28
    label_y = min(notch_height - 44, 24)
    draw.rectangle([(label_x - 10, label_y - 6), (label_x + 220, label_y + 34)], fill=(0, 0, 0, 130))
    draw.text((label_x, label_y), label, fill=white)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


st.markdown(
    """
    <style>
    /* 1. 전체 앱 배경 */
    .stApp {
        background-color: #111111;
        color: #F2F2F2;
    }
    
    /* 2. 타이틀 및 포인트 컬러 */
    h1, h2, h3, h4 {
        color: #E60012 !important;
    }

    /* 3. 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #191919;
    }
    /* 사이드바 내 모든 글자 요소 화이트 고정 */
    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] li {
        color: #F2F2F2 !important;
    }
    /* 사이드바 내 수치 박스 배경 */
    [data-testid="stSidebar"] code {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }

    /* 4. 업로드 영역 디자인 수정 (이미지 안내 텍스트 포함) */
    [data-testid="stFileUploader"] section {
        background-color: #1E1E1E !important;
        border: 1px dashed #444444 !important;
    }
    
    /* 업로드 버튼 옆의 파일 제한 정보 텍스트를 밝게 */
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] small, 
    [data-testid="stFileUploader"] p {
        color: #BBBBBB !important; /* 약간 밝은 회색으로 가독성 확보 */
    }

    /* 업로드 버튼 자체 스타일 */
    [data-testid="stFileUploader"] button {
        background-color: #333333 !important;
        color: #F2F2F2 !important;
        border: 1px solid #555555 !important;
    }
    
    /* 업로드 아이콘 컬러 */
    [data-testid="stFileUploader"] svg {
        fill: #F2F2F2 !important;
    }

    /* 5. 검수 결과 스타일 */
    .check-pass { font-size: 2rem; font-weight: 800; color: #00E676; margin: 0.4rem 0 1rem 0; }
    .check-fail { font-size: 2rem; font-weight: 800; color: #FF5252; margin: 0.4rem 0 1rem 0; }
    .check-detail { font-size: 1.05rem; color: #DDDDDD; margin-bottom: 1.2rem; }
    
    .block-container { padding-top: 1.4rem; max-width: 1400px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("스플래시 가이드 검증기")
st.caption("브랜드사 시안 규격 및 안전영역(크롭/노치/AD 마크) 셀프 검수 도구")

with st.sidebar:
    st.header("검수 옵션")
    selected_os = st.radio(
        "OS 선택",
        options=["Android", "iOS"],
        index=0,
        horizontal=False,
    )
    st.write("---")
    st.write("**기준 규격**")
    st.write(f"- iOS: `{OS_SPECS['iOS']['size'][0]} x {OS_SPECS['iOS']['size'][1]} px`")
    st.write(f"- Android: `{OS_SPECS['Android']['size'][0]} x {OS_SPECS['Android']['size'][1]} px`")

uploaded_file = st.file_uploader(
    "시안 이미지를 업로드하세요 (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("좌측에서 OS를 선택한 뒤 이미지를 업로드하면 자동으로 검수가 진행됩니다.")
else:
    # 파일 용량 체크
    file_size_kb = uploaded_file.size / 1024  # 바이트를 KB로 변환
    SIZE_LIMIT_KB = 500
    is_size_valid = file_size_kb <= SIZE_LIMIT_KB

    # 이미지 규격 체크
    image = Image.open(uploaded_file).convert("RGB")
    actual_w, actual_h = image.size
    expected_w, expected_h = OS_SPECS[selected_os]["size"]
    is_dim_valid = (actual_w, actual_h) == (expected_w, expected_h)

    # 규격/용량 결과를 나란히 표시
    col1, col2 = st.columns(2)

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
        st.markdown(f"**용량:** {file_size_kb:.1f} KB (기준: {SIZE_LIMIT_KB} KB 이하)")

    st.divider()  # 결과와 이미지 사이 구분선

    # 가이드 오버레이 생성 및 출력
    result_image = apply_guide_overlay(image, selected_os)

    left, center, right = st.columns([1, 7, 1])
    with center:
        st.image(
            result_image,
            caption=f"{selected_os} 가이드 오버레이 결과",
            use_container_width=True,
        )
