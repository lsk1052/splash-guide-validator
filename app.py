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
    .stApp {
        background-color: #111111;
        color: #F2F2F2;
    }
    h1, h2, h3, h4 {
        color: #E60012 !important; /* 롯데ON 레드 포인트 */
    }
    .block-container {
        padding-top: 1.4rem;
        max-width: 1400px;
    }
    
    /* 사이드바 배경 설정 */
    [data-testid="stSidebar"] {
        background-color: #191919;
    }
    
    /* 사이드바 내부의 모든 텍스트(라벨, 설명, 코드 수치 등)를 하얗게 고정 */
    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] code {
        color: #F2F2F2 !important;
        -webkit-text-fill-color: #F2F2F2 !important;
    }

    /* 검수 결과 텍스트 스타일 */
    .check-pass {
        font-size: 2rem;
        font-weight: 800;
        color: #00E676;
        margin: 0.4rem 0 1rem 0;
    }
    .check-fail {
        font-size: 2rem;
        font-weight: 800;
        color: #FF5252;
        margin: 0.4rem 0 1rem 0;
    }
    .check-detail {
        font-size: 1.05rem;
        color: #DDDDDD;
        margin-bottom: 1.2rem;
    }
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
    image = Image.open(uploaded_file).convert("RGB")
    actual_w, actual_h = image.size
    expected_w, expected_h = OS_SPECS[selected_os]["size"]

    is_valid = (actual_w, actual_h) == (expected_w, expected_h)
    if is_valid:
        st.markdown('<div class="check-pass">✅ 규격 통과</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="check-fail">❌ 규격 불일치</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="check-detail">
        선택 OS: <b>{selected_os}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
        기준 규격: <b>{expected_w} x {expected_h}px</b> &nbsp;&nbsp;|&nbsp;&nbsp;
        업로드 이미지: <b>{actual_w} x {actual_h}px</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_image = apply_guide_overlay(image, selected_os)

    left, center, right = st.columns([1, 7, 1])
    with center:
        st.image(
            result_image,
            caption=f"{selected_os} 가이드 오버레이 결과",
            use_container_width=True,
        )
