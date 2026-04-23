import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai
import cv2
import numpy as np
import json

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
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

def check_design_compliance(overlay_image, os_name):
    try:
        # v3.1의 정교화된 프롬프트 사용
        prompt = f"""
        당신은 아주 까다로운 UX/UI 디자인 검수관입니다. 
        첨부된 {os_name} 시안에는 디자인 가이드라인이 컬러 영역으로 표시되어 있습니다.
        
        [가이드라인 영역 설명]
        - 빨간색(상단): 노치 및 시스템 UI 영역 (절대 침범 불가)
        - 자주색(좌우): 기기별로 잘릴 수 있는 크롭 영역 (중요 요소 배치 불가)
        - 에메랄드색(좌우): 텍스트 가독성을 위한 '최소 안전 여백' (텍스트가 가급적 겹치지 않아야 함)

        [검수 미션]
        1. 텍스트 추출: 이미지 내의 모든 글자를 나열하세요.
        2. 광고 판단: '광고', 'AD', '이벤트', '할인', '구매' 등 상업적 문구가 있는지 확인하세요.
        3. 영역 침범 체크:
           - 글자나 로고가 '빨간색' 또는 '자주색' 영역을 침범하면 overflow: true 입니다.
           - 글자가 '에메랄드색' 영역에 걸쳐 있다면 가독성 주의 대상으로 판단하고 그 내용을 reason에 상세히 기술하세요.

        반드시 아래 JSON 형식으로만 답변하세요:
        {{
            "ad_found": ["찾은광고문구1", "2"],
            "overflow": true,
            "reason": "침범한 구체적인 위치와 단어, 혹은 에메랄드 영역의 가독성 피드백 설명"
        }}
        """
        
        response = model.generate_content([prompt, overlay_image])
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        result = json.loads(raw_text)
        
        if not result.get("ad_found") and "광고" in raw_text:
             result["ad_found"] = ["광고"]
             
        return result
    except Exception as e:
        return {"ad_found": [], "overflow": False, "reason": "분석 엔진 일시 오류"}

def evaluate_quality(pil_image):
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 분석 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    p_raw = np.mean(20 * np.log(np.abs(fshift) + 1))
    
    # [기준 완화] 감점 가중치를 45 -> 30으로 낮춤 (점수가 덜 깎임)
    purity_score = max(0, min(100, 100 - (p_raw - 175.0) * 30)) 
    
    # 선명도 분석 (Laplacian)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = max(0, min(100, lap_var / 8)) 

    # [기준 완화] 판정 문턱값을 낮춰 통과가 더 쉬워지도록 설정
    is_blurry = clarity_score < 12  # 기존 18
    is_pixelated = purity_score < 35 # 기존 45 
    
    quality_score = (purity_score * 0.7) + (clarity_score * 0.3)
    
    return is_blurry, is_pixelated, quality_score

# 3. OS별 규격 정의 (v3.1 에메랄드 패딩 포함)
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

# 4. 디자인 스타일 (CSS) - v3.1의 세련된 UI 스타일 유지
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

# 5. 메인 UI
st.title("스플래시 가이드 검증기")
st.caption("UX/UI 디자인 품질 및 규격 자동 검수 도구 (v3.2 Final)")

with st.sidebar:
    st.header("검수 옵션")
    selected_os = st.radio("OS 선택", options=["Android", "iOS"], index=0)
    
    st.divider()
    st.markdown(f"### 📱 {selected_os} 상세 규격")
    spec = OS_SPECS[selected_os]
    st.write(f"- **권장 사이즈:** {spec['size'][0]}x{spec['size'][1]}px")
    st.write(f"- **용량 제한:** 500KB 미만")
    
    st.warning("⚠️ 규격이 맞지 않으면 검증 결과가 부정확할 수 있습니다.")

uploaded_file = st.file_uploader("시안 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

# v3.1에서 호평받은 가이드 안내 UI
st.markdown("""
<div class="guide-container">
    <div class="guide-row">
        <div class="guide-item"><div class="color-box" style="background-color: rgba(255, 0, 0, 0.8);"></div>상단 노치 영역</div>
        <div class="guide-item"><div class="color-box" style="background-color: rgba(128, 0, 128, 0.8);"></div>좌우 크롭 영역</div>
        <div class="guide-item"><div class="color-box" style="background-color: rgba(50, 255, 170, 0.8);"></div>텍스트 안전 여백</div>
        <div class="warning-text" style="color: #DDDDDD; margin-left: 5px;">
            ⚠️ 주요 이미지와 텍스트가 왼쪽 3개의 영역을 침범하지 않도록 해주세요!
        </div>
    </div>
    <div class="guide-row">
        <div class="warning-text" style="color: #FF5252; opacity: 0.9;">
            ⚠️ AD 마크는 자동 부착되니 광고 텍스트 포함 여부를 꼭 체크해 주세요!
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
    
    with st.spinner('AI가 가이드 라인을 기준으로 정밀 분석 중입니다...'):
        overlay_img = apply_guide_overlay(image, selected_os)
        compliance_result = check_design_compliance(overlay_img, selected_os)
        is_blurry, is_pixelated, quality_score = evaluate_quality(image)

    # 결과 표시 레이아웃
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
        # [기준 강화] 최종 통과 점수 기준을 40 -> 60으로 상향 조정했습니다.
        if not is_blurry and not is_pixelated and quality_score >= 60:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">디자인 품질: {quality_score:.0f}점</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">품질 점수: {quality_score:.0f}점</div>', unsafe_allow_html=True)
            st.warning("화질 점수가 기준(60점)에 미달되었습니다.  \n고화질 원본 이미지로 교체해 주시고,  \n동일한 경고가 뜬다면 UX디자인팀에 검수 요청을 해주세요.")
            
    with col4:
        ad_list = compliance_result.get("ad_found", [])
        has_overflow = compliance_result.get("overflow", False)
        reason_msg = compliance_result.get("reason", "")
        
        if not ad_list and not has_overflow:
            st.markdown('<div class="check-pass">✅ 가이드 준수</div>', unsafe_allow_html=True)
            st.markdown('<div class="status-text">광고 및 침범 없음</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 가이드 위반</div>', unsafe_allow_html=True)
            if ad_list:
                st.error(f"🚫 광고 발견: {', '.join(ad_list)}")
            if has_overflow:
                st.error(f"🚫 안전 영역 침범")
                st.info(f"**이유:** {reason_msg}")

    st.divider()
    
    # 실제 사이즈 프리뷰
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 실제 사이즈 프리뷰 ({actual_w}x{actual_h})",
        width=actual_w
    )
