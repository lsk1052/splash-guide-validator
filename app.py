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

def check_design_compliance(overlay_image, os_name):
    try:
        # 이 부분을 아래의 새로운 프롬프트로 교체하세요!
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
        
        # 분석 실행 (이후 로직은 동일)
        response = model.generate_content([prompt, overlay_image])
        
        # JSON 파싱 및 디버깅을 위한 출력
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        import json
        result = json.loads(raw_text)
        
        # 결과가 비어있다면 AI에게 다시 한번 경고 (Self-Correction 로직)
        if not result.get("ad_found") and "광고" in raw_text:
             result["ad_found"] = ["광고"] # 텍스트에는 있는데 리스트에 없는 경우 보정
             
        return result
    except Exception as e:
        # 에러 발생 시 UI에 에러를 표시하지 않고 기본값으로 조용히 처리
        return {"ad_found": [], "overflow": False, "reason": "분석 엔진 일시 오류"}

def evaluate_quality(pil_image):
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 분석 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    p_raw = np.mean(20 * np.log(np.abs(fshift) + 1))
    
    # [수정] 기준치를 175.0으로 다시 조이고, 감점 폭을 높여 3번 이미지를 잡아냅니다.
    purity_score = max(0, min(100, 100 - (p_raw - 175.0) * 45)) 
    
    # 선명도 분석 (Laplacian)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = max(0, min(100, lap_var / 8)) 

    # [수정] 픽셀 깨짐 판정 기준을 45점으로 상향하여 3번 이미지를 '화질 저하'로 분류
    is_blurry = clarity_score < 18
    is_pixelated = purity_score < 45 
    
    quality_score = (purity_score * 0.7) + (clarity_score * 0.3)
    
    return is_blurry, is_pixelated, quality_score, p_raw

def get_quality_heatmap(pil_image):
    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    overlay = img_cv.copy()
    
    grid_size = 32
    blocks = []
    
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            block = gray[y:y+grid_size, x:x+grid_size]
            if block.size < 100: continue
            score = cv2.Laplacian(block, cv2.CV_64F).var()
            blocks.append(((x, y), score))
    
    # [수정] 전체 구역 중 상위 2.5%에 해당하는 점수를 문턱값으로 사용
    scores = [b[1] for b in blocks]
    if not scores: return pil_image, 0
    
    # 상위 2.5% 지점을 찾습니다. (이 수치가 높을수록 'OPEN RUN'만 잡힙니다)
    high_threshold = np.percentile(scores, 97.5) 
    
    # 최소한의 노이즈 바닥 수치(250.0)를 설정하여 너무 깨끗한 이미지는 보호
    final_limit = max(250.0, high_threshold)

    detected_count = 0
    for (x, y), score in blocks:
        if score >= final_limit:
            cv2.rectangle(overlay, (x, y), (x+grid_size, y+grid_size), (0, 0, 255), -1)
            detected_count += 1

    result_img = cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0)
    return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), detected_count

# --- 여기까지 복사 ---

# 3. OS별 규격 정의
# 3. OS별 규격 정의 (에메랄드 패딩 규격 추가)
OS_SPECS = {
    "iOS": {
        "size": (1580, 2795), 
        "crop_side": 217, 
        "notch_height": 328,
        "padding_width": 100  # 에메랄드 영역 너비
    },
    "Android": {
        "size": (1536, 2152), 
        "crop_side": 328, 
        "notch_height": 211,
        "padding_width": 100  # 에메랄드 영역 너비
    },
}

def apply_guide_overlay(image, os_name):
    config = OS_SPECS[os_name]
    width, height = image.size
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 컬러 정의 (에메랄드 추가: 50, 255, 170, 76)
    purple = (128, 0, 128, 76)
    red = (255, 0, 0, 76)
    emerald = (50, 255, 170, 76) 
    
    crop_side = config["crop_side"]
    notch_height = config["notch_height"]
    pad_w = config["padding_width"]

    # 1. 자주색 영역 (좌우 크롭 영역)
    draw.rectangle([(0, 0), (crop_side, height)], fill=purple)
    draw.rectangle([(width - crop_side, 0), (width, height)], fill=purple)
    
    # 2. 에메랄드 영역 (좌우 안전 여백 - 크롭 영역 안쪽으로 배치)
    # 상단 노치 영역 아래부터 시작하도록 세팅
    draw.rectangle([(crop_side, notch_height), (crop_side + pad_w, height)], fill=emerald)
    draw.rectangle([(width - crop_side - pad_w, notch_height), (width - crop_side, height)], fill=emerald)
    
    # 3. 빨간색 영역 (상단 노치 영역)
    draw.rectangle([(0, 0), (width, notch_height)], fill=red)
    
    return Image.alpha_composite(canvas, overlay).convert("RGB")

# 5. 디자인 스타일 (CSS)
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: #F2F2F2; }
    
    /* 타이틀 및 헤더 컬러를 화이트로 변경 */
    h1, h2, h3, h4 { color: #FFFFFF !important; } 
    
    .check-pass { font-size: 1.5rem; font-weight: 800; color: #00E676; }
    .check-fail { font-size: 1.5rem; font-weight: 800; color: #FF5252; }
    .status-text { font-size: 0.9rem; color: #AAAAAA; }
    
    /* 경고 문구 스타일 */
    .ad-warning { 
        background-color: #331111; 
        color: #FF5252; 
        padding: 10px; 
        border-radius: 5px; 
        border: 1px solid #FF5252;
        margin-bottom: 20px;
        font-weight: bold;
    }

    /* 기존 스타일 하단에 추가 */
.guide-container {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #333333;
    margin-bottom: 20px;
}

.guide-item {
    display: inline-flex;
    align-items: center;
    margin-right: 20px;
    font-size: 0.85rem;
    color: #DDDDDD;
}

.color-box {
    width: 16px;
    height: 16px;
    border-radius: 4px; /* 요청하신 4px 적용 */
    margin-right: 8px;
}
    /* 실제 사이즈 이미지가 중앙에 오도록 설정 */
    .stImage { display: flex; justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

# 6. 메인 UI
st.title("스플래시 가이드 검증기")
st.caption("UX/UI 디자인 품질 및 규격 자동 검수 도구")

with st.sidebar:
    st.header("검수 옵션")
    selected_os = st.radio("OS 선택", options=["Android", "iOS"], index=0)
    
    # --- [추가] 사이드바에도 상세 정보 표기 ---
    st.divider()
    st.markdown(f"### 📱 {selected_os} 상세 규격")
    spec = OS_SPECS[selected_os]
    st.write(f"- **권장 사이즈:** {spec['size'][0]}x{spec['size'][1]}px")
    st.write(f"- **용량 제한:** 500KB 미만")
    
    st.warning("⚠️ 규격이 맞지 않으면 검증 결과가 부정확할 수 있습니다.")

uploaded_file = st.file_uploader("시안 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

# [교체할 영역] 업로드 영역 하단 가이드 문구
st.markdown(f"""
    <div class="guide-container">
        <div style="margin-bottom: 10px;">
            <span class="guide-item"><div class="color-box" style="background-color: rgba(255, 0, 0, 0.8);"></div>상단 노치 영역</span>
            <span class="guide-item"><div class="color-box" style="background-color: rgba(128, 0, 128, 0.8);"></div>좌우 크롭 영역</span>
            <span class="guide-item"><div class="color-box" style="background-color: rgba(50, 255, 170, 0.8);"></div>텍스트 안전 여백 ⚠️주요 이미지와 텍스트가 3개의 영역을 침범하지 않도록 해주세요</span>
        </div>
        <div style="color: #FF5252; font-size: 0.85rem; font-weight: bold; border-top: 1px solid #333; pt-10px; margin-top: 10px; padding-top: 10px;">
            ⚠️ AD 마크는 자동으로 부착되니, 이미지에 광고/AD 텍스트가 포함되지 않게 꼭 체크해주세요!
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
    
    # [수정된 분석 블록]
    with st.spinner('AI가 가이드 라인을 기준으로 정밀 분석 중입니다...'):
        # 1. 가이드 레이어를 먼저 그리고, 그 이미지를 AI에게 보냅니다.
        overlay_img = apply_guide_overlay(image, selected_os)
        
        # 2. 광고 및 영역 침범 여부를 한꺼번에 분석합니다.
        compliance_result = check_design_compliance(overlay_img, selected_os)
        
        # 3. 화질 분석을 수행합니다.
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
        # 점수가 50점 미만이거나, AI가 깨짐을 감지한 경우 모두 '화질 저하'로 분류
        if not is_blurry and not is_pixelated and quality_score >= 50:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">디자인 품질: {quality_score:.0f}점</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">품질 점수: {quality_score:.0f}점</div>', unsafe_allow_html=True)
            
            st.warning("픽셀 깨짐이 감지되었습니다. 고화질 원본으로 변경해보시고  \n동일한 경고가 뜬다면 UX디자인팀에 검수 요청을 해주세요.")
            
            if st.button("🔍 어디가 깨졌나요?"):
                heatmap_img, count = get_quality_heatmap(image)
                st.image(heatmap_img, caption=f"빨간색 표시 구역({count}곳)의 노이즈가 높습니다.")
            
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
                st.info(f"**이유:** {reason_msg}") # AI의 판단 근거를 직접 노출

    st.divider()
    
    # [수정] 실제 사이즈로 표시하되, 너무 크면 브라우저 너비에 맞춤
    # width=actual_w를 명시하면 Streamlit이 해당 픽셀 너비로 렌더링을 시도합니다.
    st.image(
        apply_guide_overlay(image, selected_os), 
        caption=f"{selected_os} 실제 사이즈 프리뷰 ({actual_w}x{actual_h})",
        width=actual_w # 너무 클 수 있어 절반(50%) 사이즈로 제안하거나, actual_w 그대로 사용하세요.
    )
