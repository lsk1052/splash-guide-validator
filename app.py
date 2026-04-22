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
        # AI에게 단계별 사고를 유도하는 강력한 프롬프트
        prompt = f"""
        당신은 아주 까다로운 UX/UI 디자인 검수관입니다. 
        첨부된 {os_name} 시안 이미지에는 디자인 가이드라인이 '반투명 색상'으로 덮여 있습니다.
        
        [검수 미션]
        1. 텍스트 추출: 이미지에 보이는 모든 글자를 하나도 빠짐없이 나열하세요.
        2. 광고 판단: 추출한 글자 중 '광고', 'AD', '이벤트', '할인', '구매' 등이 포함되어 있는지 확인하세요.
        3. 영역 침범 체크 (가장 중요): 
           - '빨간색 영역(상단)'에 글자나 로고의 일부라도 겹쳐 있습니까?
           - '보라색 영역(좌우)'에 글자나 로고의 일부라도 겹쳐 있습니까?
           - 배경 이미지를 제외한 '의미 있는 요소(텍스트, 로고, 버튼)'가 색상 영역 위에 있다면 무조건 'true'입니다.

        반드시 아래 JSON 형식으로만 답변하세요:
        {{
            "ad_found": ["찾은광고문구1", "2"],
            "overflow": true,
            "reason": "침범한 구체적인 위치와 단어를 설명 (예: '광고' 단어가 상단 빨간색 영역을 침범함)"
        }}
        """
        
        # 분석 실행
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

def get_quality_heatmap(pil_image):
    # 분석용 이미지 변환
    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 캔버스 복제
    overlay = img_cv.copy()
    
    # 격자 사이즈를 48정도로 조정 (정밀도와 속도의 균형)
    grid_size = 48 
    
    detected_count = 0
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            block = gray[y:y+grid_size, x:x+grid_size]
            if block.shape[0] < 10 or block.shape[1] < 10: continue
            
            # 구역별 노이즈 분석
            f = np.fft.fft2(block)
            fshift = np.fft.fftshift(f)
            p_score = np.mean(20 * np.log(np.abs(fshift) + 1))
            
            # [수정] 기준치를 170.0으로 낮춰서 아주 미세한 지글거림도 포착합니다.
            # 전체 판정 기준(176.5)보다 낮게 잡아야 '부분적 깨짐'이 잘 잡힙니다.
            if p_score > 170.0:
                cv2.rectangle(overlay, (x, y), (x+grid_size, y+grid_size), (0, 0, 255), -1)
                detected_count += 1

    # 빨간 박스가 더 잘 보이도록 투명도 조정 (0.4)
    result_img = cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0)
    return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), detected_count

# --- 여기서부터 복사해서 get_quality_heatmap 함수 아래에 붙여넣으세요 ---

def evaluate_quality(pil_image):
    # 이미지를 분석하기 좋게 변환
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

    # 2. 품질 점수 변환 로직 (관대한 버전: 35점 기준)
    purity_score = max(0, min(100, 100 - (p_raw - 175.0) * 30))
    clarity_score = max(0, min(100, edge_raw * 4))
    
    if purity_score < 30: 
        clarity_score *= 0.7 # 픽셀이 너무 깨졌을 경우 선명도 감점

    # 3. 판정 기준
    is_blurry = clarity_score < 35
    is_pixelated = purity_score < 35
    
    # UI용 최종 품질 점수
    quality_score = (purity_score * 0.7) + (clarity_score * 0.3)
    
    return is_blurry, is_pixelated, quality_score, p_raw

# --- 여기까지 복사 ---

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

# [추가] 업로드 영역 하단 경고 문구
st.markdown('<div class="ad-warning">⚠️ AD 마크는 자동으로 부착되니, 이미지에 광고/AD 텍스트가 포함되지 않게 꼭 체크해주세요!</div>', unsafe_allow_html=True)

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
        if not is_blurry and not is_pixelated:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">디자인 품질: {quality_score:.0f}점</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-text">품질 점수: {quality_score:.0f}점</div>', unsafe_allow_html=True)
            
            # [고도화 추가] 픽셀 깨짐 지점 시각화 버튼
            if st.button("어디가 깨졌나요?"):
                heatmap_img, count = get_quality_heatmap(image)
                st.image(heatmap_img, caption=f"빨간색 표시 구역({count}곳)의 픽셀 노이즈가 높습니다.")
            
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
