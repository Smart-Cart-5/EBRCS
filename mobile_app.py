import streamlit as st

from mobile_nav import MOBILE_NAV_ITEMS, MOBILE_NAV_TO_PAGE
from ui_theme import apply_theme

apply_theme(
    page_title="모바일 체크아웃",
    page_icon="📱",
    current_nav="📱 모바일 홈",
    nav_items=MOBILE_NAV_ITEMS,
    nav_to_page=MOBILE_NAV_TO_PAGE,
    nav_key_prefix="mobile",
)

st.session_state.navigation_mode = "mobile"
st.session_state.home_page_path = "mobile_app.py"
st.session_state.checkout_page_path = "pages/4_Checkout_Mobile.py"

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
      <div class="icon-square" style="background:linear-gradient(135deg,#3B82F6,#2563EB);">📱</div>
      <div>
        <h1 class="page-title">Iriun 모바일 체크아웃</h1>
        <p class="subtitle-text">iPhone을 Iriun Webcam으로 연결해 실시간 체크아웃을 실행합니다.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown(
        """
        <div class="soft-card card-hover">
          <div style="display:flex; gap:14px; align-items:flex-start; margin-bottom:10px;">
            <div class="icon-square" style="background:linear-gradient(135deg,#10B981,#059669);">📦</div>
            <div>
              <h3 class="card-title">모바일 체크아웃 시작</h3>
              <p class="subtitle-text" style="margin:6px 0 0 0;">카메라 인덱스를 선택하고 ROI 없이 실시간 인식을 시작합니다.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("모바일 체크아웃 열기", key="mobile_home_start_checkout", type="primary"):
        st.switch_page("pages/4_Checkout_Mobile.py")

with right_col:
    st.markdown(
        """
        <div class="soft-card card-hover">
          <div style="display:flex; gap:14px; align-items:flex-start; margin-bottom:10px;">
            <div class="icon-square" style="background:linear-gradient(135deg,#FFB74D,#FF8A65);">✅</div>
            <div>
              <h3 class="card-title">영수증 확인</h3>
              <p class="subtitle-text" style="margin:6px 0 0 0;">인식된 상품 목록을 검수하고 수량을 수정합니다.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("영수증 확인", key="mobile_home_open_receipt"):
        st.switch_page("pages/3_Validate_Bill.py")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="soft-card">
      <h2 class="section-title" style="margin-bottom:14px;">iPhone + macOS 연결 순서</h2>
      <div style="display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px;">
        <div style="padding:16px; border-radius:12px; border:1px solid rgba(0,0,0,0.08); background:#FBFBFB;">
          <div style="font-weight:700; font-size:16px; color:#030213; margin-bottom:4px;">1) Mac 설치</div>
          <div class="card-subtitle">macOS에 Iriun Webcam Desktop 앱을 설치하고 실행합니다.</div>
        </div>
        <div style="padding:16px; border-radius:12px; border:1px solid rgba(0,0,0,0.08); background:#FBFBFB;">
          <div style="font-weight:700; font-size:16px; color:#030213; margin-bottom:4px;">2) iPhone 설치</div>
          <div class="card-subtitle">iPhone App Store에서 Iriun Webcam 앱을 설치하고 실행합니다.</div>
        </div>
        <div style="padding:16px; border-radius:12px; border:1px solid rgba(0,0,0,0.08); background:#FBFBFB;">
          <div style="font-weight:700; font-size:16px; color:#030213; margin-bottom:4px;">3) 같은 네트워크</div>
          <div class="card-subtitle">Mac과 iPhone을 동일한 Wi-Fi(또는 USB 테더링)로 연결합니다.</div>
        </div>
        <div style="padding:16px; border-radius:12px; border:1px solid rgba(0,0,0,0.08); background:#FBFBFB;">
          <div style="font-weight:700; font-size:16px; color:#030213; margin-bottom:4px;">4) 카메라 선택</div>
          <div class="card-subtitle">모바일 체크아웃 페이지에서 Iriun 인덱스를 선택해 스트리밍을 시작합니다.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
if st.button("데스크톱 홈으로 이동", key="mobile_to_desktop_home"):
    st.switch_page("pages/0_Desktop_Home.py")
