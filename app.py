import streamlit as st

from ui_theme import apply_theme

apply_theme(page_title="스마트 체크아웃", page_icon="🏪", current_nav="🏠 홈")

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
      <div class="icon-square" style="background:#E8FFF3; color:#10B981;">✅</div>
      <div>
        <h1 class="page-title">시스템 준비 완료</h1>
        <p class="subtitle-text">스마트 체크아웃 시스템에 오신 것을 환영합니다</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="soft-card card-hover">
          <div style="display:flex; gap:16px; align-items:flex-start; margin-bottom:12px;">
            <div class="icon-square" style="background:linear-gradient(135deg,#FFB74D,#FF8A65);">🛒</div>
            <div>
              <h3 class="card-title">체크아웃</h3>
              <p class="subtitle-text" style="margin:6px 0 0 0;">실시간 카메라 인식으로 장바구니를 자동 집계하고 ROI 영역 설정으로 정확도를 높입니다.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("체크아웃 시작", key="home_start_checkout", type="primary"):
        st.switch_page("pages/2_Checkout.py")

with col2:
    st.markdown(
        """
        <div class="soft-card card-hover">
          <div style="display:flex; gap:16px; align-items:flex-start; margin-bottom:12px;">
            <div class="icon-square" style="background:linear-gradient(135deg,#3B82F6,#2563EB);">✅</div>
            <div>
              <h3 class="card-title">영수증 확인</h3>
              <p class="subtitle-text" style="margin:6px 0 0 0;">체크아웃 후 인식된 상품 목록을 검수하고 수량을 수정한 뒤 영수증을 확정합니다.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("영수증 확인", key="home_open_receipt"):
        st.switch_page("pages/3_Validate_Bill.py")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="soft-card">
      <h2 class="section-title" style="margin-bottom:18px;">주요 기능</h2>
      <div style="display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:24px;">
        <div style="text-align:center;">
          <div class="icon-square" style="margin:0 auto 12px auto; background:#DCFCE7; color:#16A34A;">🎯</div>
          <div class="card-title" style="font-size:20px; margin-bottom:6px;">실시간 인식</div>
          <div class="card-subtitle">AI 기반 실시간 상품 인식으로 빠르고 정확한 체크아웃</div>
        </div>
        <div style="text-align:center;">
          <div class="icon-square" style="margin:0 auto 12px auto; background:#FFF1E7; color:#EA580C;">🧺</div>
          <div class="card-title" style="font-size:20px; margin-bottom:6px;">자동 장바구니</div>
          <div class="card-subtitle">인식된 상품이 자동으로 장바구니에 누적됩니다</div>
        </div>
        <div style="text-align:center;">
          <div class="icon-square" style="margin:0 auto 12px auto; background:#DBEAFE; color:#2563EB;">🧾</div>
          <div class="card-title" style="font-size:20px; margin-bottom:6px;">간편한 검증</div>
          <div class="card-subtitle">영수증 확인 페이지에서 수량을 쉽게 수정할 수 있습니다</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="soft-card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div>
          <h3 class="card-title">관리자 기능</h3>
          <p class="card-subtitle" style="margin-top:4px;">새 상품 이미지 등록(Add Product) 기능은 관리자 페이지에서 계속 사용할 수 있습니다.</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if st.button("상품 등록 페이지 열기", key="home_open_add_product"):
    st.switch_page("pages/1_Add_Product.py")
