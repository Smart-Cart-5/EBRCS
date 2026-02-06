import html

import streamlit as st

from ui_theme import apply_theme

apply_theme(page_title="영수증 확인", page_icon="✅", current_nav="✅ 영수증 확인")

if "billing_items" not in st.session_state:
    st.session_state.billing_items = {}

items = st.session_state.billing_items

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
      <div class="icon-square" style="background:linear-gradient(135deg,#3B82F6,#2563EB);">✅</div>
      <div>
        <h1 class="page-title">영수증 확인</h1>
        <p class="subtitle-text">상품 목록을 확인하고 수정하세요</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

total_items = int(sum(items.values()))

summary_col1, summary_col2 = st.columns(2, gap="large")

with summary_col1:
    st.markdown(
        f"""
        <div class="soft-card">
          <div style="display:flex; align-items:center; gap:16px;">
            <div class="icon-square" style="width:56px; height:56px; background:linear-gradient(135deg,#FFB74D,#FF8A65);">🛍️</div>
            <div>
              <div class="card-subtitle">총 상품 수</div>
              <div style="font-size:28px; font-weight:600; color:#030213; margin-top:4px;">{total_items}개</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_col2:
    st.markdown(
        f"""
        <div class="soft-card">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
              <div class="card-subtitle">품목 수</div>
              <div style="font-size:28px; font-weight:600; color:#030213; margin-top:4px;">{len(items)}개</div>
            </div>
            <span class="pill-badge" style="background:#FFF1E7; color:#EA580C;">검수 가능</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="soft-card" style="padding:0; overflow:hidden;">
      <div style="padding:24px; border-bottom:1px solid rgba(0,0,0,0.1); background:linear-gradient(90deg,#F9FAFB 0%, #FFFFFF 100%);">
        <div style="display:flex; align-items:center; justify-content:space-between;">
          <h2 class="section-title">📋 상품 목록</h2>
          <span class="pill-badge" style="background:#FFB74D; color:#fff;">{len(items)}개 품목</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not items:
    st.markdown(
        """
        <div class="soft-card" style="margin-top:16px; text-align:center;">
          <div style="font-size:40px; margin-bottom:8px;">🧾</div>
          <div class="card-title" style="margin-bottom:6px;">아직 확인할 상품이 없습니다</div>
          <div class="card-subtitle">체크아웃 페이지에서 상품을 먼저 인식해주세요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("체크아웃으로 이동", type="primary", key="empty_to_checkout"):
        st.switch_page("pages/2_Checkout.py")
else:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    name_list = list(items.keys())
    for idx, name in enumerate(name_list):
        qty = int(items.get(name, 0))
        safe_name = html.escape(str(name))

        row_col1, row_col2, row_col3 = st.columns([4, 2, 1], gap="medium")

        with row_col1:
            st.markdown(
                f"""
                <div class="soft-card" style="padding:14px 16px;">
                  <div style="font-size:18px; font-weight:600; color:#030213; margin-bottom:2px;">{safe_name}</div>
                  <div class="card-subtitle">수량 조절 가능</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row_col2:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("➖", key=f"dec_{idx}", help="감소", use_container_width=True):
                    if items.get(name, 0) > 1:
                        items[name] = int(items[name]) - 1
                    else:
                        del items[name]
                    st.session_state.billing_items = items
                    st.rerun()
            with c2:
                st.markdown(
                    f"""
                    <div style="height:46px; border-radius:10px; background:linear-gradient(135deg,#FFB74D,#FF8A65); color:#fff;
                                display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:700;">
                      {qty}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                if st.button("➕", key=f"inc_{idx}", help="증가", use_container_width=True):
                    items[name] = int(items.get(name, 0)) + 1
                    st.session_state.billing_items = items
                    st.rerun()

        with row_col3:
            if st.button("🗑️", key=f"del_{idx}", help="삭제", use_container_width=True):
                if name in items:
                    del items[name]
                st.session_state.billing_items = items
                st.rerun()

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

total_items = int(sum(st.session_state.billing_items.values()))
st.markdown(
    f"""
    <div class="soft-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:20px; font-weight:600; color:#030213;">총 상품 수</span>
        <span style="font-size:32px; font-weight:700; color:#FF8A65;">{total_items}개</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

action_col1, action_col2 = st.columns(2, gap="large")
with action_col1:
    if st.button("취소", key="receipt_cancel", use_container_width=True):
        st.switch_page("app.py")
with action_col2:
    if st.button("영수증 확정", key="receipt_confirm", type="primary", use_container_width=True):
        st.success("영수증이 확정되었습니다!")
        st.session_state.billing_items = {}
