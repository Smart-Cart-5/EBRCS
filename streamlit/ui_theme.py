import streamlit as st

PRIMARY_GRADIENT = ("#FFB74D", "#FF8A65")
BACKGROUND = "#F7F8FA"
CARD_BG = "#FFFFFF"
TEXT_PRIMARY = "#030213"
TEXT_SECONDARY = "#717182"
BORDER = "rgba(0, 0, 0, 0.10)"
GREEN_ACCENT = "#10B981"
BLUE_ACCENT = "#3B82F6"

DEFAULT_NAV_ITEMS = ["🏠 홈", "🛒 체크아웃", "✅ 영수증 확인"]
DEFAULT_NAV_TO_PAGE = {
    "🏠 홈": "pages/0_Desktop_Home.py",
    "🛒 체크아웃": "pages/2_Checkout.py",
    "✅ 영수증 확인": "pages/3_Validate_Bill.py",
}


def init_session_defaults() -> None:
    if "items" not in st.session_state:
        st.session_state.items = [
            {"id": "1", "name": "사과", "count": 3},
            {"id": "2", "name": "바나나", "count": 2},
            {"id": "3", "name": "오렌지", "count": 1},
        ]

    if "roi_mode" not in st.session_state:
        st.session_state.roi_mode = False

    if "current_page" not in st.session_state:
        st.session_state.current_page = "홈"


def _theme_css() -> str:
    return f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

* {{
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Noto Sans KR', sans-serif;
}}

.stApp {{
  background: {BACKGROUND};
}}

main .block-container {{
  padding-top: 2rem;
  padding-bottom: 2rem;
}}

[data-testid="stSidebar"] {{
  background: #FFFFFF;
  min-width: 280px;
  max-width: 280px;
  border-right: 1px solid {BORDER};
}}

[data-testid="stSidebarNav"] {{
  display: none;
}}

.sidebar-logo {{
  padding: 6px 2px 18px 2px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  margin-bottom: 14px;
}}

.logo-row {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.logo-icon {{
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: linear-gradient(135deg, {PRIMARY_GRADIENT[0]}, {PRIMARY_GRADIENT[1]});
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: #fff;
}}

.logo-title {{
  color: {TEXT_PRIMARY};
  font-size: 18px;
  line-height: 1.2;
  font-weight: 700;
  margin: 0;
}}

.logo-subtitle {{
  color: {TEXT_SECONDARY};
  font-size: 14px;
  margin-top: 2px;
  font-weight: 500;
}}

.sidebar-footer {{
  margin-top: 24px;
  color: {TEXT_SECONDARY};
  font-size: 14px;
}}

[data-testid="stSidebar"] div[role="radiogroup"] {{
  gap: 8px;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"] {{
  border: 1px solid transparent;
  border-radius: 12px;
  padding: 6px 8px;
  margin-bottom: 8px;
  transition: all 0.2s ease;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
  background: rgba(255, 183, 77, 0.12);
  border-color: rgba(255, 183, 77, 0.28);
  box-shadow: inset 4px 0 0 {PRIMARY_GRADIENT[0]};
}}

[data-testid="stSidebar"] label[data-baseweb="radio"] p {{
  color: #4f5470;
  font-size: 16px;
  font-weight: 600;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p {{
  color: #141726;
  font-weight: 700;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{
  display: none;
}}

.page-title {{
  font-size: 32px;
  font-weight: 600;
  color: {TEXT_PRIMARY};
  margin: 0;
  line-height: 1.25;
}}

.section-title {{
  font-size: 20px;
  font-weight: 600;
  color: {TEXT_PRIMARY};
  margin: 0;
}}

.subtitle-text {{
  color: {TEXT_SECONDARY};
  font-size: 16px;
  margin-top: 4px;
  line-height: 1.6;
}}

.soft-card {{
  background: {CARD_BG};
  border-radius: 16px;
  border: 1px solid {BORDER};
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10);
  padding: 24px;
}}

.card-title {{
  font-size: 18px;
  font-weight: 600;
  color: {TEXT_PRIMARY};
  margin: 0;
}}

.card-subtitle {{
  font-size: 14px;
  color: {TEXT_SECONDARY};
  margin: 0;
  line-height: 1.55;
}}

.card-hover {{
  transition: all 0.2s ease;
}}

.card-hover:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12) !important;
}}

.icon-square {{
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  color: #fff;
}}

.icon-circle {{
  width: 48px;
  height: 48px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  color: #fff;
}}

.metric-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}}

.metric-card {{
  background: {CARD_BG};
  border-radius: 16px;
  border: 1px solid {BORDER};
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10);
  padding: 16px;
}}

.metric-label {{
  color: {TEXT_SECONDARY};
  font-size: 14px;
  font-weight: 500;
}}

.metric-value {{
  color: {TEXT_PRIMARY};
  font-size: 24px;
  font-weight: 600;
  margin-top: 6px;
}}

.camera-shell {{
  position: relative;
  background: #0c172c;
  border-radius: 16px;
  border: 1px solid {BORDER};
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10);
  padding: 14px;
}}

.live-badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: {GREEN_ACCENT};
  color: #fff;
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
}}

.live-dot {{
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #fff;
  animation: pulse 2s infinite;
}}

.roi-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}}

.pill-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 600;
}}

.product-list-wrap {{
  max-height: 420px;
  overflow-y: auto;
  padding-right: 2px;
}}

.product-item {{
  background: linear-gradient(90deg, #F9FAFB 0%, #FFFFFF 100%);
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 12px;
  transition: all 0.2s ease;
}}

.product-item:hover {{
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}}

.count-chip {{
  width: 40px;
  height: 40px;
  border-radius: 12px;
  background: linear-gradient(135deg, {PRIMARY_GRADIENT[0]}, {PRIMARY_GRADIENT[1]});
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: 600;
}}

.confidence-badge {{
  background: #DBEAFE;
  color: #1E40AF;
}}

.status-good {{
  color: {GREEN_ACCENT};
}}

.info-blue {{
  color: {BLUE_ACCENT};
}}

.warn-orange {{
  color: {PRIMARY_GRADIENT[1]};
}}

.stButton > button {{
  border-radius: 12px;
  padding: 12px 24px;
  font-size: 16px;
  font-weight: 600;
  transition: all 0.2s ease;
  border: 1px solid {BORDER};
}}

.stButton > button:hover {{
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}}

.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {PRIMARY_GRADIENT[0]}, {PRIMARY_GRADIENT[1]});
  color: #fff;
  border: none;
}}

.stButton > button[kind="primary"]:hover {{
  background: linear-gradient(135deg, #FFA726, #FF7043);
  box-shadow: 0 4px 12px rgba(255, 138, 101, 0.32);
}}

.loading-card {{
  text-align: center;
}}

.loading-spinner {{
  width: 58px;
  height: 58px;
  margin: 0 auto 12px auto;
  border-radius: 999px;
  border: 6px solid #FFE4D4;
  border-top-color: {PRIMARY_GRADIENT[1]};
  border-right-color: {PRIMARY_GRADIENT[0]};
  animation: spin 1s linear infinite;
}}

.loading-progress {{
  width: 100%;
  height: 12px;
  border-radius: 999px;
  background: #FFE8DA;
  position: relative;
  overflow: hidden;
  margin-top: 14px;
}}

.loading-progress::before {{
  content: "";
  position: absolute;
  top: 0;
  left: -40%;
  width: 40%;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(135deg, {PRIMARY_GRADIENT[0]}, {PRIMARY_GRADIENT[1]});
  animation: loading-slide 1.1s ease-in-out infinite;
}}

.loading-caption {{
  margin-top: 10px;
  color: {TEXT_SECONDARY};
  font-size: 14px;
}}

.loading-dots {{
  display: inline-flex;
  margin-left: 4px;
}}

.loading-dots span {{
  opacity: 0.25;
  animation: dot-blink 1.2s infinite;
}}

.loading-dots span:nth-child(2) {{
  animation-delay: 0.2s;
}}

.loading-dots span:nth-child(3) {{
  animation-delay: 0.4s;
}}

@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.45; }}
}}

@keyframes spin {{
  0% {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(360deg); }}
}}

@keyframes loading-slide {{
  0% {{ left: -40%; }}
  50% {{ left: 30%; }}
  100% {{ left: 100%; }}
}}

@keyframes dot-blink {{
  0%, 100% {{ opacity: 0.25; }}
  50% {{ opacity: 1; }}
}}
</style>
"""


def render_sidebar(
    current_nav: str,
    nav_items: list[str] | None = None,
    nav_to_page: dict[str, str] | None = None,
    nav_key_prefix: str = "default",
) -> None:
    items = nav_items or DEFAULT_NAV_ITEMS
    pages = nav_to_page or DEFAULT_NAV_TO_PAGE

    if current_nav not in items:
        current_nav = items[0]
    current_index = items.index(current_nav)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
              <div class="logo-row">
                <div class="logo-icon">🏪</div>
                <div>
                  <div style="font-size:18px; font-weight:700; color:#030213; line-height:1.2;">스마트 체크아웃</div>
                  <div class="logo-subtitle">Smart Checkout</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        choice = st.radio(
            "메뉴",
            items,
            index=current_index,
            label_visibility="collapsed",
            key=f"sidebar_nav_{nav_key_prefix}_{current_nav}",
        )

        st.markdown('<div class="sidebar-footer">© 2026 Smart Checkout</div>', unsafe_allow_html=True)

    if choice != current_nav:
        st.switch_page(pages[choice])


def apply_theme(
    page_title: str,
    page_icon: str,
    current_nav: str,
    nav_items: list[str] | None = None,
    nav_to_page: dict[str, str] | None = None,
    nav_key_prefix: str = "default",
) -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    init_session_defaults()
    st.markdown(_theme_css(), unsafe_allow_html=True)
    render_sidebar(
        current_nav,
        nav_items=nav_items,
        nav_to_page=nav_to_page,
        nav_key_prefix=nav_key_prefix,
    )
