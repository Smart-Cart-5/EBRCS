import streamlit as st

st.set_page_config(
    page_title="임베딩 기반 리테일 체크아웃 시스템",
    page_icon="🛒",
)

st.title("임베딩 기반 리테일 체크아웃 시스템")

st.sidebar.success("좌측 메뉴에서 페이지를 선택하세요.")

st.markdown(
    """
    이 애플리케이션은 임베딩 기반 리테일 체크아웃 시스템을 제공합니다.
    
    **👈 좌측 사이드바에서 페이지를 선택**해서 시작하세요.
    
    ### 페이지:
    - **Add Product**: 상품명과 이미지를 업로드해 인식 DB에 등록합니다.
    - **Checkout**: 웹캠으로 상품을 인식해 수량을 집계합니다.
    - **Validate Bill**: 인식된 상품 목록을 확인하고 수정합니다.
    """
)
