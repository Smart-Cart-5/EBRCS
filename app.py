import streamlit as st

st.set_page_config(
    page_title="Zero-Shot Retail Checkout System",
    page_icon="🛒",
)

st.title("Welcome to the Zero-Shot Retail Checkout System")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This application provides a complete system for a zero-shot retail checkout experience. 
    
    **👈 Select a page from the sidebar** to get started.
    
    ### Pages:
    - **Add Product**: Onboard new products into the system by providing details and uploading images.
    - **Checkout**: Use your webcam to scan and identify products for billing.
    - **Validate Bill**: Review the items in your cart and see the final bill.
    """
)