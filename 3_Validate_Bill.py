import streamlit as st
import pandas as pd

st.set_page_config(page_title="Validate Bill")
st.title("Validate Your Bill")

def display_bill():
    st.header("Final Bill")

    # Initialize if needed
    if "billing_items" not in st.session_state:
        st.session_state.billing_items = {}

    items = st.session_state.billing_items

    if not items:
        st.warning("Your cart is empty. Please scan items in the Checkout page.")
        st.dataframe(pd.DataFrame(columns=["Product", "Quantity", "Price", "Total"]))
        st.metric("Total Amount", "₹.0.00")
        return

    # Building table
    bill_data = []
    total_amount = 0.0
    for sku, item in items.items():
        total_price = item["quantity"] * item["price"]
        bill_data.append({
            "SKU": sku,
            "Product": item["name"],
            "Quantity": item["quantity"],
            "Price": f"₹{item['price']:.2f}",
            "Total": f"₹{total_price:.2f}"
        })
        total_amount += total_price

    st.dataframe(pd.DataFrame(bill_data), use_container_width=True)
    st.metric("Total Amount", f"₹{total_amount:.2f}")

    st.subheader("Edit Your Bill")

    sku_list = list(items.keys())
    label_list = [f"{items[sku]['name']} ({sku})" for sku in sku_list]

    selected_label = st.selectbox(
        "Select item to remove from bill",
        label_list,
        key="validate_delete_select"
    )

    col_del1, col_del2 = st.columns(2)

    with col_del1:
        if st.button("Remove 1 Quantity", key="validate_remove_one"):
            idx = label_list.index(selected_label)
            sel_sku = sku_list[idx]
            if items[sel_sku]["quantity"] > 1:
                items[sel_sku]["quantity"] -= 1
            else:
                del items[sel_sku]
            st.session_state.billing_items = items
            st.success("Updated bill after removing 1 quantity.")

    with col_del2:
        if st.button("Remove Item Completely", key="validate_remove_full"):
            idx = label_list.index(selected_label)
            sel_sku = sku_list[idx]
            del items[sel_sku]
            st.session_state.billing_items = items
            st.success("Removed item completely from the bill.")

    if st.button("Clear Entire Bill", key="validate_clear_all"):
        st.session_state.billing_items = {}
        st.success("Cleared all items from the bill.")

    st.markdown("---------")

    # Confirm & Pay
    if st.button("Confirm and Pay", key="validate_confirm_pay"):
        st.success("Payment confirmed! Thank you for shopping with us.")
        st.session_state.billing_items = {}

display_bill()