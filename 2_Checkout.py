import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import uuid

DATA_DIR = "data"
PRODUCT_DB = os.path.join(DATA_DIR, "products.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
IMAGE_DIR = os.path.join(DATA_DIR, "product_images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

MODEL_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)
    return model, processor

model, processor = load_model()

def init_app_state():
    if "product_db" not in st.session_state:
        st.session_state.product_db = pd.DataFrame(
            columns=["sku", "name", "category", "price", "image_path"]
        )

    # Loading product DB
    if os.path.exists(PRODUCT_DB):
        st.session_state.product_db = pd.read_csv(PRODUCT_DB)

    # Loading FAISS index
    if "faiss_index" not in st.session_state:
        if os.path.exists(FAISS_INDEX_PATH):
            st.session_state.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            st.session_state.faiss_index = None

    if "billing_items" not in st.session_state:
        st.session_state.billing_items = {}

init_app_state()

st.set_page_config(layout="wide", page_title="Checkout")
st.title("Live Checkout")

col1, col2 = st.columns([3, 2])

with col1:
    st.header("Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.header("Billing & Tally")
    billing_placeholder = st.empty()

def get_image_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """Generate a CLIP embedding for the given BGR image crop."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)

    embedding = feats.cpu().numpy().astype("float32").flatten()
    return embedding

def update_billing_ui():
    with billing_placeholder.container():
        st.write("**Current Bill:**")
        items = st.session_state.billing_items

        if not items:
            st.dataframe(pd.DataFrame(columns=["Product", "Quantity", "Price", "Total"]))
            st.metric("Total Amount", "₹0.00")
        else:
            rows = []
            total_amount = 0.0

            for sku, item in items.items():
                qty = item["quantity"]
                price = item["price"]
                total_price = qty * price
                rows.append(
                    {
                        "Product": item["name"],
                        "Quantity": qty,
                        "Price": f"₹{price:.2f}",
                        "Total": f"₹{total_price:.2f}",
                    }
                )
                total_amount += total_price

            st.dataframe(pd.DataFrame(rows))
            st.metric("Total Amount", f"₹{total_amount:.2f}")


def run_checkout_pipeline():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=50, varThreshold=50, detectShadows=False
    )

    MIN_AREA = 2500
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=4)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

        if candidates and st.session_state.faiss_index is not None \
           and st.session_state.faiss_index.ntotal > 0:

            main_cnt = max(candidates, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_cnt)

            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + 2 * pad)
            y2 = min(frame.shape[0], y + h + 2 * pad)

            w = x2 - x
            h = y2 - y

            if w > 20 and h > 20:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                crop = frame[y:y + h, x:x + w]

                if frame_count % 5 == 0:
                    emb = get_image_embedding(crop)
                    query = np.expand_dims(emb, axis=0)

                    D, I = st.session_state.faiss_index.search(query, 1)
                    best_idx = int(I[0][0])
                    best_score = float(D[0][0])

                    if best_score > 0.25 and not st.session_state.product_db.empty \
                            and best_idx < len(st.session_state.product_db):

                        prod = st.session_state.product_db.iloc[best_idx]
                        sku = str(prod["sku"])
                        name = str(prod["name"])
                        price = float(prod["price"])

                        label = f"{name} (₹{price:.2f})"
                        cv2.putText(
                            display_frame,
                            label,
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        if sku not in st.session_state.billing_items:
                            st.session_state.billing_items[sku] = {
                                "quantity": 1,
                                "name": name,
                                "price": price,
                            }

        video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
        update_billing_ui()

    cap.release()

if __name__ == "__main__":
    run_checkout_pipeline()