import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
    """Loads the CLIP model and processor."""
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)
    return model, processor

model, processor = load_model()

def init_app_state():
    if 'product_db' not in st.session_state:
        st.session_state.product_db = pd.DataFrame(
            columns=["sku", "name", "category", "price", "image_path"]
        )

    # Loading product database if it exists
    if os.path.exists(PRODUCT_DB):
        st.session_state.product_db = pd.read_csv(PRODUCT_DB)

    # Loading FAISS index if it exists
    if 'faiss_index' not in st.session_state:
        if os.path.exists(FAISS_INDEX_PATH):
            st.session_state.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            st.session_state.faiss_index = None

    if 'tracked_items' not in st.session_state:
        st.session_state.tracked_items = {}
    if 'billing_items' not in st.session_state:
        st.session_state.billing_items = {}

st.set_page_config(page_title="Add New Product")
st.title("Onboard New Product")

init_app_state()

st.header("Existing Products")
st.dataframe(st.session_state.product_db)

st.sidebar.title("Admin Panel")
st.sidebar.header("Onboard New Product")

new_sku = st.sidebar.text_input("SKU")
new_name = st.sidebar.text_input("Product Name")
new_category = st.sidebar.text_input("Category")
new_price = st.sidebar.number_input("Price", min_value=0.0, format="%.2f")
new_images = st.sidebar.file_uploader(
    "Upload Product Images (1-3)",
    accept_multiple_files=True,
    type=['jpg', 'png', 'jpeg']
)

def get_image_embedding(image_np: np.ndarray) -> np.ndarray:
    """
    Generates a L2-normalized CLIP image embedding (float32, 1D) 
    for cosine similarity (IndexFlatIP).
    """
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    pil_img = Image.fromarray(image_np)

    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)

    emb = feats.cpu().numpy().astype("float32").flatten()
    return emb

def add_product_to_db(sku, name, category, price, image_paths):
    """Adds a new product to the CSV database and FAISS index."""
    new_product_rows = []
    new_embeddings = []

    for img_path in image_paths:
        new_product_rows.append({
            "sku": sku,
            "name": name,
            "category": category,
            "price": price,
            "image_path": img_path
        })

        try:
            image = Image.open(img_path).convert("RGB")
            img_array = np.array(image)
            emb = get_image_embedding(img_array)
            new_embeddings.append(emb)
        except Exception as e:
            st.error(f"Error processing image {img_path}: {e}")
            return

    df_new = pd.DataFrame(new_product_rows)
    st.session_state.product_db = pd.concat(
        [st.session_state.product_db, df_new],
        ignore_index=True
    )
    st.session_state.product_db.to_csv(PRODUCT_DB, index=False)

    # Update FAISS index
    if new_embeddings:
        new_embeddings = np.array(new_embeddings, dtype="float32")

        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / np.maximum(norms, 1e-10)

        if st.session_state.faiss_index is None:
            dim = new_embeddings.shape[1]
            st.session_state.faiss_index = faiss.IndexFlatIP(dim)

        st.session_state.faiss_index.add(new_embeddings)
        faiss.write_index(st.session_state.faiss_index, FAISS_INDEX_PATH)

if st.sidebar.button("Add Product"):
    if new_sku and new_name and new_price > 0 and new_images:
        saved_image_paths = []
        for uploaded_file in new_images:
            ext = os.path.splitext(uploaded_file.name)[1]
            filename = f"{new_sku}_{uuid.uuid4()}{ext}"
            img_path = os.path.join(IMAGE_DIR, filename)

            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            saved_image_paths.append(img_path)

        add_product_to_db(new_sku, new_name, new_category, new_price, saved_image_paths)
        st.sidebar.success(f"Product '{new_name}' added successfully!")
    else:
        st.sidebar.warning("Please fill all fields and upload at least one image.")