import { request } from "./base";

export interface Product {
  id?: number;
  item_no: string;
  name: string;
  price?: number;
  barcd?: string;
  embedding_count: number;
  label?: string;
}

export interface AddProductPayload {
  itemNo: string;
  name: string;
  price: number;
  barcd?: string;
  images: File[];
}

export interface ProductDetail {
  id: number;
  item_no: string;
  product_name: string;
  barcd: string | null;
  stock: number | null;
  stock_column: string | null;
  price: number | null;
  currency: string | null;
  price_source: string | null;
  price_checked_at: string | null;
  is_discounted: boolean | null;
  discount_rate: number | null;
  discount_amount: number | null;
  discount_updated_at: string | null;
  available_fields: {
    stock: boolean;
    discount: boolean;
  };
}

export interface UpdateProductDetailPayload {
  product_name?: string;
  barcd?: string | null;
  price?: number;
  stock?: number;
  is_discounted?: boolean;
  discount_rate?: number;
  discount_amount?: number;
}

export function listProducts(): Promise<{
  products: Product[];
  total_embeddings: number;
}> {
  return request("/products");
}

export function addProduct(payload: AddProductPayload) {
  const form = new FormData();
  form.append("item_no", payload.itemNo);
  form.append("name", payload.name);
  form.append("price", String(payload.price));
  if (payload.barcd?.trim()) {
    form.append("barcd", payload.barcd.trim());
  }
  payload.images.forEach((f) => form.append("images", f));

  return request<{
    status: string;
    item_no: string;
    product_name: string;
    label: string;
    price: number;
    images_count: number;
    total_products: number;
    total_embeddings: number;
  }>("/products", {
    method: "POST",
    body: form,
  });
}

export function deleteProduct(itemNo: string) {
  return request<{ status: string }>(`/products/${itemNo}`, {
    method: "DELETE",
  });
}

export function getProductDetail(itemNo: string): Promise<ProductDetail> {
  return request(`/products/${encodeURIComponent(itemNo)}/detail`);
}

export function updateProductDetail(itemNo: string, payload: UpdateProductDetailPayload) {
  return request<{ status: string; item_no: string }>(`/products/${encodeURIComponent(itemNo)}/detail`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}
