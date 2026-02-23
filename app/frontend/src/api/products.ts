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
