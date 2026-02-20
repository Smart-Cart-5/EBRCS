// API base URL: use env var in production, proxy in development
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const BASE = `${API_BASE}/api`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// --- Sessions ---

export interface SessionResponse {
  session_id: string;
}

export function createSession(): Promise<SessionResponse> {
  return request("/sessions", { method: "POST" });
}

export function deleteSession(id: string): Promise<void> {
  return request(`/sessions/${id}`, { method: "DELETE" });
}

// --- ROI ---

export interface ROIResponse {
  points: number[][] | null;
  num_vertices: number;
}

export function setROI(
  sessionId: string,
  points: number[][],
): Promise<ROIResponse> {
  return request(`/sessions/${sessionId}/roi`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
}

export function clearROI(sessionId: string): Promise<void> {
  return request(`/sessions/${sessionId}/roi`, { method: "DELETE" });
}

// --- Billing ---

export interface BillingState {
  billing_items: Record<string, number>;
  item_scores: Record<string, number>;
  total_count: number;
  item_unit_prices: Record<string, number | null>;
  item_line_totals: Record<string, number>;
  total_amount: number;
  currency: string;
  unpriced_items: string[];
}

export function getBilling(sessionId: string): Promise<BillingState> {
  return request(`/sessions/${sessionId}/billing`);
}

export function updateBilling(
  sessionId: string,
  billing_items: Record<string, number>,
): Promise<BillingState> {
  return request(`/sessions/${sessionId}/billing`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ billing_items }),
  });
}

export function confirmBilling(sessionId: string) {
  return request<{
    status: string;
    confirmed_items: Record<string, number>;
    confirmed_total: number;
    confirmed_total_amount: number;
    currency: string;
    unpriced_items: string[];
  }>(
    `/sessions/${sessionId}/billing/confirm`,
    { method: "POST" },
  );
}

// --- Products ---

export interface Product {
  name: string;
  embedding_count: number;
}

export function listProducts(): Promise<{
  products: Product[];
  total_embeddings: number;
}> {
  return request("/products");
}

export function addProduct(name: string, images: File[]) {
  const form = new FormData();
  form.append("name", name);
  images.forEach((f) => form.append("images", f));
  return request<{ status: string; product_name: string }>("/products", {
    method: "POST",
    body: form,
  });
}

// --- Video Upload ---

export function uploadVideo(
  sessionId: string,
  file: File,
): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append("file", file);
  return request(`/sessions/${sessionId}/video-upload`, {
    method: "POST",
    body: form,
  });
}

// --- Health ---

export function getHealth(): Promise<Record<string, unknown>> {
  return request("/health");
}

// --- WebSocket URL helper ---

export function wsCheckoutUrl(sessionId: string): string {
  // Use backend URL for WebSocket in production
  const backendUrl = import.meta.env.VITE_API_BASE_URL || location.origin;
  const wsBase = backendUrl.replace(/^http/, "ws");
  return `${wsBase}/api/ws/checkout/${sessionId}`;
}

// --- SSE helper ---

export function videoStatusUrl(sessionId: string, taskId: string): string {
  return `${BASE}/sessions/${sessionId}/video-status?task_id=${taskId}`;
}

// --- Authentication ---

export interface User {
  id: number;
  username: string;
  name: string | null;
  role: string;
  is_active: boolean;
}

export interface SignupData {
  username: string;
  password: string;
  name: string;
  role?: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export function signup(data: SignupData): Promise<User> {
  return request("/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function login(username: string, password: string): Promise<LoginResponse> {
  const formData = new FormData();
  formData.append("username", username);
  formData.append("password", password);
  return request("/auth/login", {
    method: "POST",
    body: formData,
  });
}

export function getMe(token: string): Promise<User> {
  return request("/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  });
}

// --- Purchase History ---

export interface PurchaseItem {
  name: string;
  count: number;
  unit_price?: number | null;
  line_total?: number;
  currency?: string;
  product_name?: string;
  price_found?: boolean;
}

export interface Purchase {
  id: number;
  user_id: number;
  username: string;
  items: PurchaseItem[];
  total_amount: number;
  timestamp: string;
  notes: string | null;
}

export interface PurchaseCreate {
  session_id: string;
  items: PurchaseItem[];
  notes?: string;
}

export function getMyPurchases(token: string): Promise<Purchase[]> {
  return request("/purchases/my", {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export function getAllPurchases(token: string): Promise<Purchase[]> {
  return request("/purchases/all", {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export function createPurchase(
  token: string,
  data: PurchaseCreate
): Promise<Purchase> {
  return request("/purchases", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
}

export interface PopularProduct {
  name: string;
  total_count: number;
}

export interface DashboardStats {
  total_purchases: number;
  total_customers: number;
  today_purchases: number;
  total_products_sold: number;
  popular_products: PopularProduct[];
  recent_purchases: Purchase[];
}

export function getDashboardStats(token: string): Promise<DashboardStats> {
  return request("/purchases/dashboard", {
    headers: { Authorization: `Bearer ${token}` },
  });
}
