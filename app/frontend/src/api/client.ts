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

export interface ROICalibrationResponse {
  phase: string;
  confirmed: boolean;
  has_pending_mask: boolean;
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

export function confirmROI(sessionId: string): Promise<ROICalibrationResponse> {
  return request(`/sessions/${sessionId}/roi/confirm`, { method: "POST" });
}

export function retryROI(sessionId: string): Promise<ROICalibrationResponse> {
  return request(`/sessions/${sessionId}/roi/retry`, { method: "POST" });
}

export interface SessionStateResponse {
  session_id: string;
  phase: string;
  cart_roi_confirmed: boolean;
  cart_roi_preview_ready: boolean;
  cart_roi_pending_polygon: number[][] | null;
  cart_roi_pending_ratio: number;
  cart_roi_auto_enabled: boolean | null;
  checkout_start_mode: string | null;
  cart_roi_available: boolean;
  cart_roi_unavailable_reason: string | null;
  last_roi_error: string | null;
  cart_roi_invalid_reason: string | null;
}

export function getSessionState(sessionId: string): Promise<SessionStateResponse> {
  return request(`/sessions/${sessionId}/state`);
}

export interface ROIModeResponse {
  session_id: string;
  cart_roi_auto_enabled: boolean;
  phase: string;
}

export function setROIMode(sessionId: string, enabled: boolean): Promise<ROIModeResponse> {
  return request(`/sessions/${sessionId}/roi/mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
}

export interface CheckoutStartResponse {
  session_id: string;
  requested_mode: "auto_roi" | "no_roi";
  effective_mode: "auto_roi" | "no_roi";
  phase: string;
  message: string | null;
}

export function checkoutStart(
  sessionId: string,
  mode: "auto_roi" | "no_roi",
): Promise<CheckoutStartResponse> {
  return request(`/sessions/${sessionId}/checkout/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
}

export interface WarpResponse {
  enabled: boolean;
  points: number[][] | null;
  size: number[];
}

export function setWarp(
  sessionId: string,
  points: number[][],
  enabled?: boolean,
  width?: number,
  height?: number,
): Promise<WarpResponse> {
  return request(`/sessions/${sessionId}/warp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points, enabled, width, height }),
  });
}

export function setWarpEnabled(
  sessionId: string,
  enabled: boolean,
): Promise<WarpResponse> {
  return request(`/sessions/${sessionId}/warp/enabled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
}

export function clearWarp(sessionId: string): Promise<WarpResponse> {
  return request(`/sessions/${sessionId}/warp`, { method: "DELETE" });
}

// --- Billing ---

export interface BillingState {
  billing_items: Record<string, number>;
  item_scores: Record<string, number>;
  total_count: number;
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
  return request<{ status: string; confirmed_items: Record<string, number> }>(
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
