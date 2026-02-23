import { request } from "./base";

export interface SessionResponse {
  session_id: string;
}

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

export interface ConfirmBillingResponse {
  status: string;
  confirmed_items: Record<string, number>;
  confirmed_total: number;
  confirmed_total_amount: number;
  currency: string;
  unpriced_items: string[];
}

export function createSession(): Promise<SessionResponse> {
  return request("/sessions", {
    method: "POST",
  });
}

export function setROI(sessionId: string, points: number[][]) {
  return request(`/sessions/${sessionId}/roi`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
}

export function clearROI(sessionId: string) {
  return request(`/sessions/${sessionId}/roi`, {
    method: "DELETE",
  });
}

export function getBilling(sessionId: string): Promise<BillingState> {
  return request(`/sessions/${sessionId}/billing`);
}

export function updateBilling(sessionId: string, billingItems: Record<string, number>): Promise<BillingState> {
  return request(`/sessions/${sessionId}/billing`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ billing_items: billingItems }),
  });
}

export function confirmBilling(sessionId: string): Promise<ConfirmBillingResponse> {
  return request(`/sessions/${sessionId}/billing/confirm`, {
    method: "POST",
  });
}

export function uploadVideo(sessionId: string, file: File): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append("file", file);
  return request(`/sessions/${sessionId}/video-upload`, {
    method: "POST",
    body: form,
  });
}

export function videoStatusUrl(sessionId: string, taskId: string) {
  return `/api/sessions/${sessionId}/video-status?task_id=${encodeURIComponent(taskId)}`;
}

export function getHealth(): Promise<{ status: string; device: string; lora_loaded: boolean; index_vectors: number; active_sessions: number }> {
  return request("/health");
}

export function wsCheckoutUrl(sessionId: string) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;
  return `${protocol}://${host}/api/ws/checkout/${sessionId}`;
}
