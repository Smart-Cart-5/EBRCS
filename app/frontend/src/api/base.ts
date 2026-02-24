import { useAuthStore } from "../stores/authStore";

export type RequestOptions = RequestInit & {
  token?: string;
};

const API_BASE = (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE || "/api";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function buildUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const base = API_BASE.replace(/\/$/, "");
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (normalized.startsWith("/api")) {
    return normalized;
  }
  return `${base}${normalized}`;
}

async function parseError(res: Response): Promise<string> {
  let raw = "";
  try {
    raw = await res.text();
  } catch {
    return `HTTP ${res.status}`;
  }

  if (!raw) {
    return `HTTP ${res.status}`;
  }

  try {
    const data = JSON.parse(raw);
    if (data?.detail) return String(data.detail);
    if (data?.message) return String(data.message);
    return JSON.stringify(data);
  } catch {
    return raw;
  }
}

function handleUnauthorized(token?: string): void {
  // Only clear auth when this was an authenticated request.
  if (!token) return;
  const { token: currentToken, user, clearAuth } = useAuthStore.getState();
  if (!currentToken && !user) return;
  clearAuth();
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { token, headers, ...rest } = options;
  const finalHeaders = new Headers(headers || {});
  if (token) {
    finalHeaders.set("Authorization", `Bearer ${token}`);
  }
  if (rest.body && typeof rest.body === "string" && !finalHeaders.has("Content-Type")) {
    finalHeaders.set("Content-Type", "application/json");
  }

  const res = await fetch(buildUrl(path), {
    ...rest,
    headers: finalHeaders,
  });

  if (!res.ok) {
    const message = await parseError(res);
    if (res.status === 401) {
      handleUnauthorized(token);
    }
    throw new ApiError(res.status, message);
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  return (await res.text()) as unknown as T;
}
