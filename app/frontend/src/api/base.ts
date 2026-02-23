export type RequestOptions = RequestInit & {
  token?: string;
};

const API_BASE = (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE || "/api";

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
  try {
    const data = await res.json();
    if (data?.detail) return String(data.detail);
    if (data?.message) return String(data.message);
    return JSON.stringify(data);
  } catch {
    try {
      const text = await res.text();
      return text || `HTTP ${res.status}`;
    } catch {
      return `HTTP ${res.status}`;
    }
  }
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
    throw new Error(message);
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  return (await res.text()) as unknown as T;
}
