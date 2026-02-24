import { request } from "./base";

export interface User {
  id: number;
  username: string;
  name?: string | null;
  role: string;
  is_active: boolean;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export function signup(payload: {
  username: string;
  password: string;
  name: string;
  role?: string;
}) {
  return request<User>("/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function login(username: string, password: string) {
  const form = new URLSearchParams();
  form.append("username", username);
  form.append("password", password);
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
}

export function getMe(token: string) {
  return request<User>("/auth/me", {
    method: "GET",
    token,
  });
}
