import { request } from "./base";

export interface PurchaseItem {
  name: string;
  count: number;
  unit_price?: number | null;
  line_total?: number | null;
}

export interface PurchaseResponse {
  id: number;
  user_id: number;
  username: string;
  items: PurchaseItem[];
  total_amount: number;
  timestamp: string;
  notes?: string | null;
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
  recent_purchases: PurchaseResponse[];
  daily_stats: Array<{ date: string; purchase_count: number; revenue: number }>;
  total_revenue: number;
  average_order_value: number;
  today_revenue: number;
}

export function getMyPurchases(token: string): Promise<PurchaseResponse[]> {
  return request("/purchases/my", { token });
}

export function getAllPurchases(token: string): Promise<PurchaseResponse[]> {
  return request("/purchases/all", { token });
}

export function createPurchase(
  token: string,
  payload: { session_id: string; items: Array<{ name: string; count: number }>; notes?: string }
): Promise<PurchaseResponse> {
  return request("/purchases", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
  });
}

export function deletePurchase(token: string, purchaseId: number) {
  return request(`/purchases/${purchaseId}`, {
    method: "DELETE",
    token,
  });
}

export async function getDashboardStats(token: string, periodDays?: number): Promise<DashboardStats> {
  const query = periodDays ? `?period_days=${periodDays}` : "";
  const data = await request<DashboardStats>(`/purchases/dashboard${query}`, { token });
  return {
    ...data,
    daily_stats: data.daily_stats ?? [],
    total_revenue: data.total_revenue ?? 0,
    average_order_value: data.average_order_value ?? 0,
    today_revenue: data.today_revenue ?? 0,
  };
}
