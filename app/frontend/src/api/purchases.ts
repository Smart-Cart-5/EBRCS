import { request } from "./base";

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

export interface PopularProduct {
  name: string;
  total_count: number;
}

export interface DashboardStats {
  total_purchases: number;
  total_customers: number;
  today_purchases: number;
  total_revenue: number;
  average_order_value: number;
  today_revenue: number;
  total_products_sold: number;
  daily_stats: Array<{
    date: string;
    purchase_count: number;
    revenue: number;
  }>;
  popular_products: PopularProduct[];
  recent_purchases: Purchase[];
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
  data: PurchaseCreate,
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

export function deletePurchase(token: string, purchaseId: number): Promise<{ status: string; purchase_id: number }> {
  return request(`/purchases/${purchaseId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });
}

export function getDashboardStats(token: string, days = 7): Promise<DashboardStats> {
  return request(`/purchases/dashboard?days=${encodeURIComponent(days)}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}
