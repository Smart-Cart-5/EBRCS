import { request } from "./base";

export interface ChatbotQueryRequest {
  question: string;
  session_id?: string;
}

export interface ChatbotProductMeta {
  name: string;
  quantity: number;
  product_name: string | null;
  item_no: string | null;
  unit_price: number | null;
  line_total: number;
  price_found: boolean;
}

export interface ChatbotQueryResponse {
  answer: string;
  cart: {
    items: ChatbotProductMeta[];
    total_count: number;
    total_price: number;
    priced_items: number;
    unpriced_items: string[];
  };
  cart_update?: {
    action: "add" | "remove" | "clear";
    item: string | null;
    quantity: number;
    new_quantity: number | null;
    billing_items: Record<string, number>;
    removed_items?: Record<string, number>;
    error?: string;
    candidates?: Array<{
      item_no: string;
      product_name: string;
      label: string;
    }>;
  } | null;
}

export function getChatbotSuggestions(sessionId?: string): Promise<{ suggestions: string[] }> {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  return request(`/chatbot/suggestions${query}`);
}

export function queryChatbot(payload: ChatbotQueryRequest): Promise<ChatbotQueryResponse> {
  return request("/chatbot/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
