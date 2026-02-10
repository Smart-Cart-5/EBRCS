import { create } from "zustand";
import * as api from "../api/client";

interface SessionStore {
  sessionId: string | null;
  billingItems: Record<string, number>;
  itemScores: Record<string, number>;
  totalCount: number;
  lastLabel: string;
  lastScore: number;
  lastStatus: string;
  annotatedFrame: string | null; // base64 JPEG

  createSession: () => Promise<string>;
  updateFromWsMessage: (data: WsMessage) => void;
  setBilling: (items: Record<string, number>) => void;
  resetSession: () => void;
}

export interface WsMessage {
  frame: string;
  billing_items: Record<string, number>;
  item_scores: Record<string, number>;
  last_label: string;
  last_score: number;
  last_status: string;
  total_count: number;
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  sessionId: null,
  billingItems: {},
  itemScores: {},
  totalCount: 0,
  lastLabel: "",
  lastScore: 0,
  lastStatus: "",
  annotatedFrame: null,

  createSession: async () => {
    const { session_id } = await api.createSession();
    set({ sessionId: session_id, billingItems: {}, itemScores: {}, totalCount: 0 });
    return session_id;
  },

  updateFromWsMessage: (data: WsMessage) => {
    set({
      annotatedFrame: data.frame,
      billingItems: data.billing_items,
      itemScores: data.item_scores,
      lastLabel: data.last_label,
      lastScore: data.last_score,
      lastStatus: data.last_status,
      totalCount: data.total_count,
    });
  },

  setBilling: (items: Record<string, number>) => {
    set({ billingItems: items, totalCount: Object.values(items).reduce((a, b) => a + b, 0) });
  },

  resetSession: () => {
    set({
      sessionId: null,
      billingItems: {},
      itemScores: {},
      totalCount: 0,
      lastLabel: "",
      lastScore: 0,
      lastStatus: "",
      annotatedFrame: null,
    });
  },
}));
