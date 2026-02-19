import { create } from "zustand";
import * as api from "../api/client";

export interface CountEvent {
  product: string;
  track_id: string | null;
  quantity: number;
  action: "add" | "remove" | "unknown";
}

interface SessionStore {
  sessionId: string | null;
  billingItems: Record<string, number>;
  itemScores: Record<string, number>;
  totalCount: number;
  lastLabel: string;
  lastScore: number;
  lastStatus: string;
  annotatedFrame: string | null;
  roiPolygon: number[][] | null;
  countEvent: CountEvent | null;
  currentTrackId: string | null;

  createSession: () => Promise<string>;
  updateFromWsMessage: (data: WsMessage) => void;
  setBilling: (items: Record<string, number>) => void;
  resetSession: () => void;
}

export interface WsMessage {
  frame?: string;
  billing_items: Record<string, number>;
  item_scores: Record<string, number>;
  last_label: string;
  last_score: number;
  last_status: string;
  total_count: number;
  roi_polygon?: number[][] | null;
  count_event?: CountEvent | null;
  current_track_id?: string | null;
}

export const useSessionStore = create<SessionStore>((set) => ({
  sessionId: null,
  billingItems: {},
  itemScores: {},
  totalCount: 0,
  lastLabel: "",
  lastScore: 0,
  lastStatus: "",
  annotatedFrame: null,
  roiPolygon: null,
  countEvent: null,
  currentTrackId: null,

  createSession: async () => {
    const { session_id } = await api.createSession();
    set({ sessionId: session_id, billingItems: {}, itemScores: {}, totalCount: 0 });
    return session_id;
  },

  updateFromWsMessage: (data: WsMessage) => {
    set({
      annotatedFrame: data.frame ?? null,
      billingItems: data.billing_items,
      itemScores: data.item_scores,
      lastLabel: data.last_label,
      lastScore: data.last_score,
      lastStatus: data.last_status,
      totalCount: data.total_count,
      roiPolygon: data.roi_polygon ?? null,
      countEvent: data.count_event ?? null,
      currentTrackId: data.current_track_id ?? null,
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
      roiPolygon: null,
      countEvent: null,
      currentTrackId: null,
    });
  },
}));
