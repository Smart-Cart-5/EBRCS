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
  itemUnitPrices: Record<string, number | null>;
  itemLineTotals: Record<string, number>;
  totalCount: number;
  totalAmount: number;
  currency: string;
  unpricedItems: string[];
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
  setBillingState: (state: api.BillingState) => void;
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
  itemUnitPrices: {},
  itemLineTotals: {},
  totalCount: 0,
  totalAmount: 0,
  currency: "KRW",
  unpricedItems: [],
  lastLabel: "",
  lastScore: 0,
  lastStatus: "",
  annotatedFrame: null,
  roiPolygon: null,
  countEvent: null,
  currentTrackId: null,

  createSession: async () => {
    const { session_id } = await api.createSession();
    set({
      sessionId: session_id,
      billingItems: {},
      itemScores: {},
      itemUnitPrices: {},
      itemLineTotals: {},
      totalCount: 0,
      totalAmount: 0,
      currency: "KRW",
      unpricedItems: [],
    });
    return session_id;
  },

  updateFromWsMessage: (data: WsMessage) => {
    set((state) => {
      const nextItems = data.billing_items;
      const nextUnitPrices = Object.fromEntries(
        Object.entries(state.itemUnitPrices).filter(([name]) => name in nextItems),
      );
      const nextLineTotals = Object.fromEntries(
        Object.entries(nextItems).map(([name, qty]) => [
          name,
          nextUnitPrices[name] != null ? (nextUnitPrices[name] as number) * qty : 0,
        ]),
      );
      const nextTotalAmount = Object.values(nextLineTotals).reduce((sum, value) => sum + value, 0);
      const nextUnpricedItems = Object.entries(nextItems)
        .filter(([name]) => nextUnitPrices[name] == null)
        .map(([name]) => name);

      return {
        annotatedFrame: data.frame ?? null,
        billingItems: nextItems,
        itemScores: data.item_scores,
        itemUnitPrices: nextUnitPrices,
        itemLineTotals: nextLineTotals,
        totalAmount: nextTotalAmount,
        unpricedItems: nextUnpricedItems,
        lastLabel: data.last_label,
        lastScore: data.last_score,
        lastStatus: data.last_status,
        totalCount: data.total_count,
        roiPolygon: data.roi_polygon ?? null,
        countEvent: data.count_event ?? null,
        currentTrackId: data.current_track_id ?? null,
      };
    });
  },

  setBilling: (items: Record<string, number>) => {
    set((state) => {
      const nextUnitPrices = Object.fromEntries(
        Object.entries(state.itemUnitPrices).filter(([name]) => name in items),
      );
      const nextLineTotals = Object.fromEntries(
        Object.entries(items).map(([name, qty]) => [
          name,
          nextUnitPrices[name] != null ? (nextUnitPrices[name] as number) * qty : 0,
        ]),
      );
      const nextTotalAmount = Object.values(nextLineTotals).reduce((sum, value) => sum + value, 0);

      return {
        billingItems: items,
        itemUnitPrices: nextUnitPrices,
        itemLineTotals: nextLineTotals,
        unpricedItems: Object.entries(items)
          .filter(([name]) => nextUnitPrices[name] == null)
          .map(([name]) => name),
        totalCount: Object.values(items).reduce((a, b) => a + b, 0),
        totalAmount: nextTotalAmount,
      };
    });
  },

  setBillingState: (state: api.BillingState) => {
    set({
      billingItems: state.billing_items,
      itemScores: state.item_scores,
      itemUnitPrices: state.item_unit_prices,
      itemLineTotals: state.item_line_totals,
      totalCount: state.total_count,
      totalAmount: state.total_amount,
      currency: state.currency || "KRW",
      unpricedItems: state.unpriced_items || [],
    });
  },

  resetSession: () => {
    set({
      sessionId: null,
      billingItems: {},
      itemScores: {},
      itemUnitPrices: {},
      itemLineTotals: {},
      totalCount: 0,
      totalAmount: 0,
      currency: "KRW",
      unpricedItems: [],
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
