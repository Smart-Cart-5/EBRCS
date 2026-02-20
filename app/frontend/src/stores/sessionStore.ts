import { create } from "zustand";
import * as api from "../api/client";

export interface DetectionBox {
  box: number[]; // [x1, y1, x2, y2] normalized coordinates (0-1)
  class: "product" | "hand";
  confidence: number;
  label?: string; // Product label (if matched via CLIP+DINO)
  score?: number; // Matching score (if matched via CLIP+DINO)
}

export interface TopKCandidate {
  label: string;
  score?: number;
  raw_score?: number;
  percent_score?: number;
  crop_w?: number;
  crop_h?: number;
  box_area_ratio?: number;
}

interface SessionStore {
  sessionId: string | null;
  wsType: string;
  phase: string;
  message: string | null;
  billingItems: Record<string, number>;
  itemScores: Record<string, number>;
  totalCount: number;
  lastLabel: string;
  lastScore: number;
  lastStatus: string;
  annotatedFrame: string | null; // base64 JPEG (optional in JSON-only mode)
  roiPolygon: number[][] | null; // Normalized ROI polygon coordinates [[x1,y1], [x2,y2], ...]
  detectionBoxes: DetectionBox[]; // YOLO detection results
  warpEnabled: boolean;
  warpPoints: number[][] | null;
  didSearch: boolean;
  skipReason: string;
  lastResultAgeMs: number | null;
  topkCandidates: TopKCandidate[];
  cartRoiConfirmed: boolean;
  cartRoiPreviewReady: boolean;
  cartRoiPendingPolygon: number[][] | null; // Normalized [0..1] coordinates
  cartRoiPendingRatio: number;
  cartRoiPolygonConfirmed: number[][] | null; // Normalized [0..1] coordinates
  confirmEnabled: boolean;
  retryEnabled: boolean;
  cartRoiAutoEnabled: boolean | null;
  checkoutStartMode: "auto_roi" | "no_roi" | null;
  cartRoiAvailable: boolean;
  cartRoiUnavailableReason: string | null;
  lastRoiError: string | null;
  cartRoiInvalidReason: string | null;

  createSession: () => Promise<string>;
  updateFromWsMessage: (data: WsMessage) => void;
  setPhaseState: (phase: string, confirmed: boolean) => void;
  setBilling: (items: Record<string, number>) => void;
  resetSession: () => void;
}

export interface WsMessage {
  type: string;
  session_id: string;
  phase: string;
  message?: string | null;
  frame?: string; // Optional: only sent when STREAM_SEND_IMAGES=true
  billing_items: Record<string, number>;
  item_scores: Record<string, number>;
  last_label: string;
  last_score: number;
  last_status: string;
  total_count: number;
  roi_polygon?: number[][] | null; // Normalized ROI polygon coordinates
  detection_boxes?: DetectionBox[]; // YOLO detection results
  did_search?: boolean;
  skip_reason?: string;
  last_result_age_ms?: number | null;
  topk_candidates?: TopKCandidate[];
  warp_enabled?: boolean;
  warp_points?: number[][] | null;
  cart_roi_confirmed?: boolean;
  cart_roi_preview_ready?: boolean;
  cart_roi_pending_polygon?: number[][] | null; // Normalized [0..1]
  cart_roi_pending_ratio?: number;
  cart_roi_polygon_confirmed?: number[][] | null; // Normalized [0..1]
  confirm_enabled?: boolean;
  retry_enabled?: boolean;
  cart_roi_auto_enabled?: boolean | null;
  checkout_start_mode?: "auto_roi" | "no_roi" | null;
  cart_roi_available?: boolean;
  cart_roi_unavailable_reason?: string | null;
  last_roi_error?: string | null;
  cart_roi_invalid_reason?: string | null;
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  sessionId: null,
  wsType: "checkout_state",
  phase: "IDLE",
  message: null,
  billingItems: {},
  itemScores: {},
  totalCount: 0,
  lastLabel: "",
  lastScore: 0,
  lastStatus: "",
  annotatedFrame: null,
  roiPolygon: null,
  detectionBoxes: [],
  warpEnabled: false,
  warpPoints: null,
  didSearch: false,
  skipReason: "init",
  lastResultAgeMs: null,
  topkCandidates: [],
  cartRoiConfirmed: false,
  cartRoiPreviewReady: false,
  cartRoiPendingPolygon: null,
  cartRoiPendingRatio: 0,
  cartRoiPolygonConfirmed: null,
  confirmEnabled: false,
  retryEnabled: false,
  cartRoiAutoEnabled: null,
  checkoutStartMode: null,
  cartRoiAvailable: false,
  cartRoiUnavailableReason: null,
  lastRoiError: null,
  cartRoiInvalidReason: null,

  createSession: async () => {
    const { session_id } = await api.createSession();
    set({
      sessionId: session_id,
      billingItems: {},
      itemScores: {},
      totalCount: 0,
      phase: "IDLE",
      cartRoiConfirmed: false,
      cartRoiPreviewReady: false,
      cartRoiPendingPolygon: null,
      cartRoiPendingRatio: 0,
      confirmEnabled: false,
      retryEnabled: false,
    });
    return session_id;
  },

  updateFromWsMessage: (data: WsMessage) => {
    set({
      sessionId: data.session_id ?? get().sessionId,
      wsType: data.type ?? "checkout_state",
      phase: data.phase ?? "IDLE",
      message: data.message ?? null,
      annotatedFrame: data.frame ?? null,
      billingItems: data.billing_items,
      itemScores: data.item_scores,
      lastLabel: data.last_label,
      lastScore: data.last_score,
      lastStatus: data.last_status,
      totalCount: data.total_count,
      roiPolygon: data.roi_polygon ?? null,
      detectionBoxes: data.detection_boxes ?? [],
      didSearch: data.did_search ?? false,
      skipReason: data.skip_reason ?? "unknown",
      lastResultAgeMs: data.last_result_age_ms ?? null,
      topkCandidates: data.topk_candidates ?? [],
      warpEnabled: data.warp_enabled ?? false,
      warpPoints: data.warp_points ?? null,
      cartRoiConfirmed: data.cart_roi_confirmed ?? false,
      cartRoiPreviewReady: data.cart_roi_preview_ready ?? false,
      cartRoiPendingPolygon: data.cart_roi_pending_polygon ?? null,
      cartRoiPendingRatio: data.cart_roi_pending_ratio ?? 0,
      cartRoiPolygonConfirmed: data.cart_roi_polygon_confirmed ?? null,
      confirmEnabled: data.confirm_enabled ?? false,
      retryEnabled: data.retry_enabled ?? false,
      cartRoiAutoEnabled: data.cart_roi_auto_enabled ?? null,
      checkoutStartMode: data.checkout_start_mode ?? null,
      cartRoiAvailable: data.cart_roi_available ?? false,
      cartRoiUnavailableReason: data.cart_roi_unavailable_reason ?? null,
      lastRoiError: data.last_roi_error ?? null,
      cartRoiInvalidReason: data.cart_roi_invalid_reason ?? null,
    });
  },

  setPhaseState: (phase: string, confirmed: boolean) => {
    set({ phase, cartRoiConfirmed: confirmed });
  },

  setBilling: (items: Record<string, number>) => {
    set({ billingItems: items, totalCount: Object.values(items).reduce((a, b) => a + b, 0) });
  },

  resetSession: () => {
    set({
      sessionId: null,
      wsType: "checkout_state",
      phase: "IDLE",
      message: null,
      billingItems: {},
      itemScores: {},
      totalCount: 0,
      lastLabel: "",
      lastScore: 0,
      lastStatus: "",
      annotatedFrame: null,
      roiPolygon: null,
      detectionBoxes: [],
      warpEnabled: false,
      warpPoints: null,
      didSearch: false,
      skipReason: "init",
      lastResultAgeMs: null,
      topkCandidates: [],
      cartRoiConfirmed: false,
      cartRoiPreviewReady: false,
      cartRoiPendingPolygon: null,
      cartRoiPendingRatio: 0,
      cartRoiPolygonConfirmed: null,
      confirmEnabled: false,
      retryEnabled: false,
      cartRoiAutoEnabled: null,
      checkoutStartMode: null,
      cartRoiAvailable: false,
      cartRoiUnavailableReason: null,
      lastRoiError: null,
      cartRoiInvalidReason: null,
    });
  },
}));
