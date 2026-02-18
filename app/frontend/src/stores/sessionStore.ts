import { create } from "zustand";
import * as api from "../api/client";

export interface DetectionBox {
  box: number[]; // [x1, y1, x2, y2] normalized coordinates (0-1)
  class: "product" | "hand";
  confidence: number;
  label?: string; // Product label (if matched via CLIP+DINO)
  score?: number; // Matching score (if matched via CLIP+DINO)
}

interface SessionStore {
  sessionId: string | null;
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

  createSession: () => Promise<string>;
  updateFromWsMessage: (data: WsMessage) => void;
  setBilling: (items: Record<string, number>) => void;
  resetSession: () => void;
}

export interface WsMessage {
  frame?: string; // Optional: only sent when STREAM_SEND_IMAGES=true
  billing_items: Record<string, number>;
  item_scores: Record<string, number>;
  last_label: string;
  last_score: number;
  last_status: string;
  total_count: number;
  roi_polygon?: number[][] | null; // Normalized ROI polygon coordinates
  detection_boxes?: DetectionBox[]; // YOLO detection results
  warp_enabled?: boolean;
  warp_points?: number[][] | null;
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
  roiPolygon: null,
  detectionBoxes: [],
  warpEnabled: false,
  warpPoints: null,

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
      detectionBoxes: data.detection_boxes ?? [],
      warpEnabled: data.warp_enabled ?? false,
      warpPoints: data.warp_points ?? null,
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
      detectionBoxes: [],
      warpEnabled: false,
      warpPoints: null,
    });
  },
}));
