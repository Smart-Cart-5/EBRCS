import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import toast, { Toaster } from "react-hot-toast";
import { useSessionStore, type WsMessage } from "../stores/sessionStore";
import { wsCheckoutUrl, uploadVideo, videoStatusUrl, setROI, getHealth, getBilling } from "../api/client";
import BillingPanel from "../components/BillingPanel";
import StatusMetrics from "../components/StatusMetrics";
import ProductDrawer from "../components/ProductDrawer";

type Mode = "camera" | "upload";
const captureIntervalRaw = Number(import.meta.env.VITE_CAPTURE_INTERVAL_MS || 80);
const CAPTURE_INTERVAL_MS = Number.isFinite(captureIntervalRaw)
  ? Math.max(33, captureIntervalRaw)
  : 80;
const FULLSCREEN_ROI = [
  [0.0, 0.0], // Top-left
  [1.0, 0.0], // Top-right
  [1.0, 1.0], // Bottom-right
  [0.0, 1.0], // Bottom-left
];

function isSessionNotFoundError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  return msg.includes("404") || msg.includes("session not found");
}

const GUIDE_STEPS = [
  {
    icon: "📦",
    title: "카트를 영역에 맞춰주세요",
    desc: "카메라 앞에 카트나 바구니를 두고\n주황색 박스 안에 카트 입구가 들어오도록\n위치를 조정해주세요.",
  },
  {
    icon: "🛍️",
    title: "아래로 내리면 담기, 위로 올리면 빼기",
    desc: "상품을 위에서 아래로 통과시키면 🟢 장바구니에 담깁니다.\n반대로 아래에서 위로 올리면 🔴 장바구니에서 제거됩니다.",
  },
  {
    icon: "🔄",
    title: "잘못 담았다면 반대로 통과시키세요",
    desc: "실수로 담은 경우 상품을 아래쪽 방향으로\n다시 통과시키면 자동으로 제거됩니다.",
  },
];

export default function CheckoutPage() {
  const [mode] = useState<Mode>("camera");
  const [connected, setConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelLoadMsg, setModelLoadMsg] = useState("AI 모델 로딩 중...");
  const [guideOpen, setGuideOpen] = useState(false);
  const [guideStep, setGuideStep] = useState(0);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [cameraSettingsOpen, setCameraSettingsOpen] = useState(false);
  const [zoomSupported, setZoomSupported] = useState(false);
  const [zoomValue, setZoomValue] = useState(1);
  const [zoomMin, setZoomMin] = useState(1);
  const [zoomMax, setZoomMax] = useState(1);
  const [zoomStep, setZoomStep] = useState(0.1);
  const [cameraHint, setCameraHint] = useState("");

  const {
    sessionId,
    createSession,
    updateFromWsMessage,
    setBilling,
    setBillingState,
    billingItems,
    itemScores,
    itemUnitPrices,
    itemLineTotals,
    totalCount,
    totalAmount,
    currency,
    unpricedItems,
    lastLabel,
    lastScore,
    lastStatus,
    countEvent,
    currentTrackId,
  } = useSessionStore();

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null); // For capturing frames to send to backend
  const displayCanvasRef = useRef<HTMLCanvasElement>(null); // For rendering at 60 FPS
  const streamRef = useRef<MediaStream | null>(null);
  const videoTrackRef = useRef<MediaStreamTrack | null>(null);
  const captureAnimRef = useRef<number>(0); // For capture/send loop
  const renderAnimRef = useRef<number>(0); // For render loop (60 FPS)
  const countFlashRef = useRef<number>(0); // count 이벤트 시 화면 플래시

  // Refs mirroring store values for use inside renderFrame closure.
  // renderFrame is created once and runs continuously, so it captures stale
  // closure values if we read store state directly. Refs are always current.
  const lastLabelRef = useRef<string>(lastLabel);
  const lastScoreRef = useRef<number>(lastScore);
  const lastStatusRef = useRef<string>(lastStatus);
  const currentTrackIdRef = useRef<string | null>(currentTrackId);

  // Keep refs in sync with store so renderFrame closure always reads latest values
  useEffect(() => {
    lastLabelRef.current = lastLabel;
    lastScoreRef.current = lastScore;
    lastStatusRef.current = lastStatus;
    currentTrackIdRef.current = currentTrackId;
  }, [lastLabel, lastScore, lastStatus, currentTrackId]);

  const selectPreferredCamera = useCallback((devices: MediaDeviceInfo[]) => {
    if (devices.length === 0) return null;

    const rearKeywords = /(back|rear|environment|후면|광각|ultra|wide)/i;
    const rear = devices.find((d) => rearKeywords.test(d.label));
    return rear?.deviceId ?? devices[0].deviceId;
  }, []);

  const refreshCameraDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter((d) => d.kind === "videoinput");
    setCameraDevices(videos);
    setSelectedCameraId((prev) => {
      if (prev && videos.some((d) => d.deviceId === prev)) {
        return prev;
      }
      return selectPreferredCamera(videos);
    });
  }, [selectPreferredCamera]);

  const initZoomCapabilities = useCallback(async (track: MediaStreamTrack) => {
    videoTrackRef.current = track;

    const capabilities = ((track.getCapabilities?.() as {
      zoom?: { min: number; max: number; step?: number };
    } | undefined) ?? {});
    const settings = (track.getSettings?.() as { zoom?: number } | undefined) ?? {};
    const zoomCap = capabilities.zoom;

    if (!zoomCap || typeof zoomCap.min !== "number" || typeof zoomCap.max !== "number") {
      setZoomSupported(false);
      setCameraHint("현재 기기/브라우저는 웹 줌 제어를 지원하지 않습니다.");
      return;
    }

    const min = Number(zoomCap.min);
    const max = Number(zoomCap.max);
    const step = Math.max(0.01, Number(zoomCap.step ?? 0.1));
    const current = typeof settings.zoom === "number"
      ? settings.zoom
      : min;

    setZoomSupported(true);
    setZoomMin(min);
    setZoomMax(max);
    setZoomStep(step);
    setZoomValue(current);
    setCameraHint(max > min ? "줌 슬라이더로 화각을 조절할 수 있습니다." : "현재 렌즈는 고정 화각입니다.");

    if (max > min) {
      try {
        await track.applyConstraints({
          advanced: [{ zoom: current } as unknown as MediaTrackConstraintSet],
        });
      } catch (error) {
        console.warn("Failed to apply initial zoom:", error);
      }
    }
  }, []);

  const applyZoom = useCallback(async (nextZoom: number) => {
    setZoomValue(nextZoom);
    const track = videoTrackRef.current;
    if (!track || !zoomSupported) return;

    try {
      await track.applyConstraints({
        advanced: [{ zoom: nextZoom } as unknown as MediaTrackConstraintSet],
      });
    } catch (error) {
      console.warn("Failed to apply zoom:", error);
    }
  }, [zoomSupported]);

  const ensureSessionAndROI = useCallback(async (): Promise<string> => {
    let activeSessionId = sessionId;
    if (!activeSessionId) {
      activeSessionId = await createSession();
    }

    try {
      await setROI(activeSessionId, FULLSCREEN_ROI);
      return activeSessionId;
    } catch (error) {
      if (isSessionNotFoundError(error)) {
        console.warn("Session not found. Recreating session...");
        activeSessionId = await createSession();
        await setROI(activeSessionId, FULLSCREEN_ROI);
        return activeSessionId;
      }
      throw error;
    }
  }, [sessionId, createSession]);

  // Poll backend health until models are loaded
  useEffect(() => {
    if (modelReady) return;
    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        try {
          const health = await getHealth();
          if (health.index_vectors && Number(health.index_vectors) > 0) {
            setModelReady(true);
            setModelLoadMsg("");
            return;
          }
          setModelLoadMsg("AI 모델 로딩 중...");
        } catch {
          setModelLoadMsg("서버 연결 대기 중...");
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };
    poll();
    return () => { cancelled = true; };
  }, [modelReady]);

  useEffect(() => {
    refreshCameraDevices().catch((error) => {
      console.warn("Failed to enumerate cameras:", error);
    });

    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.addEventListener) return;

    const onDeviceChange = () => {
      refreshCameraDevices().catch((error) => {
        console.warn("Failed to refresh cameras after device change:", error);
      });
    };

    mediaDevices.addEventListener("devicechange", onDeviceChange);
    return () => {
      mediaDevices.removeEventListener("devicechange", onDeviceChange);
    };
  }, [refreshCameraDevices]);

  // Ensure session exists and setup virtual ROI for entry-event mode.
  // Gated on modelReady to avoid ECONNREFUSED on startup (frontend starts
  // before backend is ready, and a failed createSession leaves sessionId=null).
  useEffect(() => {
    if (!modelReady) return;
    ensureSessionAndROI().catch((err) => {
      console.warn("Failed to ensure session/ROI:", err);
    });
  }, [modelReady, ensureSessionAndROI]);

  // Show toast + flash when count event fires from backend
  useEffect(() => {
    if (!countEvent) return;

    const isRemove = countEvent.action === "remove";

    // Trigger canvas border flash (green=add, red=remove)
    countFlashRef.current = isRemove ? -20 : 20; // 음수 = 제거 플래시

    const qtyText = isRemove
      ? countEvent.quantity === 0
        ? '완전히 제거됨'
        : `제거됨 (${countEvent.quantity}개 남음)`
      : `${countEvent.quantity}개 담김`;

    if (isRemove) {
      toast(`🔴 ${countEvent.product} ${qtyText}`, {
        duration: 2500,
        position: "top-center",
        style: {
          background: "#ef4444",
          color: "#fff",
          fontWeight: "600",
          fontSize: "16px",
        },
      });
    } else {
      toast.success(`🟢 ${countEvent.product} ${qtyText}`, {
        icon: "🛒",
        duration: 2500,
        position: "top-center",
        style: {
          background: "#22c55e",
          color: "#fff",
          fontWeight: "600",
          fontSize: "16px",
        },
      });
    }
  }, [countEvent]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      videoTrackRef.current = null;
      cancelAnimationFrame(captureAnimRef.current);
      cancelAnimationFrame(renderAnimRef.current);
    };
  }, []);

  const billingSignature = useMemo(
    () =>
      JSON.stringify(
        Object.entries(billingItems).sort(([a], [b]) => a.localeCompare(b)),
      ),
    [billingItems],
  );

  const hasMultipleCameras = cameraDevices.length > 1;
  const canShowZoomSlider = zoomSupported && zoomMax - zoomMin > 0.001;

  // Fetch latest pricing only when cart composition changes.
  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;
    const fetchBillingState = async () => {
      try {
        const state = await getBilling(sessionId);
        if (!cancelled) {
          setBillingState(state);
        }
      } catch (error) {
        console.warn("Failed to refresh billing prices:", error);
      }
    };

    fetchBillingState();
    return () => {
      cancelled = true;
    };
  }, [sessionId, billingSignature, setBillingState]);

  // --- Camera mode ---
  const startCamera = useCallback(async (forcedDeviceId?: string) => {
    try {
      // Start loading state
      setIsLoading(true);
      setLoadingMessage("세션 준비 중...");

      // Ensure session exists on backend before opening WebSocket
      const activeSessionId = await ensureSessionAndROI();

      setLoadingMessage("카메라 권한 요청 중...");

      // Request camera and WebSocket in parallel for better UX
      console.log("📷 Requesting camera access...");

      // Start both operations in parallel
      const activeCameraId = forcedDeviceId ?? selectedCameraId;
      const preferredConstraints: MediaTrackConstraints = activeCameraId
        ? {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            aspectRatio: { ideal: 16 / 9 },
            frameRate: { ideal: 30, max: 60 },
            deviceId: { exact: activeCameraId },
          }
        : {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            aspectRatio: { ideal: 16 / 9 },
            frameRate: { ideal: 30, max: 60 },
            facingMode: { ideal: "environment" },
          };

      const fallbackConstraints: MediaTrackConstraints = {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        aspectRatio: { ideal: 16 / 9 },
        frameRate: { ideal: 30, max: 60 },
        facingMode: { ideal: "environment" },
      };

      const cameraPromise = navigator.mediaDevices.getUserMedia({
        video: preferredConstraints,
      }).catch(async (error) => {
        if (!activeCameraId) {
          throw error;
        }
        console.warn("Selected camera unavailable, fallback to environment lens:", error);
        return navigator.mediaDevices.getUserMedia({ video: fallbackConstraints });
      });

      const ws = new WebSocket(wsCheckoutUrl(activeSessionId));
      wsRef.current = ws;

      const wsPromise = new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          cleanup();
          reject(new Error("WebSocket connection timeout"));
        }, 10000);

        const cleanup = () => {
          clearTimeout(timeout);
          ws.removeEventListener("open", onOpen);
          ws.removeEventListener("error", onError);
          ws.removeEventListener("close", onCloseBeforeOpen);
        };

        const onOpen = () => {
          cleanup();
          console.log("✅ WebSocket connected");
          resolve();
        };

        const onError = (err: Event) => {
          cleanup();
          console.error("❌ WebSocket error:", err);
          reject(new Error("WebSocket connection failed"));
        };

        const onCloseBeforeOpen = (event: CloseEvent) => {
          cleanup();
          reject(
            new Error(
              `WebSocket closed before connected (code=${event.code})`,
            ),
          );
        };

        ws.addEventListener("open", onOpen);
        ws.addEventListener("error", onError);
        ws.addEventListener("close", onCloseBeforeOpen);
      });

      // Setup WebSocket message handlers
      ws.onclose = () => {
        console.log("❌ WebSocket closed");
        setConnected(false);
      };
      ws.onmessage = (e) => {
        const data: WsMessage = JSON.parse(e.data);
        console.log("📥 Received state from backend", {
          has_frame: !!data.frame,
          has_roi: !!data.roi_polygon,
          total_count: data.total_count,
          last_label: data.last_label,
        });
        updateFromWsMessage(data);
      };

      // Wait for both camera and WebSocket to be ready
      setLoadingMessage("카메라 및 서버 연결 중...");
      const [stream] = await Promise.all([cameraPromise, wsPromise]);

      console.log("✅ Camera access granted");
      streamRef.current = stream;
      const videoTrack = stream.getVideoTracks()[0] ?? null;
      if (videoTrack) {
        await initZoomCapabilities(videoTrack);
      } else {
        setZoomSupported(false);
        setCameraHint("카메라 트랙을 찾지 못했습니다.");
      }
      await refreshCameraDevices();

      const video = videoRef.current!;
      video.srcObject = stream;

      // Wait for video to be ready
      setLoadingMessage("카메라 준비 중...");
      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => {
          console.log("📹 Video metadata loaded");
          resolve();
        };
      });

      await video.play();
      console.log("▶️ Video playing");

      // Wait for both video dimensions AND canvas refs to be ready
      // Critical for mobile browsers where DOM mounting can be slower
      setLoadingMessage("화면 초기화 중...");
      await new Promise<void>((resolve, reject) => {
        let attempts = 0;
        const maxAttempts = 100; // ~1.6 seconds max wait

        const checkReady = () => {
          attempts++;

          // Check if canvas refs are available
          if (!captureCanvasRef.current || !displayCanvasRef.current) {
            console.log(`⏳ Waiting for canvas refs... (attempt ${attempts})`);
            if (attempts < maxAttempts) {
              requestAnimationFrame(checkReady);
            } else {
              reject(new Error("Canvas refs not available after timeout"));
            }
            return;
          }

          // Check if video dimensions are available
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            console.log(`✅ Ready: video=${video.videoWidth}x${video.videoHeight}, canvases=mounted`);
            resolve();
          } else {
            console.log(`⏳ Waiting for video dimensions... (attempt ${attempts})`);
            if (attempts < maxAttempts) {
              requestAnimationFrame(checkReady);
            } else {
              reject(new Error("Video dimensions not available after timeout"));
            }
          }
        };

        checkReady();
      });

      // Setup canvases - now guaranteed to exist
      const captureCanvas = captureCanvasRef.current;
      const displayCanvas = displayCanvasRef.current;

      if (!captureCanvas || !displayCanvas) {
        throw new Error("Canvas elements not found after ready check");
      }

      // Use actual video dimensions (guaranteed to be > 0 now)
      const canvasWidth = video.videoWidth;
      const canvasHeight = video.videoHeight;

      console.log(`🎨 Setting canvas size: ${canvasWidth}x${canvasHeight}`);

      captureCanvas.width = canvasWidth;
      captureCanvas.height = canvasHeight;
      displayCanvas.width = canvasWidth;
      displayCanvas.height = canvasHeight;

      const captureCtx = captureCanvas.getContext("2d");
      const displayCtx = displayCanvas.getContext("2d");

      if (!captureCtx || !displayCtx) {
        throw new Error("Failed to get 2D context from canvas");
      }

      // 60 FPS rendering loop (local camera + overlay)
      const renderFrame = () => {
        renderAnimRef.current = requestAnimationFrame(renderFrame);

        // Draw video to display canvas
        displayCtx.drawImage(video, 0, 0, displayCanvas.width, displayCanvas.height);

        // ROI = 전체 화면이므로 테두리/오버레이 대신
        // 상단/하단에 방향 가이드 텍스트만 표시
        const W = displayCanvas.width, H = displayCanvas.height;
        const midX = W / 2;
        displayCtx.textAlign = 'center';
        displayCtx.font = 'bold 16px sans-serif';
        displayCtx.lineWidth = 3;

        // 상단 라벨 - 담기 (상단 진입 = 담기)
        displayCtx.strokeStyle = 'rgba(0,0,0,0.7)';
        displayCtx.strokeText('↓ 담기', midX, 28);
        displayCtx.fillStyle = 'rgba(34, 197, 94, 1)';
        displayCtx.fillText('↓ 담기', midX, 28);

        // 하단 라벨 - 빼기 (하단 진입 = 빼기)
        displayCtx.strokeStyle = 'rgba(0,0,0,0.7)';
        displayCtx.strokeText('↑ 빼기', midX, H - 12);
        displayCtx.fillStyle = 'rgba(239, 68, 68, 1)';
        displayCtx.fillText('↑ 빼기', midX, H - 12);

        displayCtx.textAlign = 'left'; // 기본값 복원

        // Count flash effect (green=add, red=remove)
        if (countFlashRef.current !== 0) {
          const isRemoveFlash = countFlashRef.current < 0;
          const remaining = Math.abs(countFlashRef.current);
          const alpha = Math.min(1, remaining / 15);
          displayCtx.strokeStyle = isRemoveFlash
            ? `rgba(239, 68, 68, ${alpha})`   // 빨강 = 제거
            : `rgba(34, 197, 94, ${alpha})`;   // 초록 = 담기
          displayCtx.lineWidth = 6;
          displayCtx.strokeRect(0, 0, displayCanvas.width, displayCanvas.height);
          countFlashRef.current += isRemoveFlash ? 1 : -1; // 0을 향해 수렴
        }

        // Draw status text overlay (read from refs to get latest values)
        const currentLabel = lastLabelRef.current;
        const currentScore = lastScoreRef.current;
        const currentStatus = lastStatusRef.current;
        const currentTrack = currentTrackIdRef.current;
        if (currentLabel && currentLabel !== "-") {
          const trackStr = currentTrack ? ` [ID:${currentTrack}]` : "";
          const text = `${currentLabel} (${currentScore.toFixed(3)})${trackStr}`;
          displayCtx.font = 'bold 18px sans-serif';

          displayCtx.strokeStyle = 'black';
          displayCtx.lineWidth = 4;
          displayCtx.strokeText(text, 10, 30);

          displayCtx.fillStyle = 'white';
          displayCtx.fillText(text, 10, 30);
        }

        // Draw status indicator
        if (currentStatus) {
          const statusText = currentTrack
            ? `${currentStatus} | Track:${currentTrack}`
            : currentStatus;
          const textWidth = displayCtx.measureText(statusText).width + 20;
          displayCtx.font = '14px sans-serif';
          displayCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
          displayCtx.fillRect(10, displayCanvas.height - 30, textWidth, 20);
          displayCtx.fillStyle = 'white';
          displayCtx.fillText(statusText, 15, displayCanvas.height - 15);
        }
      };

      // 10-15 FPS capture and send loop
      let lastSend = 0;
      let frameCount = 0;
      const sendFrame = () => {
        captureAnimRef.current = requestAnimationFrame(sendFrame);
        const now = performance.now();

        // Throttle capture loop (env-configurable)
        if (now - lastSend < CAPTURE_INTERVAL_MS) return;

        if (ws.readyState !== WebSocket.OPEN) {
          if (frameCount === 0) {
            console.log("⏳ Waiting for WebSocket to be ready...");
          }
          return;
        }

        captureCtx.drawImage(video, 0, 0);
        captureCanvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              ws.send(blob);
              frameCount++;
              if (frameCount === 1) {
                console.log("📤 First frame sent to backend");
              }
              lastSend = performance.now();
            }
          },
          "image/jpeg",
          0.7,
        );
      };

      console.log("🎬 Starting 60 FPS render loop and 12.5 FPS capture loop");
      renderFrame();
      sendFrame();

      // All ready - show camera feed + guide modal
      setLoadingMessage("완료!");
      setTimeout(() => {
        setConnected(true);
        setIsLoading(false);
        setLoadingMessage("");
        setGuideOpen(true);
        setGuideStep(0);
      }, 300);
    } catch (error) {
      console.error("❌ Camera error:", error);
      setIsLoading(false);
      setLoadingMessage("");
      alert(`Failed to start camera: ${error instanceof Error ? error.message : String(error)}`);
      stopCamera();
    }
  // Store values (roiPolygon, lastLabel, etc.) removed from deps — they're
  // accessed via refs inside renderFrame, so startCamera doesn't need to
  // rebuild the render loop every time the store updates.
  }, [
    ensureSessionAndROI,
    initZoomCapabilities,
    refreshCameraDevices,
    selectedCameraId,
    updateFromWsMessage,
  ]);

  const stopCamera = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    videoTrackRef.current = null;
    cancelAnimationFrame(captureAnimRef.current);
    cancelAnimationFrame(renderAnimRef.current);
    setConnected(false);
    setIsLoading(false);
    setLoadingMessage("");
    setGuideOpen(false);
    setGuideStep(0);
    setCameraSettingsOpen(false);
    setZoomSupported(false);
  }, []);

  const handleCameraDeviceChange = useCallback(async (deviceId: string) => {
    setSelectedCameraId(deviceId);
    if (!connected) return;

    stopCamera();
    await new Promise((resolve) => setTimeout(resolve, 120));
    void startCamera(deviceId);
  }, [connected, startCamera, stopCamera]);

  // --- Upload mode ---
  const handleUpload = useCallback(
    async (file: File) => {
      if (!sessionId) return;
      setUploadProgress(0);

      const { task_id } = await uploadVideo(sessionId, file);

      // Listen SSE
      const evtSource = new EventSource(videoStatusUrl(sessionId, task_id));
      evtSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        setUploadProgress(Math.round(data.progress * 100));
        setBilling(data.billing_items);

        if (data.done) {
          evtSource.close();
          setUploadProgress(null);
        }
        if (data.error) {
          evtSource.close();
          setUploadProgress(null);
          alert(`Video error: ${data.error}`);
        }
      };
      evtSource.onerror = () => {
        evtSource.close();
        setUploadProgress(null);
      };
    },
    [sessionId, setBilling],
  );

  // Full-screen loading overlay while models load
  if (!modelReady) {
    return (
      <div className="h-full flex items-center justify-center bg-[var(--color-bg)]">
        <div className="text-center space-y-6 p-8">
          {/* Animated loading indicator */}
          <div className="relative w-20 h-20 mx-auto">
            <div className="absolute inset-0 border-4 border-[var(--color-surface-light)] rounded-full" />
            <div className="absolute inset-0 border-4 border-transparent border-t-[var(--color-primary)] rounded-full animate-spin" />
          </div>

          <div className="space-y-2">
            <h2 className="text-xl font-bold text-[var(--color-text)]">
              시스템 준비 중
            </h2>
            <p className="text-sm text-[var(--color-text-muted)]">
              {modelLoadMsg}
            </p>
          </div>

          {/* Progress dots animation */}
          <div className="flex justify-center gap-1.5">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-[var(--color-primary)] animate-pulse"
                style={{ animationDelay: `${i * 0.3}s` }}
              />
            ))}
          </div>

          <p className="text-xs text-[var(--color-text-muted)]">
            DINOv2 + CLIP 모델과 FAISS 인덱스를 불러오고 있습니다
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Toast Container */}
      <Toaster />

      {/* Main Container */}
      <div className="h-full p-0 lg:p-6 flex flex-col lg:flex-row gap-0 lg:gap-6">
        {/* Camera Feed */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="bg-[#1e293b] rounded-none lg:rounded-2xl overflow-hidden relative h-[calc(100vh-64px)] lg:h-full flex items-center justify-center">
          {/* Hidden video + canvas for capture */}
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={captureCanvasRef} className="hidden" />

          {/* Display canvas - always in DOM, visibility controlled by CSS */}
          <canvas
            ref={displayCanvasRef}
            className={`max-w-full max-h-full object-contain ${
              mode === "camera" && connected ? "" : "hidden"
            }`}
          />

          {mode === "camera" ? (
            connected ? (
              <>
                {/* Live Badge */}
                <div className="absolute top-3 left-3 md:top-4 md:left-4 bg-[var(--color-success)] text-white px-2 py-1 md:px-3 md:py-1 rounded-full text-xs md:text-sm font-semibold flex items-center gap-1.5 md:gap-2">
                  <span className="w-1.5 h-1.5 md:w-2 md:h-2 bg-white rounded-full animate-pulse" />
                  Live
                </div>
                <button
                  onClick={() => setCameraSettingsOpen((prev) => !prev)}
                  className="absolute top-3 right-3 md:top-4 md:right-4 px-3 py-1.5 bg-black/50 hover:bg-black/65 text-white rounded-lg text-xs md:text-sm font-semibold transition-colors"
                >
                  카메라 설정
                </button>
                {cameraSettingsOpen && (
                  <div className="absolute top-14 right-3 md:top-16 md:right-4 z-20 w-72 bg-white/95 backdrop-blur rounded-xl border border-gray-200 shadow-xl p-3 space-y-3">
                    <div>
                      <p className="text-xs font-semibold text-gray-700 mb-1">렌즈 선택</p>
                      <select
                        value={selectedCameraId ?? ""}
                        onChange={(e) => {
                          const next = e.target.value;
                          if (next) {
                            void handleCameraDeviceChange(next);
                          }
                        }}
                        className="w-full border border-gray-300 rounded-lg px-2 py-1.5 text-sm"
                        disabled={!hasMultipleCameras}
                      >
                        {cameraDevices.length === 0 && (
                          <option value="">카메라를 찾는 중...</option>
                        )}
                        {cameraDevices.map((device, idx) => (
                          <option key={device.deviceId} value={device.deviceId}>
                            {device.label || `카메라 ${idx + 1}`}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <p className="text-xs font-semibold text-gray-700">디지털 줌</p>
                        <p className="text-xs text-gray-500">{zoomValue.toFixed(2)}x</p>
                      </div>
                      <input
                        type="range"
                        min={zoomMin}
                        max={zoomMax}
                        step={zoomStep}
                        value={zoomValue}
                        disabled={!canShowZoomSlider}
                        onChange={(e) => void applyZoom(Number(e.target.value))}
                        className="w-full accent-[var(--color-primary)] disabled:opacity-40"
                      />
                      <p className="text-[11px] text-gray-500 mt-1">
                        {cameraHint || "렌즈마다 지원 가능한 줌 범위가 다릅니다."}
                      </p>
                    </div>
                  </div>
                )}
                {/* Stop Button */}
                <button
                  onClick={stopCamera}
                  className="absolute bottom-3 right-3 md:bottom-4 md:right-4 px-4 py-1.5 md:px-6 md:py-2 bg-[var(--color-danger)] hover:bg-[var(--color-danger-hover)] text-white rounded-lg md:rounded-xl text-xs md:text-sm font-semibold transition-colors shadow-lg"
                >
                  정지
                </button>

                {/* 3-Step Guide Modal */}
                {guideOpen && (
                  <div className="absolute inset-0 z-20 flex items-end justify-center pb-16 md:pb-8">
                    <div className="bg-white rounded-2xl p-5 mx-4 w-full max-w-sm shadow-2xl">
                      <div className="text-center space-y-3">
                        <div className="text-3xl">{GUIDE_STEPS[guideStep].icon}</div>
                        <h3 className="font-bold text-gray-900 text-sm md:text-base">
                          {GUIDE_STEPS[guideStep].title}
                        </h3>
                        <p className="text-xs md:text-sm text-gray-500 whitespace-pre-line leading-relaxed">
                          {GUIDE_STEPS[guideStep].desc}
                        </p>
                        {/* Step dots */}
                        <div className="flex justify-center gap-2 py-1">
                          {GUIDE_STEPS.map((_, i) => (
                            <div
                              key={i}
                              className={`h-2 rounded-full transition-all duration-300 ${
                                i === guideStep ? "w-5 bg-orange-500" : "w-2 bg-gray-200"
                              }`}
                            />
                          ))}
                        </div>
                        <div className="flex gap-2 pt-1">
                          {guideStep > 0 && (
                            <button
                              onClick={() => setGuideStep(s => s - 1)}
                              className="flex-1 py-2.5 border border-gray-200 text-gray-600 rounded-xl text-sm font-medium"
                            >
                              이전
                            </button>
                          )}
                          <button
                            onClick={() => {
                              if (guideStep < GUIDE_STEPS.length - 1) {
                                setGuideStep(s => s + 1);
                              } else {
                                setGuideOpen(false);
                              }
                            }}
                            className="flex-1 py-2.5 bg-orange-500 hover:bg-orange-600 text-white rounded-xl text-sm font-semibold transition-colors"
                          >
                            {guideStep < GUIDE_STEPS.length - 1 ? "다음" : "시작하기"}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center">
                <span className="text-5xl md:text-6xl mb-3 md:mb-4 block">
                  {isLoading ? "⏳" : "📷"}
                </span>
                <p className="text-gray-400 mb-3 md:mb-4 text-sm md:text-base">
                  {isLoading ? loadingMessage : "카메라를 시작하려면 버튼을 클릭하세요"}
                </p>
                {isLoading && (
                  <div className="mb-4">
                    <div className="inline-block w-8 h-8 border-4 border-gray-600 border-t-[var(--color-success)] rounded-full animate-spin"></div>
                  </div>
                )}
                <button
                  onClick={() => void startCamera()}
                  disabled={isLoading}
                  className="px-5 py-2.5 md:px-6 md:py-3 bg-[var(--color-success)] hover:bg-[var(--color-success-hover)] text-white rounded-lg md:rounded-xl text-sm md:text-base font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg"
                >
                  {isLoading ? "준비 중..." : "카메라 시작"}
                </button>
                {cameraDevices.length > 0 && (
                  <div className="mt-4 max-w-xs mx-auto text-left bg-black/30 border border-white/20 rounded-xl p-3 space-y-2">
                    <p className="text-xs text-gray-200 font-semibold">렌즈 미리 선택</p>
                    <select
                      value={selectedCameraId ?? ""}
                      onChange={(e) => {
                        const next = e.target.value;
                        if (next) {
                          void handleCameraDeviceChange(next);
                        }
                      }}
                      className="w-full border border-gray-500 bg-black/20 text-gray-100 rounded-lg px-2 py-1.5 text-sm"
                    >
                      {cameraDevices.map((device, idx) => (
                        <option key={device.deviceId} value={device.deviceId} className="text-black">
                          {device.label || `카메라 ${idx + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )
          ) : uploadProgress !== null ? (
            <div className="text-center space-y-3">
              <div className="w-48 md:w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--color-primary)] transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <span className="text-gray-300 text-xs md:text-sm">
                처리 중: {uploadProgress}%
              </span>
            </div>
          ) : (
            <label className="cursor-pointer text-gray-400 hover:text-gray-200 transition-colors">
              <span className="text-5xl md:text-6xl mb-3 md:mb-4 block">📁</span>
              <span className="text-xs md:text-sm">영상 업로드</span>
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleUpload(f);
                }}
              />
            </label>
          )}
        </div>
      </div>

      {/* Status + Product List - Desktop Only */}
      <div className="hidden lg:flex w-full lg:w-[420px] flex-col gap-3 md:gap-4">
        {/* Status Metrics */}
        <StatusMetrics
          lastLabel={lastLabel}
          lastScore={lastScore}
          lastStatus={lastStatus}
          fps={undefined}
          trackId={currentTrackId}
        />

        {/* Product List */}
        <div className="flex-1 min-h-0">
          <BillingPanel
            billingItems={billingItems}
            itemScores={itemScores}
            itemUnitPrices={itemUnitPrices}
            itemLineTotals={itemLineTotals}
            totalCount={totalCount}
            totalAmount={totalAmount}
            currency={currency}
            unpricedItems={unpricedItems}
          />
        </div>
      </div>
    </div>

    {/* FAB (Floating Action Button) - Mobile Only */}
    <button
      onClick={() => setDrawerOpen(true)}
      className="lg:hidden fixed bottom-20 right-4 w-16 h-16 bg-[var(--color-primary)] text-white rounded-full shadow-lg flex items-center justify-center z-30 active:scale-95 transition-transform"
    >
      <div className="relative">
        <span className="text-2xl">🛒</span>
        {totalCount > 0 && (
          <span className="absolute -top-2 -right-2 w-6 h-6 bg-[var(--color-danger)] text-white text-xs font-bold rounded-full flex items-center justify-center">
            {totalCount}
          </span>
        )}
      </div>
    </button>

    {/* Product Drawer - Mobile Only */}
      <ProductDrawer
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        billingItems={billingItems}
        itemScores={itemScores}
        itemUnitPrices={itemUnitPrices}
        itemLineTotals={itemLineTotals}
        totalCount={totalCount}
        totalAmount={totalAmount}
        currency={currency}
        unpricedItems={unpricedItems}
      />
  </>
  );
}
