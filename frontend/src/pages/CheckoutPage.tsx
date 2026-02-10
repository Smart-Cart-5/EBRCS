import { useCallback, useEffect, useRef, useState } from "react";
import toast, { Toaster } from "react-hot-toast";
import { useSessionStore, type WsMessage } from "../stores/sessionStore";
import { wsCheckoutUrl, uploadVideo, videoStatusUrl } from "../api/client";
import BillingPanel from "../components/BillingPanel";
import StatusMetrics from "../components/StatusMetrics";
import ProductDrawer from "../components/ProductDrawer";

type Mode = "camera" | "upload";

export default function CheckoutPage() {
  const [mode, setMode] = useState<Mode>("camera");
  const [connected, setConnected] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const {
    sessionId,
    createSession,
    updateFromWsMessage,
    setBilling,
    billingItems,
    itemScores,
    totalCount,
    lastLabel,
    lastScore,
    lastStatus,
    annotatedFrame,
  } = useSessionStore();

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const prevBillingRef = useRef<Record<string, number>>({});

  // Ensure session exists
  useEffect(() => {
    if (!sessionId) {
      createSession();
    }
  }, [sessionId, createSession]);

  // Show toast when new product is detected
  useEffect(() => {
    const prevItems = prevBillingRef.current;
    const currentItems = billingItems;

    // Check for new products or increased quantities
    Object.keys(currentItems).forEach((productName) => {
      const prevQty = prevItems[productName] || 0;
      const currentQty = currentItems[productName] || 0;

      if (currentQty > prevQty) {
        const addedQty = currentQty - prevQty;
        toast.success(`${productName} ${addedQty}개 담김!`, {
          icon: "🛒",
          duration: 2000,
          position: "top-center",
          style: {
            background: "#22c55e",
            color: "#fff",
            fontWeight: "600",
          },
        });
      }
    });

    // Update prev ref
    prevBillingRef.current = { ...currentItems };
  }, [billingItems]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  // --- Camera mode ---
  const startCamera = useCallback(async () => {
    if (!sessionId) return;

    try {
      // Open WebSocket
      const ws = new WebSocket(wsCheckoutUrl(sessionId));
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("✅ WebSocket connected");
        setConnected(true);
      };
      ws.onclose = () => {
        console.log("❌ WebSocket closed");
        setConnected(false);
      };
      ws.onerror = (err) => {
        console.error("❌ WebSocket error:", err);
        alert("WebSocket connection failed. Check if backend is running.");
      };
      ws.onmessage = (e) => {
        const data: WsMessage = JSON.parse(e.data);
        console.log("📥 Received frame from backend", {
          has_frame: !!data.frame,
          total_count: data.total_count,
          last_label: data.last_label,
        });
        updateFromWsMessage(data);
      };

      // Open camera
      console.log("📷 Requesting camera access...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "environment" },
      });
      console.log("✅ Camera access granted");
      streamRef.current = stream;

    const video = videoRef.current!;
    video.srcObject = stream;

    // Wait for video to be ready
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => {
        console.log("📹 Video metadata loaded");
        resolve();
      };
    });

    await video.play();
    console.log("▶️ Video playing");

    const canvas = canvasRef.current!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d")!;
    console.log(`📐 Canvas size: ${canvas.width}x${canvas.height}`);

    let lastSend = 0;
    let frameCount = 0;
    const sendFrame = () => {
      animRef.current = requestAnimationFrame(sendFrame);
      const now = performance.now();
      if (now - lastSend < 200) return; // 5 FPS max
      if (ws.readyState !== WebSocket.OPEN) {
        if (frameCount === 0) {
          console.log("⏳ Waiting for WebSocket to be ready...");
        }
        return;
      }

      ctx.drawImage(video, 0, 0);
      canvas.toBlob(
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
    console.log("🎬 Starting frame capture loop");
    sendFrame();
    } catch (error) {
      console.error("❌ Camera error:", error);
      alert(`Failed to start camera: ${error instanceof Error ? error.message : String(error)}`);
      stopCamera();
    }
  }, [sessionId, updateFromWsMessage]);

  const stopCamera = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    cancelAnimationFrame(animRef.current);
    setConnected(false);
  }, []);

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
          <canvas ref={canvasRef} className="hidden" />

          {mode === "camera" ? (
            connected && annotatedFrame ? (
              <>
                <img
                  src={`data:image/jpeg;base64,${annotatedFrame}`}
                  alt="Camera feed"
                  className="max-w-full max-h-full object-contain"
                />
                {/* Live Badge */}
                <div className="absolute top-3 left-3 md:top-4 md:left-4 bg-[var(--color-success)] text-white px-2 py-1 md:px-3 md:py-1 rounded-full text-xs md:text-sm font-semibold flex items-center gap-1.5 md:gap-2">
                  <span className="w-1.5 h-1.5 md:w-2 md:h-2 bg-white rounded-full animate-pulse" />
                  Live
                </div>
                {/* Stop Button */}
                <button
                  onClick={stopCamera}
                  className="absolute bottom-3 right-3 md:bottom-4 md:right-4 px-4 py-1.5 md:px-6 md:py-2 bg-[var(--color-danger)] hover:bg-[var(--color-danger-hover)] text-white rounded-lg md:rounded-xl text-xs md:text-sm font-semibold transition-colors shadow-lg"
                >
                  정지
                </button>
              </>
            ) : (
              <div className="text-center">
                <span className="text-5xl md:text-6xl mb-3 md:mb-4 block">📷</span>
                <p className="text-gray-400 mb-3 md:mb-4 text-sm md:text-base">카메라 시작 중...</p>
                <button
                  onClick={startCamera}
                  disabled={connected}
                  className="px-5 py-2.5 md:px-6 md:py-3 bg-[var(--color-success)] hover:bg-[var(--color-success-hover)] text-white rounded-lg md:rounded-xl text-sm md:text-base font-semibold disabled:opacity-50 transition-colors shadow-lg"
                >
                  {connected ? "연결 중..." : "카메라 시작"}
                </button>
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
        />

        {/* Product List */}
        <div className="flex-1 min-h-0">
          <BillingPanel
            billingItems={billingItems}
            itemScores={itemScores}
            totalCount={totalCount}
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
      totalCount={totalCount}
    />
  </>
  );
}
