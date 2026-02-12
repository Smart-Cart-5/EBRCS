import { useCallback, useEffect, useRef, useState } from "react";
import toast, { Toaster } from "react-hot-toast";
import { useSessionStore, type WsMessage } from "../stores/sessionStore";
import { wsCheckoutUrl, uploadVideo, videoStatusUrl, setROI } from "../api/client";
import BillingPanel from "../components/BillingPanel";
import StatusMetrics from "../components/StatusMetrics";
import ProductDrawer from "../components/ProductDrawer";

type Mode = "camera" | "upload";

export default function CheckoutPage() {
  const [mode, setMode] = useState<Mode>("camera");
  const [connected, setConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
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
    roiPolygon,
    detectionBoxes,
  } = useSessionStore();

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null); // For capturing frames to send to backend
  const displayCanvasRef = useRef<HTMLCanvasElement>(null); // For rendering at 60 FPS
  const streamRef = useRef<MediaStream | null>(null);
  const captureAnimRef = useRef<number>(0); // For capture/send loop
  const renderAnimRef = useRef<number>(0); // For render loop (60 FPS)
  const prevBillingRef = useRef<Record<string, number>>({});

  // Ensure session exists and setup virtual ROI for entry-event mode
  useEffect(() => {
    if (!sessionId) {
      createSession();
    } else {
      // Setup full-screen virtual ROI to enable entry-event mode
      // This prevents counting the same object multiple times
      setROI(sessionId, [
        [0, 0],    // Top-left
        [1, 0],    // Top-right
        [1, 1],    // Bottom-right
        [0, 1],    // Bottom-left
      ]).catch((err) => {
        console.warn("Failed to set virtual ROI:", err);
      });
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
      cancelAnimationFrame(captureAnimRef.current);
      cancelAnimationFrame(renderAnimRef.current);
    };
  }, []);

  // --- Camera mode ---
  const startCamera = useCallback(async () => {
    if (!sessionId) return;

    try {
      // Start loading state
      setIsLoading(true);
      setLoadingMessage("카메라 권한 요청 중...");

      // Request camera and WebSocket in parallel for better UX
      console.log("📷 Requesting camera access...");

      // Start both operations in parallel
      const cameraPromise = navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "environment" },
      });

      const ws = new WebSocket(wsCheckoutUrl(sessionId));
      wsRef.current = ws;

      const wsPromise = new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error("WebSocket connection timeout"));
        }, 10000); // Increased to 10 seconds

        ws.onopen = () => {
          clearTimeout(timeout);
          console.log("✅ WebSocket connected");
          resolve();
        };
        ws.onerror = (err) => {
          clearTimeout(timeout);
          console.error("❌ WebSocket error:", err);
          reject(new Error("WebSocket connection failed"));
        };
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
      // Read state from zustand store directly each frame to avoid stale closures
      const renderFrame = () => {
        renderAnimRef.current = requestAnimationFrame(renderFrame);

        // Draw video to display canvas
        displayCtx.drawImage(video, 0, 0, displayCanvas.width, displayCanvas.height);

        // Read latest state from store (not from closure)
        const state = useSessionStore.getState();
        const boxes = state.detectionBoxes;
        const roi = state.roiPolygon;
        const label = state.lastLabel;
        const score = state.lastScore;
        const status = state.lastStatus;

        // Draw YOLO detection bounding boxes
        if (boxes && boxes.length > 0) {
          boxes.forEach(detection => {
            const [x1, y1, x2, y2] = detection.box;
            const x = x1 * displayCanvas.width;
            const y = y1 * displayCanvas.height;
            const w = (x2 - x1) * displayCanvas.width;
            const h = (y2 - y1) * displayCanvas.height;

            // Color: product = green, hand = red
            displayCtx.strokeStyle = detection.class === 'product' ? 'rgb(0, 255, 0)' : 'rgb(255, 0, 0)';
            displayCtx.lineWidth = 2;
            displayCtx.strokeRect(x, y, w, h);

            // Draw label if matched
            if (detection.label && detection.score) {
              const text = `${detection.label} (${detection.score.toFixed(3)})`;
              displayCtx.font = 'bold 16px sans-serif';

              // Text shadow
              displayCtx.strokeStyle = 'black';
              displayCtx.lineWidth = 3;
              displayCtx.strokeText(text, x, Math.max(20, y - 5));

              displayCtx.fillStyle = 'white';
              displayCtx.fillText(text, x, Math.max(20, y - 5));
            }
          });
        }

        // Draw ROI polygon overlay
        if (roi && roi.length > 0) {
          displayCtx.strokeStyle = 'rgb(0, 181, 255)';
          displayCtx.lineWidth = 2;
          displayCtx.beginPath();

          roi.forEach((point, i) => {
            const x = point[0] * displayCanvas.width;
            const y = point[1] * displayCanvas.height;
            if (i === 0) {
              displayCtx.moveTo(x, y);
            } else {
              displayCtx.lineTo(x, y);
            }
          });

          displayCtx.closePath();
          displayCtx.stroke();
        }

        // Draw status text overlay (top-left)
        if (label) {
          const text = `${label} (${score.toFixed(3)})`;
          displayCtx.font = 'bold 20px sans-serif';

          // Text shadow for better visibility
          displayCtx.strokeStyle = 'black';
          displayCtx.lineWidth = 4;
          displayCtx.strokeText(text, 10, 30);

          displayCtx.fillStyle = 'white';
          displayCtx.fillText(text, 10, 30);
        }

        // Draw status indicator (bottom-left)
        if (status) {
          displayCtx.font = '14px sans-serif';
          displayCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
          displayCtx.fillRect(10, displayCanvas.height - 30, 150, 20);
          displayCtx.fillStyle = 'white';
          displayCtx.fillText(status, 15, displayCanvas.height - 15);
        }
      };

      // 10-15 FPS capture and send loop
      let lastSend = 0;
      let frameCount = 0;
      const sendFrame = () => {
        captureAnimRef.current = requestAnimationFrame(sendFrame);
        const now = performance.now();

        // Throttle to 10-15 FPS (66-100ms interval)
        if (now - lastSend < 80) return; // ~12.5 FPS

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

      // All ready - show camera feed
      setLoadingMessage("완료!");
      setTimeout(() => {
        setConnected(true);
        setIsLoading(false);
        setLoadingMessage("");
      }, 300); // Small delay for smooth transition
    } catch (error) {
      console.error("❌ Camera error:", error);
      setIsLoading(false);
      setLoadingMessage("");
      alert(`Failed to start camera: ${error instanceof Error ? error.message : String(error)}`);
      stopCamera();
    }
  }, [sessionId, updateFromWsMessage]);

  const stopCamera = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    cancelAnimationFrame(captureAnimRef.current);
    cancelAnimationFrame(renderAnimRef.current);
    setConnected(false);
    setIsLoading(false);
    setLoadingMessage("");
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
                  onClick={startCamera}
                  disabled={isLoading}
                  className="px-5 py-2.5 md:px-6 md:py-3 bg-[var(--color-success)] hover:bg-[var(--color-success-hover)] text-white rounded-lg md:rounded-xl text-sm md:text-base font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg"
                >
                  {isLoading ? "준비 중..." : "카메라 시작"}
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
