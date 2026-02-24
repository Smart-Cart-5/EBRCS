import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { addProduct, deleteProduct, listProducts } from "../api/products";

const MIN_IMAGES = 3;
const MAX_IMAGES = 5;

interface CapturedImage {
  id: string;
  file: File;
  url: string;
}

export default function ProductsPage() {
  const queryClient = useQueryClient();
  const [itemNo, setItemNo] = useState("");
  const [name, setName] = useState("");
  const [price, setPrice] = useState("");
  const [barcd, setBarcd] = useState("");

  const [captured, setCaptured] = useState<CapturedImage[]>([]);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [startingCamera, setStartingCamera] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<{
    itemNo: string;
    name: string;
  } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const capturedRef = useRef<CapturedImage[]>([]);

  const images = useMemo(() => captured.map((c) => c.file), [captured]);
  const priceValue = Number(price);
  const hasValidPrice = Number.isFinite(priceValue) && priceValue > 0;
  const canSubmit =
    itemNo.trim().length > 0 &&
    name.trim().length > 0 &&
    hasValidPrice &&
    images.length >= MIN_IMAGES &&
    images.length <= MAX_IMAGES;

  const { data, isLoading } = useQuery({
    queryKey: ["products"],
    queryFn: listProducts,
  });

  const mutation = useMutation({
    mutationFn: () =>
      addProduct({
        itemNo,
        name,
        price: priceValue,
        barcd,
        images,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["products"] });
      setItemNo("");
      setName("");
      setPrice("");
      setBarcd("");
      setCaptured((prev) => {
        prev.forEach((img) => URL.revokeObjectURL(img.url));
        return [];
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (itemNoValue: string) => deleteProduct(itemNoValue),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
  });

  const loadDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = list.filter((device) => device.kind === "videoinput");
      setDevices(videoInputs);
      if (!selectedDeviceId && videoInputs.length > 0) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch {
      // ignore
    }
  };

  const stopCamera = (updateState = true) => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (updateState) {
      setCameraActive(false);
    }
  };

  const startCamera = async (deviceId?: string) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("이 브라우저는 카메라를 지원하지 않습니다.");
      return;
    }
    setStartingCamera(true);
    setCameraError(null);
    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : { facingMode: "environment" },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
      await loadDevices();
    } catch (err) {
      setCameraError("카메라 접근에 실패했습니다. 권한과 HTTPS를 확인해주세요.");
      setCameraActive(false);
    } finally {
      setStartingCamera(false);
    }
  };

  const captureImage = async () => {
    if (!videoRef.current) return;
    if (images.length >= MAX_IMAGES) {
      setCameraError(`최대 ${MAX_IMAGES}장까지 촬영할 수 있습니다.`);
      return;
    }

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setCameraError("카메라가 아직 준비되지 않았습니다.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setCameraError("캡처에 실패했습니다.");
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.92)
    );

    if (!blob) {
      setCameraError("캡처에 실패했습니다.");
      return;
    }

    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const file = new File([blob], `capture-${Date.now()}.jpg`, {
      type: "image/jpeg",
    });
    const url = URL.createObjectURL(blob);

    setCaptured((prev) => [...prev, { id, file, url }]);
  };

  const removeCapture = (id: string) => {
    setCaptured((prev) => {
      const target = prev.find((img) => img.id === id);
      if (target) URL.revokeObjectURL(target.url);
      return prev.filter((img) => img.id !== id);
    });
  };

  const clearCaptures = () => {
    setCaptured((prev) => {
      prev.forEach((img) => URL.revokeObjectURL(img.url));
      return [];
    });
  };

  useEffect(() => {
    capturedRef.current = captured;
  }, [captured]);

  useEffect(() => {
    loadDevices();
    return () => {
      stopCamera(false);
      capturedRef.current.forEach((img) => URL.revokeObjectURL(img.url));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!cameraActive) return;
    if (selectedDeviceId) {
      startCamera(selectedDeviceId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDeviceId]);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h2 className="text-xl font-bold">상품 등록 (카메라)</h2>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <div className="space-y-6">
          <div className="bg-white border border-[var(--color-border)] rounded-xl p-6 space-y-4">
            <div className="flex items-center justify-between gap-3">
              <h3 className="font-semibold">카메라 미리보기</h3>
              <div className="flex items-center gap-2">
                {devices.length > 1 && (
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => setSelectedDeviceId(e.target.value)}
                    className="border border-[var(--color-border)] rounded-lg px-2 py-1 text-sm"
                  >
                    {devices.map((device, idx) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `카메라 ${idx + 1}`}
                      </option>
                    ))}
                  </select>
                )}
                <button
                  type="button"
                  onClick={() => (cameraActive ? stopCamera() : startCamera(selectedDeviceId || undefined))}
                  className="px-3 py-1.5 rounded-lg text-sm bg-[var(--color-primary)] text-white"
                  disabled={startingCamera}
                >
                  {cameraActive ? "카메라 종료" : "카메라 시작"}
                </button>
              </div>
            </div>

            <div className="relative w-full aspect-video bg-[var(--color-bg-muted)] rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {!cameraActive && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-[var(--color-text-muted)]">
                  카메라를 시작하면 화면이 표시됩니다.
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={captureImage}
                disabled={!cameraActive || startingCamera}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-[var(--color-primary)] text-white disabled:opacity-50"
              >
                촬영
              </button>
              <button
                type="button"
                onClick={clearCaptures}
                disabled={captured.length === 0}
                className="px-4 py-2 rounded-lg text-sm font-medium border border-[var(--color-border)] disabled:opacity-50"
              >
                전체 삭제
              </button>
              <span className="text-sm text-[var(--color-text-muted)]">
                {captured.length}/{MAX_IMAGES}장 (최소 {MIN_IMAGES}장)
              </span>
            </div>

            {cameraError && (
              <p className="text-sm text-[var(--color-danger)]">{cameraError}</p>
            )}
            <p className="text-xs text-[var(--color-text-muted)]">
              HTTPS(또는 localhost) 환경에서 카메라 권한이 필요합니다.
            </p>
          </div>

          <div className="bg-white border border-[var(--color-border)] rounded-xl p-6 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">촬영 이미지</h3>
              {captured.length > 0 && (
                <span className="text-xs text-[var(--color-text-muted)]">
                  클릭해서 개별 삭제 가능
                </span>
              )}
            </div>
            {captured.length === 0 ? (
              <p className="text-sm text-[var(--color-text-muted)]">
                아직 촬영된 이미지가 없습니다.
              </p>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {captured.map((img) => (
                  <div key={img.id} className="relative group">
                    <img
                      src={img.url}
                      alt="captured"
                      className="w-full h-28 object-cover rounded-lg border border-[var(--color-border)]"
                    />
                    <button
                      type="button"
                      onClick={() => removeCapture(img.id)}
                      className="absolute top-2 right-2 px-2 py-1 text-xs rounded bg-[var(--color-danger)] text-white opacity-0 group-hover:opacity-100 transition"
                    >
                      삭제
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white border border-[var(--color-border)] rounded-xl p-6 space-y-4">
            <h3 className="font-semibold">상품 정보</h3>
            <div>
              <label className="block text-sm font-medium mb-1">상품번호</label>
              <input
                type="text"
                value={itemNo}
                onChange={(e) => setItemNo(e.target.value)}
                placeholder="상품번호 (숫자)"
                className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">상품명</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="상품명"
                className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">가격</label>
              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="가격"
                className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">바코드</label>
              <input
                type="text"
                value={barcd}
                onChange={(e) => setBarcd(e.target.value)}
                placeholder="바코드 (선택)"
                className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>

            <button
              onClick={() => mutation.mutate()}
              disabled={!canSubmit || mutation.isPending}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-[var(--color-primary)] text-white disabled:opacity-50"
            >
              {mutation.isPending ? "등록 중..." : "상품 등록"}
            </button>

            {!canSubmit && (
              <p className="text-xs text-[var(--color-text-muted)]">
                상품 정보 입력 후 {MIN_IMAGES}~{MAX_IMAGES}장을 촬영해주세요.
              </p>
            )}

            {mutation.isError && (
              <p className="text-sm text-[var(--color-danger)]">
                {(mutation.error as Error).message}
              </p>
            )}
            {mutation.isSuccess && (
              <p className="text-sm text-[var(--color-success)]">
                상품이 성공적으로 등록되었습니다!
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="bg-white border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="font-semibold mb-3">상품 목록</h3>
        {isLoading ? (
          <p className="text-sm text-[var(--color-text-muted)]">Loading...</p>
        ) : data?.products.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">등록된 상품이 없습니다.</p>
        ) : (
          <ul className="divide-y divide-[var(--color-border)]">
            {data?.products.map((p) => (
              <li
                key={p.item_no}
                className="py-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between text-sm"
              >
                <div>
                  <p className="font-medium">{p.name}</p>
                  <p className="text-xs text-[var(--color-text-muted)]">상품번호: {p.item_no}</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[var(--color-text-muted)]">
                    {p.embedding_count} embeddings
                  </span>
                  <button
                    className="px-2 py-1 rounded bg-[var(--color-danger)] text-white text-xs"
                    onClick={() => setConfirmDelete({ itemNo: p.item_no, name: p.name })}
                    disabled={deleteMutation.isPending}
                  >
                    삭제
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
        {data && (
          <p className="text-xs text-[var(--color-text-muted)] mt-3">
            Total embeddings: {data.total_embeddings}
          </p>
        )}
      </div>

      {confirmDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => {
              if (!deleteMutation.isPending) setConfirmDelete(null);
            }}
          />
          <div className="relative w-full max-w-sm rounded-xl bg-white border border-[var(--color-border)] p-6 shadow-xl">
            <h4 className="text-base font-semibold mb-2">상품 삭제 확인</h4>
            <p className="text-sm text-[var(--color-text-secondary)]">
              &quot;{confirmDelete.name}&quot; 상품을 정말 삭제하시겠습니까?
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirmDelete(null)}
                disabled={deleteMutation.isPending}
                className="px-3 py-2 rounded-lg text-sm border border-[var(--color-border)] disabled:opacity-50"
              >
                취소
              </button>
              <button
                type="button"
                onClick={() => {
                  deleteMutation.mutate(confirmDelete.itemNo);
                  setConfirmDelete(null);
                }}
                disabled={deleteMutation.isPending}
                className="px-3 py-2 rounded-lg text-sm bg-[var(--color-danger)] text-white disabled:opacity-50"
              >
                삭제
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
