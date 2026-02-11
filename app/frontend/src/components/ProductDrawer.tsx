import { useEffect, useState } from "react";

interface ProductDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  billingItems: Record<string, number>;
  itemScores: Record<string, number>;
  totalCount: number;
}

export default function ProductDrawer({
  isOpen,
  onClose,
  billingItems,
  itemScores,
  totalCount,
}: ProductDrawerProps) {
  const [touchStart, setTouchStart] = useState(0);
  const [touchEnd, setTouchEnd] = useState(0);

  const entries = Object.entries(billingItems).sort(([a], [b]) =>
    a.localeCompare(b),
  );

  // Swipe down to close
  const handleTouchStart = (e: React.TouchEvent) => {
    setTouchStart(e.targetTouches[0].clientY);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientY);
  };

  const handleTouchEnd = () => {
    if (touchStart - touchEnd < -50) {
      // Swiped down
      onClose();
    }
  };

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }
    return () => {
      document.body.style.overflow = "auto";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 lg:hidden"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed bottom-0 left-0 right-0 bg-white rounded-t-3xl shadow-2xl z-50 lg:hidden max-h-[80vh] flex flex-col">
        {/* Handle */}
        <div
          className="py-3 cursor-grab active:cursor-grabbing"
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        >
          <div className="w-12 h-1 bg-gray-300 rounded-full mx-auto" />
        </div>

        {/* Header */}
        <div className="px-4 pb-3 border-b border-[var(--color-border)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xl">📦</span>
              <h3 className="text-lg font-bold text-[var(--color-text)]">
                인식된 상품
              </h3>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-[var(--color-primary-light)] text-[var(--color-primary)] text-sm font-semibold rounded-full">
                {entries.length}개 품목
              </span>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 text-2xl"
              >
                ×
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {entries.length === 0 ? (
            <div className="p-8 text-center">
              <span className="text-5xl mb-3 block">🛒</span>
              <p className="text-sm text-[var(--color-text-secondary)]">
                인식된 상품이 없습니다
              </p>
            </div>
          ) : (
            <div className="divide-y divide-[var(--color-border)]">
              {entries.map(([name, qty]) => (
                <div key={name} className="p-4 flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-[var(--color-text)]">
                        {name}
                      </span>
                      <span className="px-2 py-0.5 bg-[var(--color-secondary-light)] text-[var(--color-secondary)] text-xs font-medium rounded">
                        {((itemScores[name] ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-xs text-[var(--color-text-secondary)]">
                      유사도: {(itemScores[name] ?? 0).toFixed(3)}
                    </p>
                  </div>
                  <div className="w-10 h-10 rounded-full bg-[var(--color-primary)] text-white flex items-center justify-center font-bold">
                    {qty}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[var(--color-border)] bg-gray-50">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-[var(--color-text)]">
              총 상품 수
            </span>
            <span className="text-2xl font-bold text-[var(--color-primary)]">
              {totalCount}개
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
