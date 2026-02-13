import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useSessionStore } from "../stores/sessionStore";
import { useAuthStore } from "../stores/authStore";
import { updateBilling, confirmBilling, createPurchase } from "../api/client";

export default function ValidatePage() {
  const navigate = useNavigate();
  const { token } = useAuthStore();
  const {
    sessionId,
    billingItems,
    itemScores,
    totalCount,
    setBilling,
    resetSession,
  } = useSessionStore();

  const entries = Object.entries(billingItems).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const itemCount = entries.length;

  const handleQtyChange = useCallback(
    async (name: string, delta: number) => {
      if (!sessionId) return;
      const updated = { ...billingItems };
      const newQty = (updated[name] ?? 0) + delta;
      if (newQty <= 0) {
        delete updated[name];
      } else {
        updated[name] = newQty;
      }
      const result = await updateBilling(sessionId, updated);
      setBilling(result.billing_items);
    },
    [sessionId, billingItems, setBilling],
  );

  const handleConfirm = useCallback(async () => {
    if (!sessionId || !token) return;

    try {
      // Create purchase record
      const items = Object.entries(billingItems).map(([name, count]) => ({
        name,
        count,
      }));

      await createPurchase(token, {
        session_id: sessionId,
        items,
      });

      // Confirm billing
      await confirmBilling(sessionId);

      // Reset and navigate
      resetSession();
      alert("구매가 완료되었습니다! 마이페이지에서 구매 내역을 확인할 수 있습니다.");
      navigate("/mypage");
    } catch (error) {
      console.error("Purchase confirmation failed:", error);
      alert("구매 확정 중 오류가 발생했습니다.");
    }
  }, [sessionId, token, billingItems, resetSession, navigate]);

  return (
    <div className="h-full flex flex-col bg-[var(--color-bg)]">
      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-4xl mx-auto p-4 md:p-6 lg:p-8 space-y-4 md:space-y-6">
          {/* Header */}
          <div className="bg-white rounded-xl md:rounded-2xl p-4 md:p-6 border border-[var(--color-border)] shadow-sm">
            <div className="flex items-start gap-3 md:gap-4">
              <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-[var(--color-secondary-light)] flex items-center justify-center flex-shrink-0">
                <svg
                  className="w-6 h-6 md:w-7 md:h-7 text-[var(--color-secondary)]"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2.5}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
              <div>
                <h2 className="text-xl md:text-2xl font-bold text-[var(--color-text)] mb-1 md:mb-2">
                  영수증 확인
                </h2>
                <p className="text-sm md:text-base text-[var(--color-text-secondary)]">
                  상품 목록을 확인하고 수정하세요
                </p>
              </div>
            </div>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 gap-3 md:gap-4">
            {/* Total Items */}
            <div className="bg-white rounded-xl md:rounded-2xl p-4 md:p-6 border border-[var(--color-border)] shadow-sm">
              <div className="flex items-center gap-2 md:gap-3 mb-2 md:mb-3">
                <div className="w-8 h-8 md:w-10 md:h-10 rounded-lg md:rounded-xl bg-[var(--color-primary-light)] flex items-center justify-center">
                  <span className="text-xl md:text-2xl">📦</span>
                </div>
                <span className="text-xs md:text-sm text-[var(--color-text-secondary)]">
                  총 상품 수
                </span>
              </div>
              <div className="text-2xl md:text-3xl font-bold text-[var(--color-text)]">
                {totalCount}개
              </div>
            </div>

            {/* Product Types */}
            <div className="bg-white rounded-xl md:rounded-2xl p-4 md:p-6 border border-[var(--color-border)] shadow-sm">
              <div className="flex items-center justify-between mb-2 md:mb-3">
                <div className="flex items-center gap-2 md:gap-3">
                  <div className="w-8 h-8 md:w-10 md:h-10 rounded-lg md:rounded-xl bg-[var(--color-secondary-light)] flex items-center justify-center">
                    <span className="text-xl md:text-2xl">📊</span>
                  </div>
                  <span className="text-xs md:text-sm text-[var(--color-text-secondary)]">
                    품목 수
                  </span>
                </div>
                {itemCount > 0 && (
                  <span className="px-2 py-0.5 md:px-3 md:py-1 bg-[var(--color-primary-light)] text-[var(--color-primary)] text-xs font-semibold rounded-full">
                    검수 가능
                  </span>
                )}
              </div>
              <div className="text-2xl md:text-3xl font-bold text-[var(--color-text)]">
                {itemCount}개
              </div>
            </div>
          </div>

          {/* Product List */}
          <div className="bg-white rounded-2xl border border-[var(--color-border)] shadow-sm overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-[var(--color-border)]">
              <div className="flex items-center gap-2">
                <span className="text-xl">📋</span>
                <h3 className="text-lg font-bold text-[var(--color-text)]">
                  상품 목록
                </h3>
              </div>
              <span className="px-3 py-1 bg-[var(--color-primary-light)] text-[var(--color-primary)] text-sm font-semibold rounded-full">
                {itemCount}개 품목
              </span>
            </div>

            {/* Empty State or Product List */}
            {entries.length === 0 ? (
              <div className="p-12 text-center space-y-4">
                <div className="w-24 h-24 mx-auto bg-gray-100 rounded-2xl flex items-center justify-center">
                  <span className="text-5xl">🧾</span>
                </div>
                <div>
                  <p className="text-lg font-semibold text-[var(--color-text)] mb-2">
                    아직 확인할 상품이 없습니다
                  </p>
                  <p className="text-sm text-[var(--color-text-secondary)]">
                    체크아웃 페이지에서 상품을 먼저 인식해주세요.
                  </p>
                </div>
                <button
                  onClick={() => navigate("/checkout")}
                  className="px-6 py-3 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white rounded-xl font-semibold transition-colors shadow-sm"
                >
                  체크아웃으로 이동
                </button>
              </div>
            ) : (
              <div className="divide-y divide-[var(--color-border)]">
                {entries.map(([name, qty]) => (
                  <div
                    key={name}
                    className="p-6 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="font-semibold text-[var(--color-text)]">
                            {name}
                          </span>
                          <span className="px-2 py-0.5 bg-[var(--color-secondary-light)] text-[var(--color-secondary)] text-xs font-medium rounded">
                            {((itemScores[name] ?? 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-sm text-[var(--color-text-secondary)]">
                          유사도: {(itemScores[name] ?? 0).toFixed(3)}
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
                          <button
                            onClick={() => handleQtyChange(name, -1)}
                            className="w-8 h-8 rounded-lg hover:bg-white text-lg font-bold transition-colors"
                          >
                            -
                          </button>
                          <span className="w-10 text-center font-bold text-[var(--color-text)]">
                            {qty}
                          </span>
                          <button
                            onClick={() => handleQtyChange(name, 1)}
                            className="w-8 h-8 rounded-lg hover:bg-white text-lg font-bold transition-colors"
                          >
                            +
                          </button>
                        </div>
                        <button
                          onClick={() => handleQtyChange(name, -qty)}
                          className="px-4 py-2 text-sm text-[var(--color-danger)] hover:bg-red-50 rounded-lg font-medium transition-colors"
                        >
                          삭제
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-[var(--color-border)] bg-white">
        <div className="max-w-4xl mx-auto p-4 md:p-6">
          <div className="flex items-center justify-between mb-3 md:mb-4">
            <span className="text-base md:text-lg font-semibold text-[var(--color-text)]">
              총 상품 수
            </span>
            <span className="text-3xl md:text-4xl font-bold text-[var(--color-primary)]">
              {totalCount}개
            </span>
          </div>
          <div className="flex gap-2 md:gap-3">
            <button
              onClick={() => navigate("/checkout")}
              className="flex-1 px-4 py-3 md:px-6 md:py-4 bg-white hover:bg-gray-50 border-2 border-[var(--color-border)] text-[var(--color-text)] rounded-lg md:rounded-xl text-sm md:text-base font-semibold transition-colors"
            >
              취소
            </button>
            <button
              onClick={handleConfirm}
              disabled={entries.length === 0}
              className="flex-1 px-4 py-3 md:px-6 md:py-4 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white rounded-lg md:rounded-xl text-sm md:text-base font-semibold transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              영수증 확정
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
