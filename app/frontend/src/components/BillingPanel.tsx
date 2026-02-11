import { useNavigate } from "react-router-dom";

interface Props {
  billingItems: Record<string, number>;
  itemScores: Record<string, number>;
  totalCount: number;
}

export default function BillingPanel({
  billingItems,
  itemScores,
  totalCount,
}: Props) {
  const navigate = useNavigate();
  const entries = Object.entries(billingItems).sort(([a], [b]) =>
    a.localeCompare(b),
  );

  return (
    <div className="bg-white border border-[var(--color-border)] rounded-2xl p-6 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📦</span>
          <h3 className="font-bold text-[var(--color-text)]">인식된 상품</h3>
        </div>
        <div className="w-8 h-8 rounded-full bg-[var(--color-primary)] text-white flex items-center justify-center text-sm font-bold">
          {totalCount}
        </div>
      </div>

      {/* Product List */}
      <div className="flex-1 overflow-auto space-y-2 mb-4">
        {entries.length === 0 ? (
          <div className="text-center py-8 text-[var(--color-text-secondary)]">
            <p className="text-sm">인식된 상품이 없습니다</p>
          </div>
        ) : (
          entries.map(([name, qty]) => {
            const score = itemScores[name] ?? 0;
            const scorePercent = Math.round(score * 100);
            return (
              <div
                key={name}
                className="bg-gray-50 rounded-xl p-3 flex items-center justify-between"
              >
                <div className="flex-1">
                  <p className="font-medium text-[var(--color-text)]">{name}</p>
                  <p className="text-xs text-[var(--color-secondary)] font-semibold">
                    {scorePercent}%
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">
                    신뢰도: {score.toFixed(2)}
                  </p>
                </div>
                <div className="w-8 h-8 rounded-full bg-[var(--color-primary)] text-white flex items-center justify-center text-sm font-bold flex-shrink-0">
                  {qty}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Total & Checkout Button */}
      <div className="space-y-3 pt-4 border-t border-[var(--color-border)]">
        <div className="flex items-center justify-between">
          <span className="text-sm text-[var(--color-text-secondary)]">
            총 상품 수
          </span>
          <span className="text-2xl font-bold text-[var(--color-primary)]">
            {totalCount}개
          </span>
        </div>
        {totalCount > 0 && (
          <button
            onClick={() => navigate("/validate")}
            className="w-full py-3 px-6 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white font-bold rounded-xl transition-colors"
          >
            체크아웃 완료
          </button>
        )}
      </div>
    </div>
  );
}
