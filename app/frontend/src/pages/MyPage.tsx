import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "../stores/authStore";
import { deletePurchase, getMyPurchases } from "../api/purchases";

export default function MyPage() {
  const { user, token } = useAuthStore();
  const queryClient = useQueryClient();
  const [deletingPurchaseId, setDeletingPurchaseId] = useState<number | null>(null);
  const formatAmount = (value: number) => `₩${value.toLocaleString("ko-KR")}`;

  const { data: purchases, isLoading } = useQuery({
    queryKey: ["purchases", "my"],
    queryFn: () => getMyPurchases(token!),
    enabled: !!token,
  });

  const deleteMutation = useMutation({
    mutationFn: async (purchaseId: number) => {
      setDeletingPurchaseId(purchaseId);
      return deletePurchase(token!, purchaseId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["purchases", "my"] });
    },
    onError: (error) => {
      alert((error as Error).message || "구매 내역 삭제 중 오류가 발생했습니다.");
    },
    onSettled: () => {
      setDeletingPurchaseId(null);
    },
  });

  const handleDeletePurchase = useCallback((purchaseId: number) => {
    if (deleteMutation.isPending) return;
    const ok = window.confirm("이 구매 내역을 삭제하시겠습니까?");
    if (!ok) return;
    deleteMutation.mutate(purchaseId);
  }, [deleteMutation]);

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--color-text)] mb-2">
          마이페이지
        </h1>
        <p className="text-[var(--color-text-secondary)]">
          {user?.name}님의 구매 내역을 확인하세요
        </p>
      </div>

      {/* Purchase History */}
      <div className="bg-white rounded-2xl shadow-sm border border-[var(--color-border)] overflow-hidden">
        <div className="p-6 border-b border-[var(--color-border)]">
          <h2 className="text-xl font-semibold text-[var(--color-text)]">
            구매 내역
          </h2>
        </div>

        <div className="p-6">
          {isLoading ? (
            <div className="text-center py-12">
              <p className="text-[var(--color-text-secondary)]">로딩 중...</p>
            </div>
          ) : purchases && purchases.length > 0 ? (
            <div className="space-y-4">
              {purchases.map((purchase) => (
                <div
                  key={purchase.id}
                  className="p-4 border border-[var(--color-border)] rounded-xl hover:bg-gray-50 transition-colors"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <p className="text-sm text-[var(--color-text-secondary)]">
                        {new Date(purchase.timestamp).toLocaleString("ko-KR")}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-semibold text-[var(--color-primary)]">
                        {formatAmount(purchase.total_amount ?? 0)}
                      </p>
                      <button
                        type="button"
                        onClick={() => handleDeletePurchase(purchase.id)}
                        disabled={deleteMutation.isPending}
                        className="px-2.5 py-1 text-xs font-medium rounded-lg border border-red-200 text-red-600 hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {deleteMutation.isPending && deletingPurchaseId === purchase.id
                          ? "삭제 중..."
                          : "삭제"}
                      </button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {purchase.items.map((item, itemIdx) => (
                      <div
                        key={itemIdx}
                        className="flex justify-between text-sm"
                      >
                        <span className="text-[var(--color-text)]">
                          {item.name}
                        </span>
                        <div className="text-right">
                          <p className="text-[var(--color-text-secondary)]">
                            {item.count}개
                          </p>
                          {item.line_total != null && (
                            <p className="text-[var(--color-primary)] font-medium">
                              {formatAmount(item.line_total)}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🛒</div>
              <p className="text-[var(--color-text-secondary)] mb-2">
                구매 내역이 없습니다
              </p>
              <p className="text-sm text-[var(--color-text-secondary)]">
                체크아웃을 통해 상품을 구매해보세요!
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
