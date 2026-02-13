import { useQuery } from "@tanstack/react-query";
import { useAuthStore } from "../stores/authStore";
import { getMyPurchases } from "../api/client";

export default function MyPage() {
  const { user, token } = useAuthStore();

  const { data: purchases, isLoading } = useQuery({
    queryKey: ["purchases", "my"],
    queryFn: () => getMyPurchases(token!),
    enabled: !!token,
  });

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
              {purchases.map((purchase: any, idx: number) => (
                <div
                  key={idx}
                  className="p-4 border border-[var(--color-border)] rounded-xl hover:bg-gray-50 transition-colors"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <p className="text-sm text-[var(--color-text-secondary)]">
                        {new Date(purchase.timestamp).toLocaleString("ko-KR")}
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {purchase.items.map((item: any, itemIdx: number) => (
                      <div
                        key={itemIdx}
                        className="flex justify-between text-sm"
                      >
                        <span className="text-[var(--color-text)]">
                          {item.name}
                        </span>
                        <span className="text-[var(--color-text-secondary)]">
                          {item.count}개
                        </span>
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
