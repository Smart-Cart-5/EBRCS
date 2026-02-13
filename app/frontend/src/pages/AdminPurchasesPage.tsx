import { useQuery } from "@tanstack/react-query";
import { useAuthStore } from "../stores/authStore";
import { getAllPurchases } from "../api/client";

export default function AdminPurchasesPage() {
  const { token } = useAuthStore();

  const { data: purchases, isLoading } = useQuery({
    queryKey: ["purchases", "all"],
    queryFn: () => getAllPurchases(token!),
    enabled: !!token,
  });

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--color-text)] mb-2">
          전체 구매 내역 관리
        </h1>
        <p className="text-[var(--color-text-secondary)]">
          모든 사용자의 구매 내역을 확인하세요
        </p>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white rounded-2xl p-6 border border-[var(--color-border)] shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center">
              <span className="text-2xl">📊</span>
            </div>
            <span className="text-sm text-[var(--color-text-secondary)]">총 구매 건수</span>
          </div>
          <div className="text-3xl font-bold text-[var(--color-text)]">
            {purchases?.length || 0}건
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-[var(--color-border)] shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-green-100 flex items-center justify-center">
              <span className="text-2xl">👥</span>
            </div>
            <span className="text-sm text-[var(--color-text-secondary)]">고객 수</span>
          </div>
          <div className="text-3xl font-bold text-[var(--color-text)]">
            {purchases ? new Set(purchases.map(p => p.user_id)).size : 0}명
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-[var(--color-border)] shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center">
              <span className="text-2xl">📦</span>
            </div>
            <span className="text-sm text-[var(--color-text-secondary)]">총 상품 수</span>
          </div>
          <div className="text-3xl font-bold text-[var(--color-text)]">
            {purchases?.reduce((sum, p) => sum + p.items.reduce((s, i) => s + i.count, 0), 0) || 0}개
          </div>
        </div>
      </div>

      {/* Purchase History Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-[var(--color-border)] overflow-hidden">
        <div className="p-6 border-b border-[var(--color-border)]">
          <h2 className="text-xl font-semibold text-[var(--color-text)]">
            구매 내역
          </h2>
        </div>

        <div className="overflow-x-auto">
          {isLoading ? (
            <div className="text-center py-12">
              <p className="text-[var(--color-text-secondary)]">로딩 중...</p>
            </div>
          ) : purchases && purchases.length > 0 ? (
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-[var(--color-border)]">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
                    사용자
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
                    구매 일시
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
                    상품 목록
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
                    총 개수
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border)]">
                {purchases.map((purchase: any) => (
                  <tr key={purchase.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-[var(--color-text)]">
                      #{purchase.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm">
                        <div className="font-medium text-[var(--color-text)]">
                          {purchase.username}
                        </div>
                        <div className="text-[var(--color-text-secondary)]">
                          ID: {purchase.user_id}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-[var(--color-text-secondary)]">
                      {new Date(purchase.timestamp).toLocaleString("ko-KR")}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-2">
                        {purchase.items.map((item: any, idx: number) => (
                          <span
                            key={idx}
                            className="inline-flex items-center gap-1 px-2.5 py-1 bg-blue-50 text-blue-700 text-xs font-medium rounded-lg"
                          >
                            {item.name} × {item.count}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm font-semibold text-[var(--color-text)]">
                        {purchase.items.reduce((sum: number, item: any) => sum + item.count, 0)}개
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📊</div>
              <p className="text-[var(--color-text-secondary)] mb-2">
                구매 내역이 없습니다
              </p>
              <p className="text-sm text-[var(--color-text-secondary)]">
                아직 등록된 구매 내역이 없습니다.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
