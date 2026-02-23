import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAuthStore } from "../stores/authStore";
import { getDashboardStats } from "../api/purchases";

const PERIOD_OPTIONS = [7, 30, 90] as const;
type PeriodDays = (typeof PERIOD_OPTIONS)[number];

const formatAmount = (value: number) => `₩${(value ?? 0).toLocaleString("ko-KR")}`;

function formatCompactAmount(value: number): string {
  if (value <= 0) return "0";
  if (value >= 100000000) return `${(value / 100000000).toFixed(1)}억`;
  if (value >= 10000) return `${Math.round(value / 10000)}만`;
  return `${Math.round(value / 1000)}천`;
}

function formatDateLabel(date: string): string {
  return new Date(date).toLocaleDateString("ko-KR", {
    month: "numeric",
    day: "numeric",
  });
}

function buildOrderTicks(maxValue: number): number[] {
  const ratios = [1, 0.75, 0.5, 0.25, 0];
  const raw = ratios.map((ratio) => Math.round(maxValue * ratio));
  const unique = Array.from(new Set(raw));
  if (!unique.includes(0)) unique.push(0);
  return unique;
}

export default function HomePage() {
  const { isAdmin, token } = useAuthStore();
  const [periodDays, setPeriodDays] = useState<PeriodDays>(7);

  // Fetch dashboard stats for admin
  const { data: stats, isLoading } = useQuery({
    queryKey: ["dashboard", "stats", periodDays],
    queryFn: () => getDashboardStats(token!, periodDays),
    enabled: isAdmin() && !!token,
  });

  // Admin Dashboard
  if (isAdmin()) {
    const dailyStats = stats?.daily_stats ?? [];
    const maxOrderCount = Math.max(1, ...dailyStats.map((row) => row.purchase_count));
    const maxPopularCount = Math.max(1, ...(stats?.popular_products ?? []).map((p) => p.total_count));
    const orderTicks = buildOrderTicks(maxOrderCount);
    const chartMinWidth = Math.max(560, dailyStats.length * 42);
    const revenueBadgeStride =
      dailyStats.length <= 10 ? 1 : dailyStats.length <= 30 ? 3 : 7;
    const labelStride =
      dailyStats.length <= 14 ? 1 : dailyStats.length <= 40 ? 4 : 8;

    return (
      <div className="p-4 md:p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-[var(--color-text)] mb-1">
            관리자 대시보드
          </h1>
          <p className="text-sm md:text-base text-[var(--color-text-secondary)]">
            매출/주문 추이와 고객 구매 패턴을 한눈에 확인하세요
          </p>
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <p className="text-[var(--color-text-secondary)]">로딩 중...</p>
          </div>
        ) : stats ? (
          <>
            <div className="bg-white rounded-2xl p-5 md:p-6 border border-[var(--color-border)] shadow-sm">
              <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                <div>
                  <h2 className="text-xl md:text-2xl font-bold text-[var(--color-text)]">
                    최근 {periodDays}일 매출/주문 추이
                  </h2>
                  <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                    좌측 축: 주문건수, 원형 뱃지: 일매출
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="inline-flex rounded-xl border border-orange-200 bg-orange-50/70 p-1">
                    {PERIOD_OPTIONS.map((daysOption) => (
                      <button
                        key={daysOption}
                        type="button"
                        onClick={() => setPeriodDays(daysOption)}
                        className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                          periodDays === daysOption
                            ? "bg-orange-500 text-white shadow-sm"
                            : "text-orange-700 hover:bg-orange-100"
                        }`}
                      >
                        {daysOption}일
                      </button>
                    ))}
                  </div>
                  <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 text-orange-700 text-xs font-semibold">
                    <span>오늘 매출</span>
                    <span>{formatAmount(stats.today_revenue)}</span>
                  </div>
                </div>
              </div>

              {dailyStats.length > 0 ? (
                <>
                  <div className="mt-6 overflow-x-auto pb-2">
                    <div style={{ minWidth: `${chartMinWidth}px` }}>
                      <div className="flex items-stretch gap-2">
                        <div className="w-10 shrink-0 h-72 relative">
                          {orderTicks.map((tick, idx) => {
                            const top = (idx / (orderTicks.length - 1 || 1)) * 100;
                            return (
                              <div
                                key={idx}
                                className="absolute right-0 -translate-y-1/2 text-[11px] text-orange-700 font-semibold"
                                style={{ top: `${top}%` }}
                              >
                                {tick}
                              </div>
                            );
                          })}
                        </div>

                        <div className="flex-1">
                          <div className="relative h-72">
                            {orderTicks.map((_, idx) => {
                              const top = (idx / (orderTicks.length - 1 || 1)) * 100;
                              return (
                                <div
                                  key={idx}
                                  className="absolute left-0 right-0 border-t border-orange-100"
                                  style={{ top: `${top}%` }}
                                />
                              );
                            })}

                            <div
                              className="absolute inset-0 grid gap-2"
                              style={{
                                gridTemplateColumns: `repeat(${dailyStats.length}, minmax(0, 1fr))`,
                              }}
                            >
                              {dailyStats.map((point, idx) => {
                                const baseHeight =
                                  point.purchase_count > 0
                                    ? 10 + Math.round((point.purchase_count / maxOrderCount) * 78)
                                    : 5;
                                const showRevenueBadge =
                                  idx % revenueBadgeStride === 0 || idx === dailyStats.length - 1;

                                return (
                                  <div key={point.date} className="relative flex items-end justify-center">
                                    <div
                                      className="w-full max-w-[24px] rounded-t-md bg-gradient-to-t from-orange-500 via-orange-400 to-amber-300"
                                      style={{ height: `${baseHeight}%` }}
                                    />
                                    {showRevenueBadge && (
                                      <div
                                        className="absolute left-1/2 -translate-x-1/2 h-7 px-2 rounded-full border border-orange-300 bg-white text-orange-700 text-[10px] font-semibold flex items-center justify-center shadow-sm whitespace-nowrap"
                                        style={{ bottom: `calc(${baseHeight}% + 6px)` }}
                                      >
                                        {formatCompactAmount(point.revenue)}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          <div
                            className="mt-3 grid gap-2"
                            style={{
                              gridTemplateColumns: `repeat(${dailyStats.length}, minmax(0, 1fr))`,
                            }}
                          >
                            {dailyStats.map((point, idx) => {
                              const showLabel =
                                idx % labelStride === 0 || idx === dailyStats.length - 1;
                              return (
                                <div
                                  key={`${point.date}-label`}
                                  className="text-center text-xs text-[var(--color-text-secondary)]"
                                >
                                  {showLabel ? formatDateLabel(point.date) : ""}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="mt-6 h-64 rounded-xl border border-dashed border-[var(--color-border)] flex items-center justify-center text-sm text-[var(--color-text-secondary)]">
                  최근 7일 차트 데이터가 없습니다.
                </div>
              )}

              <div className="mt-4 text-xs text-[var(--color-text-secondary)]">
                과거 가격 미확정 주문은 0원으로 집계될 수 있습니다.
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              <div className="bg-white rounded-2xl p-5 border border-[var(--color-border)] shadow-sm">
                <p className="text-sm text-[var(--color-text-secondary)]">총 매출</p>
                <p className="mt-2 text-2xl font-bold text-[var(--color-text)]">
                  {formatAmount(stats.total_revenue)}
                </p>
              </div>

              <div className="bg-white rounded-2xl p-5 border border-[var(--color-border)] shadow-sm">
                <p className="text-sm text-[var(--color-text-secondary)]">평균 객단가</p>
                <p className="mt-2 text-2xl font-bold text-[var(--color-text)]">
                  {formatAmount(stats.average_order_value)}
                </p>
              </div>

              <div className="bg-white rounded-2xl p-5 border border-[var(--color-border)] shadow-sm">
                <p className="text-sm text-[var(--color-text-secondary)]">총 주문수</p>
                <p className="mt-2 text-2xl font-bold text-[var(--color-text)]">
                  {stats.total_purchases}건
                </p>
              </div>

              <div className="bg-white rounded-2xl p-5 border border-[var(--color-border)] shadow-sm">
                <p className="text-sm text-[var(--color-text-secondary)]">오늘 주문수</p>
                <p className="mt-2 text-2xl font-bold text-[var(--color-text)]">
                  {stats.today_purchases}건
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-2xl p-6 border border-[var(--color-border)] shadow-sm">
                <h2 className="text-lg md:text-xl font-semibold text-[var(--color-text)] mb-4 flex items-center gap-2">
                  <span className="text-xl">🏆</span>
                  인기 상품 TOP 5
                </h2>
                {stats.popular_products.length > 0 ? (
                  <div className="space-y-3">
                    {stats.popular_products.map((product, index) => {
                      const ratio = Math.max(8, Math.round((product.total_count / maxPopularCount) * 100));
                      return (
                        <div key={product.name} className="rounded-xl border border-[var(--color-border)] p-3">
                          <div className="flex items-center justify-between gap-3">
                            <p className="text-sm font-semibold text-[var(--color-text)]">
                              {index + 1}. {product.name}
                            </p>
                            <span className="text-xs font-semibold text-[var(--color-primary)]">
                              {product.total_count}개
                            </span>
                          </div>
                          <div className="mt-2 h-2 rounded-full bg-gray-100 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-orange-500 to-amber-400"
                              style={{ width: `${ratio}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="h-40 rounded-xl border border-dashed border-[var(--color-border)] flex items-center justify-center text-sm text-[var(--color-text-secondary)]">
                    아직 판매된 상품이 없습니다.
                  </div>
                )}
              </div>

              <div className="bg-white rounded-2xl p-6 border border-[var(--color-border)] shadow-sm">
                <h2 className="text-lg md:text-xl font-semibold text-[var(--color-text)] mb-4 flex items-center gap-2">
                  <span className="text-xl">🧾</span>
                  최근 구매 내역
                </h2>
                {stats.recent_purchases.length > 0 ? (
                  <div className="space-y-3">
                    {stats.recent_purchases.map((purchase) => (
                      <div
                        key={purchase.id}
                        className="p-3 border border-[var(--color-border)] rounded-xl hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div>
                            <p className="text-sm font-semibold text-[var(--color-text)]">
                              {purchase.username}
                            </p>
                            <p className="text-xs text-[var(--color-text-secondary)]">
                              {new Date(purchase.timestamp).toLocaleDateString("ko-KR")}
                            </p>
                          </div>
                          <span className="text-sm font-semibold text-[var(--color-primary)]">
                            {formatAmount(purchase.total_amount ?? 0)}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {purchase.items.map((item, idx) => (
                            <span
                              key={idx}
                              className="text-xs px-2 py-1 bg-orange-50 text-orange-700 rounded"
                            >
                              {item.name} × {item.count}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-40 rounded-xl border border-dashed border-[var(--color-border)] flex items-center justify-center text-sm text-[var(--color-text-secondary)]">
                    아직 구매 내역이 없습니다.
                  </div>
                )}
              </div>
            </div>
          </>
        ) : null}
      </div>
    );
  }

  // User HomePage (기존 UI)
  return (
    <div className="p-4 md:p-6 lg:p-8 max-w-6xl mx-auto space-y-4 md:space-y-6 lg:space-y-8">
      {/* System Ready Banner */}
      <div className="bg-white rounded-2xl p-6 border border-[var(--color-border)] shadow-sm">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-full bg-[var(--color-success-light)] flex items-center justify-center flex-shrink-0">
            <svg
              className="w-7 h-7 text-[var(--color-success)]"
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
            <h2 className="text-2xl font-bold text-[var(--color-text)] mb-2">
              시스템 준비 완료
            </h2>
            <p className="text-[var(--color-text-secondary)]">
              스마트 체크아웃 시스템에 오신 것을 환영합니다
            </p>
          </div>
        </div>
      </div>

      {/* CTA Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Checkout Card */}
        <div className="bg-white rounded-2xl p-8 border border-[var(--color-border)] shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-14 h-14 rounded-2xl bg-[var(--color-primary-light)] flex items-center justify-center flex-shrink-0">
              <span className="text-3xl">🛒</span>
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-[var(--color-text)] mb-2">
                체크아웃
              </h3>
              <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                실시간 카메라 피드를 통해 상품을 인식하고 장바구니를 자동으로 구성합니다.
                정확한 인식을 위해 ROI 영역을 설정할 수 있습니다.
              </p>
            </div>
          </div>
          <Link
            to="/checkout"
            className="block w-full py-3 px-6 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white font-medium rounded-xl text-center transition-colors"
          >
            체크아웃 시작
          </Link>
        </div>

        {/* Validate Card */}
        <div className="bg-white rounded-2xl p-8 border border-[var(--color-border)] shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-14 h-14 rounded-2xl bg-[var(--color-secondary-light)] flex items-center justify-center flex-shrink-0">
              <span className="text-3xl">📋</span>
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-[var(--color-text)] mb-2">
                영수증 확인
              </h3>
              <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                체크아웃한 상품 목록을 확인하고 수정할 수 있습니다. 최종
                결제 전에 상품을 검토하고 조정하세요.
              </p>
            </div>
          </div>
          <Link
            to="/validate"
            className="block w-full py-3 px-6 bg-white hover:bg-gray-50 text-[var(--color-text)] font-medium rounded-xl text-center border border-[var(--color-border)] transition-colors"
          >
            영수증 확인
          </Link>
        </div>
      </div>

      {/* Key Features */}
      <div>
        <h3 className="text-lg font-bold text-[var(--color-text)] mb-4">
          주요 기능
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Feature 1 */}
          <div className="bg-white rounded-xl p-6 border border-[var(--color-border)] text-center">
            <div className="w-12 h-12 rounded-full bg-[var(--color-success-light)] flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-6 h-6 text-[var(--color-success)]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <h4 className="font-semibold text-[var(--color-text)] mb-2">
              실시간 인식
            </h4>
            <p className="text-sm text-[var(--color-text-secondary)]">
              AI 기반 실시간 상품 인식으로 빠르고 정확한 체크아웃
            </p>
          </div>

          {/* Feature 2 */}
          <div className="bg-white rounded-xl p-6 border border-[var(--color-border)] text-center">
            <div className="w-12 h-12 rounded-full bg-[var(--color-primary-light)] flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🛒</span>
            </div>
            <h4 className="font-semibold text-[var(--color-text)] mb-2">
              자동 장바구니
            </h4>
            <p className="text-sm text-[var(--color-text-secondary)]">
              인식된 상품이 자동으로 장바구니에 추가
            </p>
          </div>

          {/* Feature 3 */}
          <div className="bg-white rounded-xl p-6 border border-[var(--color-border)] text-center">
            <div className="w-12 h-12 rounded-full bg-[var(--color-secondary-light)] flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-6 h-6 text-[var(--color-secondary)]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                />
              </svg>
            </div>
            <h4 className="font-semibold text-[var(--color-text)] mb-2">
              간편한 검증
            </h4>
            <p className="text-sm text-[var(--color-text-secondary)]">
              영수증 확인 페이지에서 쉽게 검토 및 수정
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
