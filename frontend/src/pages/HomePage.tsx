import { Link } from "react-router-dom";

export default function HomePage() {
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
