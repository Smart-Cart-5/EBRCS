import { useState, useEffect, useLayoutEffect, useRef } from "react";
import { Routes, Route, Link, useLocation, Navigate } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CheckoutPage from "./pages/CheckoutPage";
import ProductsPage from "./pages/ProductsPage";
import ValidatePage from "./pages/ValidatePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import MyPage from "./pages/MyPage";
import AdminPurchasesPage from "./pages/AdminPurchasesPage";
import ChatbotWidget from "./components/ChatbotWidget";
import { useAuthStore } from "./stores/authStore";
import { shouldRedirectForAdminRoute } from "./routing/guards";

// User menu items
const USER_NAV_ITEMS = [
  { path: "/", label: "홈", icon: "🏠" },
  { path: "/checkout", label: "장보GO", icon: "jangbogo" },
  { path: "/validate", label: "영수증 확인", icon: "📋" },
  { path: "/mypage", label: "마이페이지", icon: "👤" },
];

// Admin menu items
const ADMIN_NAV_ITEMS = [
  { path: "/", label: "홈", icon: "🏠" },
  { path: "/products", label: "상품 관리", icon: "📦" },
  { path: "/admin/purchases", label: "구매 내역", icon: "📊" },
];

export default function App() {
  const { pathname } = useLocation();
  const isCheckoutPage = pathname === "/checkout";
  const isAuthPage = pathname === "/login" || pathname === "/signup";

  const { user, clearAuth, isAuthenticated, isAdmin } = useAuthStore();
  const isAdminUser = isAdmin();
  const shouldRenderChatbot = true;
  const shouldUseMobileHeaderChatbot = !isCheckoutPage;
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);

  // Force scroll reset for both container and window (iOS may use window scroll).
  useLayoutEffect(() => {
    const resetScroll = () => {
      mainRef.current?.scrollTo({ top: 0, left: 0 });
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    };

    resetScroll();
    const rafId = window.requestAnimationFrame(resetScroll);

    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [pathname]);

  // Keep browser from restoring the previous scroll position on route changes.
  useEffect(() => {
    if (!("scrollRestoration" in window.history)) {
      return;
    }

    const previous = window.history.scrollRestoration;
    window.history.scrollRestoration = "manual";

    return () => {
      window.history.scrollRestoration = previous;
    };
  }, []);

  // Close profile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: PointerEvent) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target as Node)) {
        setIsProfileMenuOpen(false);
      }
    };

    if (isProfileMenuOpen) {
      document.addEventListener("pointerdown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("pointerdown", handleClickOutside);
    };
  }, [isProfileMenuOpen]);

  // Reset transient overlays when route changes to avoid mobile transition glitches.
  useEffect(() => {
    setIsProfileMenuOpen(false);
    setIsChatbotOpen(false);
  }, [pathname]);

  // Auth pages (login/signup)
  if (isAuthPage) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
      </Routes>
    );
  }

  // Require authentication for all other pages
  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }

  // Block direct access to admin routes for non-admin users.
  if (shouldRedirectForAdminRoute(pathname, isAdminUser)) {
    return <Navigate to="/" replace />;
  }

  // Select menu based on role
  const NAV_ITEMS = isAdminUser ? ADMIN_NAV_ITEMS : USER_NAV_ITEMS;

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Desktop Sidebar - Hidden on mobile */}
      <aside className="hidden lg:flex w-64 bg-[var(--color-sidebar)] border-r border-[var(--color-border)] flex-col relative z-[70]">
        {/* Logo */}
        <div className="p-6 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-3">
            <img
              src="/jangbogo.svg"
              alt="장보GO 로고"
              className="w-10 h-10 rounded-xl object-cover"
            />
            <div>
              <h1 className="text-lg font-bold text-[var(--color-text)]">
                장보GO
              </h1>
              <p className="text-xs text-[var(--color-text-secondary)]">
                JangboGO
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-[var(--color-primary-light)] text-[var(--color-primary)]"
                    : "text-[var(--color-text-secondary)] hover:bg-white hover:text-[var(--color-text)]"
                }`}
              >
                {item.icon === "jangbogo" ? (
                  <img
                    src="/jangbogo.svg"
                    alt="장보GO 아이콘"
                    className="w-5 h-5 rounded object-cover"
                  />
                ) : (
                  <span className="text-lg">{item.icon}</span>
                )}
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* User Info & Logout */}
        <div className="p-4 border-t border-[var(--color-border)]">
          {isAuthenticated() ? (
            <div className="space-y-3">
              <div className="px-4 py-2 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500">로그인됨</p>
                <p className="text-sm font-medium">{user?.name}</p>
                <p className="text-xs text-gray-500">{user?.role === 'admin' ? '관리자' : '사용자'}</p>
              </div>
              <button
                onClick={() => clearAuth()}
                className="w-full px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                로그아웃
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className="block w-full px-4 py-2 text-center text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors"
            >
              로그인
            </Link>
          )}
        </div>
      </aside>

      {/* Mobile Header - Hidden on checkout page */}
      {!isCheckoutPage && (
        <header className="lg:hidden bg-white border-b border-[var(--color-border)] px-4 py-3 relative z-40">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img
                src="/jangbogo.svg"
                alt="장보GO 로고"
                className="w-8 h-8 rounded-lg object-cover"
              />
              <h1 className="text-base font-bold text-[var(--color-text)]">
                장보GO
              </h1>
            </div>

            {/* Profile Menu Button */}
            <div className="flex items-center gap-2">
              {shouldUseMobileHeaderChatbot && (
                <button
                  type="button"
                  onClick={() => setIsChatbotOpen((prev) => !prev)}
                  className="w-8 h-8 rounded-full bg-[var(--color-primary)] text-white flex items-center justify-center text-sm"
                  aria-label={isChatbotOpen ? "챗봇 닫기" : "챗봇 열기"}
                >
                  {isChatbotOpen ? "✕" : "🤖"}
                </button>
              )}

              <div className="relative" ref={profileMenuRef}>
                <button
                  onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
                  className="w-8 h-8 rounded-full bg-[var(--color-primary-light)] text-[var(--color-primary)] flex items-center justify-center font-semibold text-sm"
                >
                  {user?.name?.[0] || "U"}
                </button>

                {/* Dropdown Menu */}
                {isProfileMenuOpen && (
                  <div className="absolute right-0 top-12 w-64 bg-white rounded-xl shadow-lg border border-[var(--color-border)] overflow-hidden z-50">
                    {/* User Info */}
                    <div className="p-4 bg-gray-50 border-b border-[var(--color-border)]">
                      <p className="text-xs text-gray-500">로그인됨</p>
                      <p className="text-sm font-semibold text-[var(--color-text)] mt-1">
                        {user?.name}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {user?.role === "admin" ? "관리자" : "사용자"}
                      </p>
                    </div>

                    {/* Logout Button */}
                    <button
                      onClick={() => {
                        clearAuth();
                        setIsProfileMenuOpen(false);
                      }}
                      className="w-full px-4 py-3 text-left text-sm font-medium text-red-600 hover:bg-red-50 transition-colors"
                    >
                      로그아웃
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>
      )}

      {/* Main Content */}
      <main
        ref={mainRef}
        className={`flex-1 overflow-auto ${isCheckoutPage ? 'pb-0 lg:pb-0' : 'pb-16 lg:pb-0'}`}
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/checkout" element={<CheckoutPage />} />
          <Route path="/validate" element={<ValidatePage />} />
          <Route path="/mypage" element={<MyPage />} />
          <Route path="/products" element={<ProductsPage />} />
          <Route path="/admin/purchases" element={<AdminPurchasesPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {/* Mobile Bottom Navigation - Only on mobile */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-[var(--color-border)] safe-area-pb z-40">
        <div className="grid h-16" style={{ gridTemplateColumns: `repeat(${NAV_ITEMS.length}, minmax(0, 1fr))` }}>
            {NAV_ITEMS.map((item) => {
              const isActive = pathname === item.path;
              return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex flex-col items-center justify-center gap-1 transition-colors ${
                  isActive
                    ? "text-[var(--color-primary)]"
                    : "text-[var(--color-text-secondary)]"
                }`}
              >
                {item.icon === "jangbogo" ? (
                  <img
                    src="/jangbogo.svg"
                    alt="장보GO 아이콘"
                    className="w-6 h-6 rounded object-cover"
                  />
                ) : (
                  <span className="text-xl">{item.icon}</span>
                )}
                <span className="text-xs font-medium">{item.label}</span>
              </Link>
            );
            })}
        </div>
      </nav>

      {shouldRenderChatbot && (
        <ChatbotWidget
          open={isChatbotOpen}
          onOpenChange={setIsChatbotOpen}
          hideMobileTrigger={shouldUseMobileHeaderChatbot}
        />
      )}
    </div>
  );
}
