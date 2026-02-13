import { useState, useEffect, useRef } from "react";
import { Routes, Route, Link, useLocation, Navigate } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CheckoutPage from "./pages/CheckoutPage";
import ProductsPage from "./pages/ProductsPage";
import ValidatePage from "./pages/ValidatePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import MyPage from "./pages/MyPage";
import AdminPurchasesPage from "./pages/AdminPurchasesPage";
import { useAuthStore } from "./stores/authStore";

// User menu items
const USER_NAV_ITEMS = [
  { path: "/", label: "홈", icon: "🏠" },
  { path: "/checkout", label: "체크아웃", icon: "🛒" },
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
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  // Close profile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target as Node)) {
        setIsProfileMenuOpen(false);
      }
    };

    if (isProfileMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isProfileMenuOpen]);

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

  // Select menu based on role
  const NAV_ITEMS = isAdmin() ? ADMIN_NAV_ITEMS : USER_NAV_ITEMS;

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Desktop Sidebar - Hidden on mobile */}
      <aside className="hidden lg:flex w-64 bg-[var(--color-sidebar)] border-r border-[var(--color-border)] flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[var(--color-primary)] flex items-center justify-center text-white text-xl">
              🏪
            </div>
            <div>
              <h1 className="text-lg font-bold text-[var(--color-text)]">
                스마트 체크아웃
              </h1>
              <p className="text-xs text-[var(--color-text-secondary)]">
                Smart Checkout
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
                <span className="text-lg">{item.icon}</span>
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
        <header className="lg:hidden bg-white border-b border-[var(--color-border)] px-4 py-3 relative">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white text-lg">
                🏪
              </div>
              <h1 className="text-base font-bold text-[var(--color-text)]">
                스마트 체크아웃
              </h1>
            </div>

            {/* Profile Menu Button */}
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
        </header>
      )}

      {/* Main Content */}
      <main className={`flex-1 overflow-auto ${isCheckoutPage ? 'pb-0 lg:pb-0' : 'pb-16 lg:pb-0'}`}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/checkout" element={<CheckoutPage />} />
          <Route path="/validate" element={<ValidatePage />} />
          <Route path="/mypage" element={<MyPage />} />
          <Route path="/products" element={<ProductsPage />} />
          <Route path="/admin/purchases" element={<AdminPurchasesPage />} />
        </Routes>
      </main>

      {/* Mobile Bottom Navigation - Only on mobile */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-[var(--color-border)] safe-area-pb">
        <div className="grid grid-cols-4 h-16">
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
                <span className="text-xl">{item.icon}</span>
                <span className="text-xs font-medium">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
