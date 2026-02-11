import { Routes, Route, Link, useLocation } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CheckoutPage from "./pages/CheckoutPage";
import ProductsPage from "./pages/ProductsPage";
import ValidatePage from "./pages/ValidatePage";

const NAV_ITEMS = [
  { path: "/", label: "홈", icon: "🏠" },
  { path: "/checkout", label: "체크아웃", icon: "🛒" },
  { path: "/products", label: "상품 관리", icon: "📦" },
  { path: "/validate", label: "영수증 확인", icon: "📋" },
];

export default function App() {
  const { pathname } = useLocation();
  const isCheckoutPage = pathname === "/checkout";

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
      </aside>

      {/* Mobile Header - Hidden on checkout page */}
      {!isCheckoutPage && (
        <header className="lg:hidden bg-white border-b border-[var(--color-border)] px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white text-lg">
              🏪
            </div>
            <h1 className="text-base font-bold text-[var(--color-text)]">
              스마트 체크아웃
            </h1>
          </div>
        </header>
      )}

      {/* Main Content */}
      <main className={`flex-1 overflow-auto ${isCheckoutPage ? 'pb-0 lg:pb-0' : 'pb-16 lg:pb-0'}`}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/checkout" element={<CheckoutPage />} />
          <Route path="/products" element={<ProductsPage />} />
          <Route path="/validate" element={<ValidatePage />} />
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
