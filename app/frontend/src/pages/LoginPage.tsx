import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import { login } from "../api/auth";
import { useAuthStore } from "../stores/authStore";

export default function LoginPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const mutation = useMutation({
    mutationFn: () => login(username, password),
    onSuccess: (data) => {
      setAuth(data.access_token, data.user);
      navigate("/"); // Redirect to home after login
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-bg)]">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-2xl shadow-xl border border-[var(--color-border)]">
        {/* Logo */}
        <div className="flex flex-col items-center">
          <img
            src="/jangbogo.svg"
            alt="장보GO 로고"
            className="w-16 h-16 rounded-2xl object-cover mb-4"
          />
          <h2 className="text-center text-3xl font-bold text-[var(--color-text)]">
            장보GO!
          </h2>
          <p className="mt-2 text-center text-sm text-[var(--color-text-secondary)]">
            로그인
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-[var(--color-text)]">
                아이디
              </label>
              <input
                id="username"
                name="username"
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="mt-1 block w-full px-4 py-2.5 border border-[var(--color-border)] rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)] transition-colors"
                placeholder="아이디를 입력하세요..."
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-[var(--color-text)]">
                비밀번호
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full px-4 py-2.5 border border-[var(--color-border)] rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)] transition-colors"
                placeholder="••••••••"
              />
            </div>
          </div>

          {mutation.isError && (
            <div className="text-sm text-[var(--color-danger)] bg-red-50 p-3 rounded-lg border border-red-100">
              {(mutation.error as Error).message}
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={mutation.isPending}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-xl shadow-sm text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {mutation.isPending ? "로그인 중..." : "로그인"}
            </button>
          </div>

          <div className="text-center text-sm">
            <span className="text-[var(--color-text-secondary)]">계정이 없으신가요? </span>
            <Link to="/signup" className="font-semibold text-[var(--color-primary)] hover:text-[var(--color-primary-hover)] transition-colors">
              회원가입
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}
