import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import { signup } from "../api/client";

export default function SignupPage() {
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState<"user" | "admin">("user");

  const mutation = useMutation({
    mutationFn: () => signup({ username, password, name, role }),
    onSuccess: () => {
      alert("회원가입이 완료되었습니다! 로그인해주세요.");
      navigate("/login");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-bg)] py-12">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-2xl shadow-xl border border-[var(--color-border)]">
        {/* Logo */}
        <div className="flex flex-col items-center">
          <div className="w-16 h-16 rounded-2xl bg-[var(--color-primary)] flex items-center justify-center text-white text-3xl mb-4">
            🏪
          </div>
          <h2 className="text-center text-3xl font-bold text-[var(--color-text)]">
            회원가입
          </h2>
          <p className="mt-2 text-center text-sm text-[var(--color-text-secondary)]">
            EBRCS 스마트 체크아웃 시스템
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-[var(--color-text)]">
                이름
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 block w-full px-4 py-2.5 border border-[var(--color-border)] rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)] transition-colors"
                placeholder="홍길동"
              />
            </div>

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
                placeholder="admin"
                minLength={3}
                maxLength={20}
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
                minLength={4}
                maxLength={50}
              />
              <p className="mt-1 text-xs text-[var(--color-text-secondary)]">4~50자</p>
            </div>

            <div>
              <label htmlFor="role" className="block text-sm font-medium text-[var(--color-text)]">
                역할
              </label>
              <select
                id="role"
                name="role"
                value={role}
                onChange={(e) => setRole(e.target.value as "user" | "admin")}
                className="mt-1 block w-full px-4 py-2.5 border border-[var(--color-border)] rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)] transition-colors bg-white"
              >
                <option value="user">일반 사용자</option>
                <option value="admin">관리자</option>
              </select>
            </div>
          </div>

          {mutation.isError && (
            <div className="text-sm text-[var(--color-danger)] bg-red-50 p-3 rounded-lg border border-red-100">
              {(mutation.error as Error).message}
            </div>
          )}

          {mutation.isSuccess && (
            <div className="text-sm text-[var(--color-success)] bg-green-50 p-3 rounded-lg border border-green-100">
              회원가입 성공! 로그인 페이지로 이동합니다...
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={mutation.isPending}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-xl shadow-sm text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {mutation.isPending ? "가입 중..." : "회원가입"}
            </button>
          </div>

          <div className="text-center text-sm">
            <span className="text-[var(--color-text-secondary)]">이미 계정이 있으신가요? </span>
            <Link to="/login" className="font-semibold text-[var(--color-primary)] hover:text-[var(--color-primary-hover)] transition-colors">
              로그인
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}
