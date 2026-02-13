import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "../api/client";

interface AuthState {
  token: string | null;
  user: User | null;
  setAuth: (token: string, user: User) => void;
  clearAuth: () => void;
  isAuthenticated: () => boolean;
  isAdmin: () => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,

      setAuth: (token, user) => set({ token, user }),

      clearAuth: () => set({ token: null, user: null }),

      isAuthenticated: () => {
        const { token, user } = get();
        return token !== null && user !== null;
      },

      isAdmin: () => {
        const { user } = get();
        return user?.role === "admin";
      },
    }),
    {
      name: "ebrcs-auth", // localStorage key
    }
  )
);
