import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  // GitHub Pages base path (저장소 이름과 동일하게 설정)
  base: process.env.NODE_ENV === 'production' ? '/EBRCS_streaming/' : '/',

  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,  // WebSocket 지원 추가!
      },
    },
  },
});
