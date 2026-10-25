import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the API runs on :8000 (uvicorn) and Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist" },
});
