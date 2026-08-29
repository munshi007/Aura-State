import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: { outDir: "../aura_state/ui/static", emptyOutDir: true, assetsDir: "assets" },
  server: { proxy: { "/api": "http://127.0.0.1:8155" } },
});
