import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("cytoscape")) {
            return "cytoscape";
          }
          if (id.includes("react")) {
            return "react-vendor";
          }
          return undefined;
        }
      }
    }
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/search": "http://127.0.0.1:8000",
      "/answer": "http://127.0.0.1:8000",
      "/graph": "http://127.0.0.1:8000"
    }
  }
});
