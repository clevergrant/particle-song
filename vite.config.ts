import { defineConfig } from "vite";

export default defineConfig({
  // AudioWorklet requires ES-module scripts. Vite's default worker format
  // is 'iife', which won't load via `audioWorklet.addModule`. This flag
  // affects both ?worker imports and ?worker&url URL imports.
  worker: {
    format: "es",
  },
  server: {
    port: 80,
    hmr: {
      overlay: false,
    },
  },
});
