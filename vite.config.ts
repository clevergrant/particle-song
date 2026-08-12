import { defineConfig, type Plugin } from "vite";
import { appendFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";

/**
 * Dev-server middleware that accepts POSTed music-log lines and appends
 * them to music-log.jsonl at the project root. One request may contain
 * a batch of records (one JSON object per line).
 *
 * Endpoints (dev-only):
 *   POST /__music-log         body = raw text, each line already JSON
 *   POST /__music-log/reset   truncates the file
 */
function musicLogPlugin(): Plugin {
  const logPath = resolve(__dirname, "music-log.jsonl");

  const ensureFile = () => {
    const dir = dirname(logPath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    if (!existsSync(logPath)) writeFileSync(logPath, "");
  };

  return {
    name: "music-log",
    apply: "serve",
    configureServer(server) {
      ensureFile();
      server.middlewares.use((req, res, next) => {
        if (!req.url || req.method !== "POST") return next();
        if (req.url !== "/__music-log" && req.url !== "/__music-log/reset") {
          return next();
        }
        if (req.url === "/__music-log/reset") {
          try {
            writeFileSync(logPath, "");
            res.statusCode = 204;
            res.end();
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
          return;
        }
        let body = "";
        req.on("data", (chunk) => { body += chunk; });
        req.on("end", () => {
          try {
            if (body.length > 0) {
              const suffix = body.endsWith("\n") ? "" : "\n";
              appendFileSync(logPath, body + suffix);
            }
            res.statusCode = 204;
            res.end();
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [musicLogPlugin()],
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
