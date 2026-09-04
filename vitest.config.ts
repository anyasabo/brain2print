import { defineConfig } from "vitest/config";

// Unit tests only. Playwright specs under e2e/ run via `npm run test:e2e`.
export default defineConfig({
  test: {
    include: ["**/*.test.{js,ts}"],
    exclude: ["node_modules", "dist", "e2e"],
  },
});
