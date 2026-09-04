import { defineConfig, devices } from "@playwright/test";

// Boot smoke test only. It loads the built app and asserts the shell renders
// without uncaught errors. It does NOT exercise segmentation or meshing, which
// need WebGPU — unavailable in headless CI, so run this locally or on macOS.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  reporter: "list",
  use: {
    baseURL: "http://localhost:4173",
    trace: "on-first-retry",
  },
  webServer: {
    command: "npm run build && npm run preview",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
