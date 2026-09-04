import { defineConfig, devices } from "@playwright/test";

// Two modes:
//
// 1. Default (no CDP_ENDPOINT): boot smoke test. Builds and previews the app in
//    place and asserts the shell renders without uncaught errors. Runs headless,
//    so it does NOT exercise segmentation or meshing (those need WebGPU).
//
// 2. CDP mode (CDP_ENDPOINT set): the full segment-and-mesh pipeline test in
//    e2e/pipeline.gpu.spec.js connects to a real WebGPU browser over CDP. The
//    app server is started separately (see e2e/README.md), so no webServer here.
const cdp = process.env.CDP_ENDPOINT;

export default defineConfig({
  testDir: "./e2e",
  testMatch: cdp ? /pipeline\.gpu\.spec\.js/ : /boot\.spec\.js/,
  fullyParallel: true,
  reporter: "list",
  use: {
    baseURL: cdp ? (process.env.APP_URL ?? "http://localhost:5173") : "http://localhost:4173",
    trace: "on-first-retry",
  },
  webServer: cdp
    ? undefined
    : {
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
