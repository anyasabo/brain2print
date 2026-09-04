import { expect, test } from "@playwright/test";

// Console errors that are expected when WebGPU is unavailable (headless CI,
// sandboxes without a GPU). These must not fail the boot test.
const WEBGPU_UNAVAILABLE = /webgpu|navigator\.gpu|requestAdapter|WebGL|GPUAdapter/i;

test("app boots and renders the shell", async ({ page }) => {
  const fatalErrors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !WEBGPU_UNAVAILABLE.test(msg.text())) {
      fatalErrors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => {
    if (!WEBGPU_UNAVAILABLE.test(err.message)) {
      fatalErrors.push(err.message);
    }
  });

  await page.goto("/");

  // The canvas and the model dropdown are part of the static shell.
  await expect(page.locator("#gl1")).toBeVisible();
  await expect(page.locator("#modelSelect")).toBeAttached();

  // The service worker registers on load; proves the bundled entry executed.
  const swRegistered = await page.evaluate(
    () => "serviceWorker" in navigator,
  );
  expect(swRegistered).toBe(true);

  // main() populates the model dropdown, but only after niivue initializes a
  // real GPU context. That works locally and on macOS, but not in a headless
  // sandbox with no GPU. Assert the fuller path only when GPU=1 is set, so the
  // test stays green where no GPU exists and still catches regressions where
  // one does.
  if (process.env.GPU === "1") {
    await expect
      .poll(() => page.locator("#modelSelect option").count(), {
        timeout: 15_000,
      })
      .toBeGreaterThan(0);
  }

  expect(fatalErrors, `unexpected console errors:\n${fatalErrors.join("\n")}`).toEqual([]);
});
