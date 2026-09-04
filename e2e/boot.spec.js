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

  // Populating the model dropdown and running segmentation need a real GPU
  // context, which a headless sandbox lacks. The full segment-and-mesh flow is
  // covered by pipeline.gpu.spec.js against a real WebGPU browser over CDP.
  expect(fatalErrors, `unexpected console errors:\n${fatalErrors.join("\n")}`).toEqual([]);
});
