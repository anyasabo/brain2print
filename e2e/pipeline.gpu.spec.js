import { expect, test } from "@playwright/test";

// Full segment-and-mesh pipeline test. It needs a real WebGPU browser, which
// the sandbox does not have. Run it against a WebGPU browser on your own
// machine, reverse-forwarded over SSH. See e2e/README.md for the setup.
//
//   CDP_ENDPOINT=http://localhost:9222 npm run test:e2e:cdp
//
// Skipped unless CDP_ENDPOINT is set.
const CDP = process.env.CDP_ENDPOINT;

test.skip(!CDP, "set CDP_ENDPOINT to a WebGPU browser to run this");

test("segment then create mesh in a real WebGPU browser", async () => {
  const { chromium } = await import("@playwright/test");
  const browser = await chromium.connectOverCDP(CDP);
  const context = browser.contexts()[0] ?? (await browser.newContext());
  const page = await context.newPage();
  const baseURL = process.env.APP_URL ?? "http://localhost:5173";

  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto(baseURL);

  // WebGPU must actually be present, or this test is meaningless.
  const hasGpu = await page.evaluate(() => "gpu" in navigator);
  expect(hasGpu, "navigator.gpu missing — not a WebGPU browser").toBe(true);

  // The default T1 volume loads and populates the model dropdown.
  await expect
    .poll(() => page.locator("#modelSelect option").count(), { timeout: 30_000 })
    .toBeGreaterThan(0);

  // Pick the Tissue GWM model (a good default per the README) and segment.
  await page.selectOption("#modelSelect", { label: /Tissue GWM/i });

  // Segmentation adds an overlay volume; the save button enables when it lands.
  await expect(page.locator("#saveBtn")).toBeEnabled({ timeout: 120_000 });
  await expect(page.locator("#createMeshBtn")).toBeEnabled();

  // Create Mesh on the faster path, then apply.
  await page.locator("#createMeshBtn").click();
  await page.locator("#applyBtn").click();

  // A mesh loads; the mesh-save button enables.
  await expect(page.locator("#saveMeshBtn")).toBeEnabled({ timeout: 120_000 });

  expect(errors, `page errors:\n${errors.join("\n")}`).toEqual([]);

  await page.close();
  await browser.close();
});
