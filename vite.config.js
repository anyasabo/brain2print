import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  // root: '.',
  base: "./",
  server: {
    open: "index.html",
  },
  // Do not auto-open a browser for `vite preview` (used by tests and CI).
  preview: {
    open: false,
  },
  worker: {
    format: "esm",
  },
  // exclude @niivue/niimath from optimization
  optimizeDeps: {
    exclude: [
      "@niivue/niimath",
      "@itk-wasm/cuberille",
      "@itk-wasm/mesh-filters",
    ],
  },
  plugins: [
    // put lazy loaded JavaScript and Wasm bundles in dist directory
    viteStaticCopy({
      // vite-plugin-static-copy v4 preserves the matched source directory
      // structure by default; stripBase flattens the files into dest/ as v2 did.
      targets: [
        {
          src: "node_modules/@itk-wasm/cuberille/dist/pipelines/*.{js,wasm,wasm.zst}",
          dest: "pipelines",
          rename: { stripBase: true },
        },
        {
          src: "node_modules/@itk-wasm/mesh-filters/dist/pipelines/*.{js,wasm,wasm.zst}",
          dest: "pipelines",
          rename: { stripBase: true },
        },
      ],
    }),
    VitePWA({
      registerType: "autoUpdate",
      // manifest.json is maintained by hand in public/, so let the plugin use it.
      manifest: false,
      includeAssets: [
        "manifest.json",
        "niivue.css",
        "t1_crop.nii.gz",
        "models/**",
        "pipelines/**",
      ],
      workbox: {
        globPatterns: ["**/*.{js,css,html,wasm,zst,nii.gz,json,png,ico}"],
        // The brainchop web worker bundle is ~1.7 MB; raise the default 2 MB cap.
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
      },
    }),
  ],
});
