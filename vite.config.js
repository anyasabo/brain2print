import { defineConfig } from "vite";
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  // root: '.',
  base: "./",
  server: {
    open: "index.html",
  },
  worker: {
    format: "esm",
  },
  // exclude @niivue/niimath from optimization
  optimizeDeps: {
    exclude: ["@niivue/niimath", "@itk-wasm/cuberille", "@itk-wasm/mesh-filters"],
  },
  plugins: [
    // put lazy loaded JavaScript and Wasm bundles in dist directory
    viteStaticCopy({
      // vite-plugin-static-copy v4 preserves the matched source directory
      // structure by default; stripBase flattens the files into dest/ as v2 did.
      targets: [
        { src: 'node_modules/@itk-wasm/cuberille/dist/pipelines/*.{js,wasm,wasm.zst}', dest: 'pipelines', rename: { stripBase: true } },
        { src: 'node_modules/@itk-wasm/mesh-filters/dist/pipelines/*.{js,wasm,wasm.zst}', dest: 'pipelines', rename: { stripBase: true } },
      ],
    })
  ],
});
