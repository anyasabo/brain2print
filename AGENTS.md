# Agent guide

brain2print is a browser-only app: it segments an MRI in the browser (brainchop
on TensorFlow.js) and turns the result into a printable 3D mesh (ITK-Wasm +
niimath), rendered with NiiVue. No server, no data upload.

## Stack

- Vanilla TypeScript, built with Vite. No framework.
- Rendering: `@niivue/niivue` (v1). Segmentation: `@tensorflow/tfjs`. Meshing:
  `@itk-wasm/cuberille`, `@itk-wasm/mesh-filters`, `@niivue/niimath`.
- `main.ts` is the entry. It wires DOM controls (referenced as id-globals, typed
  in `globals.d.ts`) to a single NiiVue instance and a segmentation web worker
  (`brainchop-webworker.ts`).
- `bwlabels.ts` (connected-components) is pure and unit-tested. The rest of the
  tfjs code is in `tensor-utils.ts` and the worker.

## Commands

```bash
npm run dev         # Vite dev server (hot reload)
npm run build       # production build to dist/
npm run preview      # serve the built dist/
npm run lint        # Biome lint + format check
npm run format      # Biome format --write
npm run typecheck   # tsc --noEmit (strict)
npm run test        # Vitest unit tests
npm run test:e2e    # Playwright headless boot smoke test
```

Run `npm run typecheck` and `npm run build` before committing. The strict types
catch real bugs here, like missing awaits and swapped args. Do not paper over
them with `any` unless a third-party type is genuinely wrong.

## Rendering needs a GPU

The full pipeline (segmentation, meshing, 3D render) needs WebGPU or WebGL2.
The default backend is WebGPU. Append `?backend=webgl2` to the URL to force
WebGL2. Where WebGPU is unavailable, NiiVue falls back to WebGL2 on its own.

`npm run test:e2e` only asserts the app boots without errors. It does not
exercise the pipeline. To test the real pipeline, open `npm run dev` in a
browser with a GPU. A Coder devspace or VS Code remote forwards the port to your
local browser automatically.

### Rendering headless in a CPU-only sandbox

A sandbox with no GPU can still render through software rasterizers, under a
virtual display. Use it to screenshot-debug the app without a browser. Drive the
built app with vite's `preview()` and Playwright's `page.screenshot()`, launched
under `xvfb-run`:

```bash
xvfb-run -a --server-args="-screen 0 1400x1000x24" node script.mjs
```

For reliable screenshots, use the WebGL2 backend (`?backend=webgl2`) through
SwiftShader:

```js
chromium.launch({ channel: "chromium", args: [
  "--no-sandbox", "--ignore-gpu-blocklist",
  "--use-gl=angle", "--use-angle=swiftshader",
  "--disable-software-rasterizer=false",
]})
```

WebGPU also runs headless here (Mesa lavapipe / SwiftShader Vulkan) with these
flags:

```js
args: ["--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
       "--enable-features=Vulkan", "--use-angle=vulkan"]
```

Two caveats for WebGPU headless. First, `navigator.gpu` is `undefined` on
`about:blank`, so navigate to the served page (a secure `http://localhost`
context) before probing. Second, a WebGPU canvas screenshot comes out blank
here, and `canvas.toDataURL` / `drawImage` read transparent pixels. The cause is
niivue's WebGPU swapchain: it configures the context with `usage:
RENDER_ATTACHMENT | COPY_DST` (no `COPY_SRC`), so the presented texture is not
readable. This is internal to niivue and cannot be changed from the app.

So to verify a rendering change, screenshot with the WebGL2 backend
(`?backend=webgl2`). WebGL2 and WebGPU share niivue's scene and layout logic and
only differ in the GPU path, so the WebGL2 picture is a faithful proxy for what
WebGPU draws. Verify WebGPU separately by its console line (`niivue-info WebGPU
via ...`) and the absence of errors. That pair catches both classes of bug seen
during the v1 migration: layout/render regressions show up in the WebGL2 image,
and WebGPU-specific failures (a lost adapter, a texture-size overflow) show up
in the WebGPU console.

SwiftShader's max texture is 8192 (real GPUs are 16384+). Large-volume
segmentation hits a texture-size limit that a real GPU clears. That ceiling is a
sandbox artifact, not a bug.

## niivue v1 notes

v1 (the `niivue/mono` monorepo, currently `1.0.0-rc.13`) is a large API change
from 0.x. Key differences this app relies on:

- `NiiVue` is the default export (not `Niivue`). `NVImage`/`NVMesh` are plain
  types with no instance methods.
- `conform()` moved out of core into `@niivue/nv-ext-image-processing`. Register
  it via `nv.createExtensionContext().registerVolumeTransform(conform)` and call
  `ctx.applyVolumeTransform("conform", vol, opts)`.
- Build a volume from raw header + image with the exported `nii2volume(hdr,
  img)`. Add a computed mesh as a File via `nv.addMesh({ url: new File([...],
  "mesh.obj") })`. v1 reads OBJ/STL/PLY/MZ3/GII. Save with `nv.saveMesh(index,
  filename)`.
- `attachToCanvas` is async, so await it. Size the canvas via CSS
  (`width/height: 100%`). v1 reads the drawing-buffer size from the CSS box.
  0.x set it internally instead.
- Per-volume opacity: set `volume.opacity` then `nv.updateGLVolume()`.
  `removeMesh`/`removeVolume` take an index. `colormaps` is a getter.
  `on*` callbacks became `addEventListener(...)`.

The `nv-ext-image-processing` version must match the niivue version exactly (its
peer dependency is pinned).

## Sandbox / lockfile

The repo uses a committed `.npmrc` with `omit-lockfile-registry-resolved=true`
so `package-lock.json` records no registry URLs. Keep it. Do not commit registry
URLs, proxy settings, or any host-specific configuration.
