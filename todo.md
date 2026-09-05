# TODO

Future work, roughly in priority order. See `AGENTS.md` for how to run and test
the app, and `upstream-potential.md` for things to raise with niivue upstream.

## Batch model comparison

Give the tool a NIfTI file and run it against every segmentation model, then
emit one mesh (STL) per model so a user can compare which gives the best result.
Extend it to sweep mesh parameters (smoothing iterations, shrink percent) so a
user can tune the output.

- Reuse the existing UI pipeline (`modelSelect.onchange` then the apply path in
  `main.ts`), do not fork the segmentation or meshing logic.
- Drive it headless with Playwright against the built app on a GPU machine (or a
  sandbox with a large-enough software texture limit). Save each mesh named by
  model and parameters.
- Record per-model runtime and pass/fail. That output feeds the GPU-compat
  estimates below.
- brainchop model definitions live in `~/brainchop` (the model source) and in
  `brainchop-parameters.ts`.

## Up-front GPU-compatibility feedback

The README says not all models work with all graphics cards. Today the app finds
out only after a model is selected and inference fails with a memory or
texture-size error (see the error paths in `brainchop-webworker.ts`).

- At load time, read the WebGPU adapter limits (`maxTextureDimension2D/3D`,
  `maxBufferSize`) or the WebGL2 max texture size, and compare each model's
  estimated need.
- Chosen UX: annotate risky models in the dropdown and surface the existing
  per-model `warning` proactively, but keep them selectable (avoid
  false-negatives that hide a working setup).
- `~/birdnet-go` has prior art for small-runtime model execution (a backend
  abstraction, thread tuning, quantization). Server-side Go, not browser JS, but
  the capability-probing idea transfers.

## niivue v1: move off the release candidate

The app is on `@niivue/niivue` and `@niivue/nv-ext-image-processing` at
`1.0.0-rc.13`. Bump both to the stable 1.0 when it ships. The two versions must
match exactly (pinned peer dependency).

## Reevaluate dependencies and architecture

Now that the app is modernized and typed, weigh larger changes:

- Adopt more of the NiiVue v1 extension model. `nv-ext-niimath` and
  `nv-ext-image-processing` may cover work now done through direct itk-wasm and
  niimath calls.
- Align tooling with the niivue mono repo (it uses Biome and bun).
- Consider a small typed component layer instead of the id-as-global DOM
  structure in `main.ts`.

## Tooling

- Add a PR CI workflow (lint, typecheck, unit test, build) if this fork becomes
  actively maintained. Deliberately skipped for now.
- The Playwright boot test is headless-only. The full segment-and-mesh flow
  needs a manual check in a GPU browser (see `AGENTS.md`).
