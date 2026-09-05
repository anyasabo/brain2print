# Upstream-potential notes

Things found while working on this fork that are worth raising with upstream
projects (niivue, brainchop). Not commitments. A running list, so the context is
not lost.

## niivue: WebGPU canvas is not readable for screenshots

Found during the niivue v1 migration (2026-09-04).

**What:** With the WebGPU backend, a canvas screenshot comes out blank, and
`canvas.toDataURL()` / drawing the canvas into a 2D context reads transparent
pixels. WebGL2 captures fine. Rendering to the screen works on both backends.
Only the readback is blank.

**Why:** niivue configures the WebGPU canvas context without `COPY_SRC` usage:

```
// packages/niivue/src/wgpu/NVViewGPU.ts
this.context.configure({
  device: this.device,
  format: this.preferredCanvasFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  alphaMode: 'premultiplied',
})
```

Without `GPUTextureUsage.COPY_SRC`, the presented swapchain texture cannot be
copied out (`copyTextureToBuffer`), so headless capture and `toDataURL` return
nothing.

**Impact:** Blocks automated visual/screenshot testing of WebGPU rendering in
CI. Teams fall back to the WebGL2 backend to get screenshots, which does not
exercise the WebGPU path.

**Possible fix to propose:** add `GPUTextureUsage.COPY_SRC` to the context
`usage`, or expose a `screenshot()` / `saveScene()` method that does a proper
GPU-to-CPU readback (older niivue had a screenshot helper). A version behind an
option avoids any cost for users who do not need readback.

**Repro:** load a volume with the WebGPU backend, then call
`canvas.toDataURL()` or Playwright `page.screenshot()`. The canvas region is
blank, while the same scene on WebGL2 captures correctly.

## Open upstream PRs worth tracking (niivue/brain2print)

- PR #14 (thewtex): all-labels surface + quality extraction. Already ported into
  this fork's meshing path.
- PR #8 (yarikoptic): codespell config. Not adopted here. Low-risk quality
  tooling for upstream to merge.
