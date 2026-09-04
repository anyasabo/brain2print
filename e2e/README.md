# End-to-end tests

## boot.spec.js — headless boot check

```bash
npm run test:e2e
```

Builds and previews the app, then asserts the shell renders with no unexpected
console errors. It runs in the sandbox's headless Chromium, which has no WebGPU,
so it does not run segmentation or mesh generation.

## Full segment-and-mesh check — by hand, in your own browser

The pipeline needs WebGPU, which the headless test environment lacks. To check
the full flow, open the app in a WebGPU browser on your own machine:

1. Start the dev server:

   ```bash
   npm run dev
   ```

2. Open the forwarded URL in your browser. In a Coder devspace or VS Code
   remote, the port is forwarded automatically. The printed `localhost` URL
   opens on your machine.

3. In the app: pick a `Segmentation Model` (Tissue GWM is a good default),
   press `Create Mesh`, choose settings, and press `Apply`. Check that the mesh
   appears with no console errors.
