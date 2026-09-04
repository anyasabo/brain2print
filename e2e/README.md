# End-to-end tests

Two Playwright specs, run by two modes of the same config.

## boot.spec.js — headless boot check

```bash
npm run test:e2e
```

Builds and previews the app, then asserts the shell renders with no unexpected
console errors. It runs in the sandbox's headless Chromium, which has no WebGPU,
so it does not run segmentation or mesh generation.

## pipeline.gpu.spec.js — full segment-and-mesh check

This spec runs the real pipeline: load the default T1 volume, segment with a
model, then create a mesh. It needs a real WebGPU browser. The sandbox has none,
so it drives a browser on your own machine over the Chrome DevTools Protocol
(CDP).

Setup, one time per run:

1. On your machine, start Chrome (or Edge) with remote debugging:

   ```bash
   # macOS
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
     --remote-debugging-port=9222 --user-data-dir=/tmp/cdp-profile
   # Linux
   google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/cdp-profile
   ```

   To confirm it works, open `http://localhost:9222/json/version` on your
   machine. It returns JSON. Also confirm the browser reports WebGPU: open the
   app page and confirm that `navigator.gpu` is defined.

2. Reverse-forward that port into the devspace, so the devspace can reach your
   browser at `localhost:9222`. This is a Coder devspace, so use `coder ssh`:

   ```bash
   coder ssh <workspace-name> -- -R 9222:localhost:9222
   ```

   For a plain SSH host, the form is `ssh -R 9222:localhost:9222 <host>`. The
   devspace exempts `127.0.0.1`/`localhost` from the egress proxy, so this
   connection is direct.

3. In the sandbox, start the dev server and note its URL (default
   `http://localhost:5173`):

   ```bash
   npm run dev
   ```

4. In the sandbox, run the spec, pointing it at the forwarded browser and the
   dev server:

   ```bash
   CDP_ENDPOINT=http://localhost:9222 APP_URL=http://localhost:5173 npm run test:e2e:cdp
   ```

Without `CDP_ENDPOINT`, the spec skips itself. So `npm run test:e2e` never runs
it.
