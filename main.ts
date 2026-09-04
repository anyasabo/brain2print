import {
  antiAliasCuberille,
  cuberille,
  setPipelinesBaseUrl as setCuberillePipelinesUrl,
} from "@itk-wasm/cuberille";
import {
  keepLargestComponent,
  repair,
  setPipelinesBaseUrl as setMeshFiltersPipelinesUrl,
  smoothRemesh,
} from "@itk-wasm/mesh-filters";
import { iwm2meshCore, nii2iwi } from "@niivue/cbor-loader";
import { Niimath } from "@niivue/niimath";
import { NiiVue, nii2volume, SHOW_RENDER, SLICE_TYPE } from "@niivue/niivue";
import { conform } from "@niivue/nv-ext-image-processing";
import { brainChopOpts, inferenceModelsList } from "./brainchop-parameters.js";
import { isChrome, localSystemDetails } from "./brainchop-telemetry.js";
import MyWorker from "./brainchop-webworker.js?worker";
import { positionsIndicesToObj } from "./mesh-io.js";

// Use local, vendored WebAssembly module assets
const viteBaseUrl = import.meta.env.BASE_URL;
const pipelinesBaseUrl = new URL(
  `${viteBaseUrl}pipelines`,
  document.location.origin,
).href;
setCuberillePipelinesUrl(pipelinesBaseUrl);
setMeshFiltersPipelinesUrl(pipelinesBaseUrl);

import { registerSW } from "virtual:pwa-register";

// Offline support: vite-plugin-pwa generates and precaches a service worker
// (see vite.config.js). Everything runs client-side, so a cached install keeps
// working with no network — which matters for the privacy of MRI data.
registerSW({ immediate: true });

async function main() {
  let chopWorker: Worker | undefined;
  let extCtx: ReturnType<NiiVue["createExtensionContext"]>;
  const niimath = new Niimath();
  await niimath.init();
  niimath.setOutputDataType("input"); // call before setting image since this is passed to the image constructor
  aboutBtn.onclick = () => {
    const url = "https://github.com/niivue/brain2print";
    window.open(url, "_blank");
  };
  function updateBackgroundOpacity() {
    // v1 dropped setOpacity(idx, v); set the volume's opacity field directly.
    nv1.volumes[0].opacity = Number(opacitySlider0.value) / 255;
    nv1.updateGLVolume();
  }
  opacitySlider0.oninput = updateBackgroundOpacity;
  opacitySlider1.oninput = () => {
    if (nv1.volumes.length < 2) return;
    nv1.volumes[1].opacity = Number(opacitySlider1.value) / 255;
    nv1.updateGLVolume();
  };
  async function ensureConformed() {
    const nii = nv1.volumes[0];
    const dims = nii.dims;
    const permRAS = nii.permRAS;
    let isConformed =
      dims !== undefined &&
      permRAS !== undefined &&
      dims[1] === 256 &&
      dims[2] === 256 &&
      dims[3] === 256 &&
      nii.img instanceof Uint8Array &&
      nii.img.length === 256 * 256 * 256;
    if (
      permRAS === undefined ||
      permRAS[0] !== -1 ||
      permRAS[1] !== 3 ||
      permRAS[2] !== -2
    )
      isConformed = false;
    if (isConformed) return;
    // v1: conform lives in @niivue/nv-ext-image-processing. Old positional call
    // conform(nii, false, true, false, true) = toRAS:false, isLinear:true,
    // asFloat32:false, isRobustMinMax:true.
    const nii2 = await extCtx.applyVolumeTransform("conform", nii, {
      toRAS: false,
      isLinear: true,
      asFloat32: false,
      isRobustMinMax: true,
    });
    await nv1.removeVolume(0);
    await nv1.addVolume(nii2);
  }
  async function closeAllOverlays() {
    while (nv1.volumes.length > 1) {
      await nv1.removeVolume(1);
    }
  }
  modelSelect.onchange = async () => {
    if (modelSelect.selectedIndex < 0) modelSelect.selectedIndex = 11;
    await closeAllOverlays();
    await ensureConformed();
    // brainchop-parameters.js entries are augmented at runtime with these flags.
    const model = inferenceModelsList[
      modelSelect.selectedIndex
    ] as (typeof inferenceModelsList)[number] & {
      isNvidia: boolean;
      isScalar: boolean;
    };
    model.isNvidia = false;
    model.isScalar = scalarCheck.checked;
    // v1 no longer exposes nv1.gl; probe the renderer with a throwaway context.
    const probeGl = document.createElement("canvas").getContext("webgl2");
    const rendererInfo = probeGl?.getExtension("WEBGL_debug_renderer_info");
    if (probeGl && rendererInfo) {
      model.isNvidia = (
        probeGl.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL) as string
      ).includes("NVIDIA");
    }
    const opts = brainChopOpts as typeof brainChopOpts & { rootURL: string };
    opts.rootURL = location.href;
    const isLocalhost = Boolean(
      window.location.hostname === "localhost" ||
        // [::1] is the IPv6 localhost address.
        window.location.hostname === "[::1]" ||
        // 127.0.0.1/8 is considered localhost for IPv4.
        window.location.hostname.match(
          /^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/,
        ),
    );
    if (isLocalhost) {
      opts.rootURL = location.protocol + "//" + location.host;
    }
    if (typeof chopWorker !== "undefined") {
      console.log(
        "Unable to start new segmentation: previous call has not completed",
      );
      return;
    }
    const worker = new MyWorker();
    chopWorker = worker;
    const vol0 = nv1.volumes[0];
    const hdr = {
      datatypeCode: vol0.hdr?.datatypeCode,
      dims: vol0.hdr?.dims,
    };
    const msg = {
      opts: opts,
      modelEntry: model,
      niftiHeader: hdr,
      niftiImage: vol0.img,
    };
    worker.postMessage(msg);
    worker.onmessage = (event) => {
      const cmd = event.data.cmd;
      if (cmd === "ui") {
        if (event.data.modalMessage !== "") {
          worker.terminate();
          chopWorker = undefined;
        }
        callbackUI(
          event.data.message,
          event.data.progressFrac,
          event.data.modalMessage,
        );
      }
      if (cmd === "img") {
        worker.terminate();
        chopWorker = undefined;
        callbackImg(event.data.img, event.data.opts, event.data.modelEntry);
      }
    };
  };
  saveBtn.onclick = () => {
    // v1: NVImage lost saveToDisk(); save the overlay volume by index.
    nv1.saveVolume({ filename: "Custom.nii", volumeByIndex: 1 });
  };
  clipCheck.onchange = () => {
    if (clipCheck.checked) {
      nv1.setClipPlane([0, 0, 90]);
    } else {
      nv1.setClipPlane([2, 0, 90]);
    }
  };
  scalarCheck.onchange = () => {
    modelSelect.selectedIndex = -1;
  };
  function doLoadImage() {
    saveBtn.disabled = true;
    updateBackgroundOpacity();
  }
  async function fetchJSON(fnm: string) {
    const response = await fetch(fnm);
    const js = await response.json();
    return js;
  }
  async function callbackImg(
    img: ArrayBufferLike,
    opts: { atlasSelectedColorTable: string },
    modelEntry: { isScalar?: boolean; colormapPath?: string },
  ) {
    await closeAllOverlays();
    const bgHdr = nv1.volumes[0].hdr;
    if (!bgHdr) return;
    // v1: NVImage lost clone()/zeroImage(); build the overlay from the
    // background geometry plus the segmentation label image via nii2volume.
    const overlayHdr = JSON.parse(JSON.stringify(bgHdr));
    overlayHdr.scl_inter = 0;
    overlayHdr.scl_slope = 1;
    const isScalar = modelEntry.isScalar === true;
    const useLabelColormap = !isScalar && Boolean(modelEntry.colormapPath);
    if (isScalar) {
      overlayHdr.scl_slope = 1 / 255;
    } else if (useLabelColormap) {
      // n.b. most models create indexed labels, but those without colormap mask scalar input
      overlayHdr.intent_code = 1002; // NIFTI_INTENT_LABEL
    }
    const overlayVolume = nii2volume(
      overlayHdr,
      new Uint8Array(img),
      "overlay",
    );
    if (isScalar) {
      overlayVolume.colormap = "viridis";
    } else if (!useLabelColormap) {
      let colormap = opts.atlasSelectedColorTable.toLowerCase();
      const cmaps = nv1.colormaps;
      if (!cmaps.includes(colormap)) {
        colormap = "actc";
      }
      overlayVolume.colormap = colormap;
    }
    overlayVolume.opacity = Number(opacitySlider1.value) / 255;
    await nv1.addVolume(overlayVolume);
    if (useLabelColormap && modelEntry.colormapPath) {
      const cmap = await fetchJSON(modelEntry.colormapPath);
      await nv1.setColormapLabel(nv1.volumes.length - 1, cmap);
    }
    saveBtn.disabled = false;
    createMeshBtn.disabled = false;
  }
  function callbackUI(message = "", progressFrac = -1, modalMessage = "") {
    const locationEl = document.getElementById("location");
    if (message !== "" && locationEl) {
      console.log(message);
      locationEl.innerHTML = message;
    }
    if (Number.isNaN(progressFrac)) {
      //memory issue
      memstatus.style.color = "red";
      memstatus.innerHTML = "Memory Issue";
    } else if (progressFrac >= 0) {
      modelProgress.value = progressFrac * modelProgress.max;
    }
    if (modalMessage !== "") {
      window.alert(modalMessage);
    }
  }
  function handleLocationChange(data: { string: string }) {
    const locationEl = document.getElementById("location");
    if (locationEl) {
      locationEl.innerHTML = "&nbsp;&nbsp;" + data.string;
    }
  }
  const defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    is3DCrosshairVisible: true,
    // niivue v1's combined build defaults to WebGPU, which overflows the
    // texture-size limit on the multiplanar tiling here. Use the WebGL2 backend,
    // which is what 0.69 used and what the tfjs segmentation already runs on.
    backend: "webgl2" as const,
    // Show the multiplanar view with the 3D render, replacing the old
    // multiplanarForceRender option that v1 removed.
    sliceType: SLICE_TYPE.MULTIPLANAR,
    showRender: SHOW_RENDER.ALWAYS,
  };
  createMeshBtn.onclick = () => {
    if (nv1.meshes.length > 0) nv1.removeMesh(0);
    saveMeshBtn.disabled = true;
    if (nv1.volumes.length < 1) {
      window.alert("Image not loaded. Drag and drop an image.");
    } else {
      remeshDialog.show();
    }
  };
  function updateQualityControls() {
    const isBetterQuality = Boolean(Number(qualitySelect.value));
    const opacity = String(1.0 - 0.5 * Number(isBetterQuality));
    largestCheck.disabled = isBetterQuality;
    largestClusterGroup.style.opacity = opacity;
    hollowGroup.style.opacity = opacity;
    hollowSelect.disabled = isBetterQuality;
    bubbleCheck.disabled = isBetterQuality;
    bubbleGroup.style.opacity = opacity;
    closeMM.disabled = isBetterQuality;
    closeGroup.style.opacity = opacity;
  }
  qualitySelect.onchange = updateQualityControls;
  applyBtn.onclick = async () => {
    const isBetterQuality = Boolean(Number(qualitySelect.value));
    const startTime = performance.now();
    if (isBetterQuality) await applyQuality();
    else await applyFaster();
    console.log(
      `Execution time: ${Math.round(performance.now() - startTime)} ms`,
    );
  };
  async function applyFaster() {
    const niiBuffer = await nv1.saveVolume({
      filename: "",
      isSaveDrawing: false,
      volumeByIndex: nv1.volumes.length - 1,
    });
    if (typeof niiBuffer === "boolean") return;
    const niiFile = new File([niiBuffer as BlobPart], "image.nii");
    let processor = niimath.image(niiFile);
    loadingCircle.classList.remove("hidden");
    //mesh with specified isosurface
    let isoValue = 0.5;
    if (nv1.volumes[nv1.volumes.length - 1].hdr?.intent_code === 0) {
      isoValue = 240; //isScalar
    }
    //const largestCheckValue = largestCheck.checked
    const reduce = Math.min(Math.max(Number(shrinkPct.value) / 100, 0.01), 1);
    let hollowSz = Number(hollowSelect.value);
    let closeSz = Number(closeMM.value);
    const pixDims = nv1.volumes[0].hdr?.pixDims;
    const pixDim = pixDims
      ? Math.min(Math.min(pixDims[1], pixDims[2]), pixDims[3])
      : 1;
    if (pixDim < 0.2 && (hollowSz !== 0 || closeSz !== 0)) {
      hollowSz *= pixDim;
      closeSz *= pixDim;
      console.log(
        "Very small pixels, scaling hollow and close values by ",
        pixDim,
      );
    }
    if (hollowSz < 0) {
      processor = processor.hollow(0.5, hollowSz);
    }
    if (isFinite(closeSz) && closeSz > 0) {
      processor = processor.close(isoValue, closeSz, 2 * closeSz);
    }
    processor = processor.mesh({
      i: isoValue,
      l: largestCheck.checked ? 1 : 0,
      r: reduce,
      b: bubbleCheck.checked ? 1 : 0,
    });
    console.log(
      "niimath operation",
      (processor as unknown as { commands: unknown }).commands,
    );
    const retBlob = await processor.run("test.mz3");
    loadingCircle.classList.add("hidden");
    if (nv1.meshes.length > 0) nv1.removeMesh(0);
    // niivue v1 dropped loadFromArrayBuffer; add the niimath MZ3 as a File.
    await nv1.addMesh({ url: new File([retBlob], "test.mz3") });
  }
  async function applyQuality() {
    const volIdx = nv1.volumes.length - 1;
    const hdr = nv1.volumes[volIdx].hdr;
    const img = nv1.volumes[volIdx].img;
    /*let hollowInt = Number(hollowSelect.value )
    if (hollowInt < 0){
      const vol = nv1.volumes[volIdx]
      const niiBuffer = await nv1.saveVolume({volumeByIndex: nv1.volumes.length - 1})
      const niiBlob = new Blob([niiBuffer], { type: 'application/octet-stream' })
      const niiFile = new File([niiBlob], 'input.nii')
      niimath.setOutputDataType('input') // call before setting image since this is passed to the image constructor
      let image = niimath.image(niiFile)
      image = image.gz(0)
      image = image.ras()
      image = image.hollow(0.5, hollowInt)
      const outBlob = await image.run('output.nii') 
      let outFile = new File([outBlob], 'hollow.nii')
      const outVol = await NVImage.loadFromFile({
        file: outFile,
        name: outFile.name
      })
      hdr = outVol.hdr
      img = outVol.img
    }*/
    loadingCircle.classList.remove("hidden");
    meshProcessingMsg.classList.remove("hidden");
    meshProcessingMsg.textContent = "Generating mesh from segmentation";
    const itkImage = nii2iwi(hdr, img, false);
    itkImage.size = itkImage.size.map(Number);
    let mesh;
    if (nv1.volumes[nv1.volumes.length - 1].hdr?.intent_code === 0) {
      ({ mesh } = await cuberille(itkImage, { isoSurfaceValue: 240 }));
    } else {
      // Binarize the image: set all values >= 1 to 1
      for (let i = 0; i < itkImage.data.length; i++) {
        if (itkImage.data[i] >= 1) {
          itkImage.data[i] = 1;
        }
      }
      ({ mesh } = await antiAliasCuberille(itkImage, { noClosing: true }));
    }

    meshProcessingMsg.textContent = "Generating manifold";
    const { outputMesh: repairedMesh } = await repair(mesh, {
      maximumHoleArea: 50.0,
    });
    meshProcessingMsg.textContent = "Keep largest mesh component";
    const { outputMesh: largestOnly } =
      await keepLargestComponent(repairedMesh);
    while (nv1.meshes.length > 0) {
      nv1.removeMesh(0);
    }
    const initialNiiMesh = iwm2meshCore(largestOnly);
    const initialObj = positionsIndicesToObj(
      initialNiiMesh.positions,
      initialNiiMesh.indices,
    );
    await nv1.addMesh({ url: new File([initialObj], "trefoil.obj") });
    meshProcessingMsg.textContent = "Smoothing and remeshing";
    const smooth = parseInt(smoothSlide.value);
    const shrink = parseFloat(shrinkPct.value);
    console.log(`smoothing iterations ${smooth} shrink percent ${shrink}`);
    const { outputMesh: smoothedMesh } = await smoothRemesh(largestOnly, {
      newtonIterations: smooth,
      numberPoints: shrink,
    });
    const { outputMesh: smoothedRepairedMesh } = await repair(smoothedMesh, {
      maximumHoleArea: 50.0,
    });
    const niiMesh = iwm2meshCore(smoothedRepairedMesh);
    loadingCircle.classList.add("hidden");
    meshProcessingMsg.classList.add("hidden");
    while (nv1.meshes.length > 0) {
      nv1.removeMesh(0);
    }
    const finalObj = positionsIndicesToObj(niiMesh.positions, niiMesh.indices);
    await nv1.addMesh({ url: new File([finalObj], "trefoil.obj") });
  }
  saveMeshBtn.onclick = () => {
    if (nv1.meshes.length < 1) {
      window.alert("No mesh open for saving. Use 'Create Mesh'.");
    } else {
      saveDialog.show();
    }
  };
  applySaveBtn.onclick = async () => {
    if (nv1.meshes.length < 1) {
      return;
    }
    let format = "obj";
    if (formatSelect.selectedIndex === 0) {
      format = "mz3";
    }
    if (formatSelect.selectedIndex === 2) {
      format = "stl";
    }
    const scale = 1 / Number(scaleSelect.value);
    const mesh0 = nv1.meshes[0];
    if (!mesh0.positions) return;
    // v1's saveMesh writes the mesh's own positions and infers the format from
    // the filename. It has no scale option, so scale positions in place, save,
    // then restore.
    const positions = mesh0.positions;
    if (scale !== 1) {
      for (let i = 0; i < positions.length; i++) positions[i] *= scale;
    }
    await nv1.saveMesh(0, `mesh.${format}`);
    if (scale !== 1) {
      for (let i = 0; i < positions.length; i++) positions[i] /= scale;
    }
  };
  const nv1 = new NiiVue(defaults);
  // niivue v1 moved conform() out of the core into an extension.
  extCtx = nv1.createExtensionContext();
  extCtx.registerVolumeTransform(conform);
  // v1: attachToCanvas is async. Await it before loading so the GL context is
  // ready and the scene draws on load (otherwise the canvas stays black until an
  // interaction forces a redraw).
  await nv1.attachToCanvas(gl1);
  nv1.setDragMode("pan");
  nv1.crosshairGap = 11;
  await nv1.loadVolumes([{ url: "./t1_crop.nii.gz" }]);
  // v1 sizes the canvas at attach time; re-measure once layout has settled so
  // the scene draws at the correct size on load (not just after an interaction).
  nv1.resize();
  nv1.drawScene();
  for (let i = 0; i < inferenceModelsList.length; i++) {
    const option = document.createElement("option");
    option.text = inferenceModelsList[i].modelName;
    option.value = inferenceModelsList[i].id.toString();
    modelSelect.appendChild(option);
  }
  updateQualityControls();
  nv1.addEventListener("volumeLoaded", doLoadImage);
  nv1.addEventListener("meshLoaded", () => {
    saveMeshBtn.disabled = false;
  });
  nv1.addEventListener("locationChange", (data) => {
    handleLocationChange(data.detail);
  });
  modelSelect.selectedIndex = -1;
  console.log("brain2print 20241230");
  // uncomment next two lines to automatically run segmentation when web page is loaded
  // modelSelect.selectedIndex = 11
  // modelSelect.onchange()
}

main();
