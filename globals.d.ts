// The HTML markup gives many elements an `id`, and browsers expose each such
// element as a global of the same name. main.ts relies on that. Declare the
// ones it uses so TypeScript knows their element types instead of erroring on
// an undefined name. This mirrors index.html; keep the two in sync.

declare const gl1: HTMLCanvasElement;

declare const aboutBtn: HTMLButtonElement;
declare const applyBtn: HTMLButtonElement;
declare const applySaveBtn: HTMLButtonElement;
declare const createMeshBtn: HTMLButtonElement;
declare const saveBtn: HTMLButtonElement;
declare const saveMeshBtn: HTMLButtonElement;

declare const formatSelect: HTMLSelectElement;
declare const hollowSelect: HTMLSelectElement;
declare const modelSelect: HTMLSelectElement;
declare const qualitySelect: HTMLSelectElement;
declare const scaleSelect: HTMLSelectElement;

declare const bubbleCheck: HTMLInputElement;
declare const clipCheck: HTMLInputElement;
declare const closeMM: HTMLInputElement;
declare const largestCheck: HTMLInputElement;
declare const opacitySlider0: HTMLInputElement;
declare const opacitySlider1: HTMLInputElement;
declare const scalarCheck: HTMLInputElement;
declare const shrinkPct: HTMLInputElement;
declare const smoothSlide: HTMLInputElement;

declare const remeshDialog: HTMLDialogElement;
declare const saveDialog: HTMLDialogElement;

declare const modelProgress: HTMLProgressElement;

// @niivue/cbor-loader ships no type declarations.
declare module "@niivue/cbor-loader";

declare const bubbleGroup: HTMLElement;
declare const closeGroup: HTMLElement;
declare const hollowGroup: HTMLElement;
declare const largestClusterGroup: HTMLElement;
declare const loadingCircle: HTMLElement;
declare const memstatus: HTMLElement;
declare const meshProcessingMsg: HTMLElement;
