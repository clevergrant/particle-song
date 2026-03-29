import { simulations } from "./simulations";
import { attachNumberScroll } from "./number-scroll";
import type { Simulation, GpuContext } from "./types";
import { ShaderMenu } from "./shader-menu";
import "./num-input"; // register <num-input> custom element
import "./ui/floating-window"; // register <floating-window> custom element
import "./ui/widgets"; // register all widget custom elements
import { WindowManager } from "./ui/window-manager";
import { FabToggle } from "./ui/fab-toggle";

// Styles
import "./styles/main.scss";

// Global error overlay — pre-rendered and hidden, just flip opacity on error
const STOIC_API = "https://www.stoic-quotes.com/api/quote";
const FALLBACK_QUOTE = { text: "ha. fix ur code, dummy", author: "You" };

const overlay = document.createElement("div");
overlay.id = "error-overlay";
const quoteText = document.createElement("p");
const quoteAuthor = document.createElement("cite");
overlay.innerHTML = `<div class="error-backdrop"></div>`;
const figure = document.createElement("figure");
figure.className = "error-quote";
figure.innerHTML = `<blockquote></blockquote><figcaption>— </figcaption>`;
figure.querySelector("blockquote")!.appendChild(quoteText);
figure.querySelector("figcaption")!.appendChild(quoteAuthor);
overlay.appendChild(figure);
document.body.appendChild(overlay);

function renderQuote(text: string, author: string) {
  quoteText.textContent = text;
  quoteAuthor.textContent = author;
}

function prefetchQuote() {
  fetch(STOIC_API)
    .then(r => r.json())
    .then(({ text, author }: { text: string; author: string }) => renderQuote(text, author))
    .catch(() => renderQuote(FALLBACK_QUOTE.text, FALLBACK_QUOTE.author));
}

renderQuote(FALLBACK_QUOTE.text, FALLBACK_QUOTE.author);
prefetchQuote();

overlay.addEventListener("click", () => {
  overlay.classList.remove("visible");
});

function showErrorOverlay(err?: unknown) {
  console.error("[particle-sim] Unhandled error:", err);
  if (overlay.classList.contains("visible")) return;
  overlay.classList.add("visible");
  prefetchQuote();
}

window.addEventListener("error", (e) =>
  showErrorOverlay(e.error ?? `${e.message} (${e.filename}:${e.lineno}:${e.colno})`),
);
window.addEventListener("unhandledrejection", (e) => showErrorOverlay(e.reason));

const STORAGE_KEY_SETTINGS = "particle-sim:settings:";

interface StoredSettings {
  version?: string;
  data: Record<string, string>;
  sections?: Record<string, string>;
  ledgerOpen?: boolean;
  mutedOrganisms?: string[];
}


/** All containers whose [data-setting] elements participate in save/load */
function getSettingsRoots(): HTMLElement[] {
  if (usingWindows && windowManager) {
    // In window mode, sweep all window content elements
    const roots = windowManager.getAllContentElements();
    if (shaderMenu) {
      roots.push(shaderMenu.particleParamsEl, shaderMenu.postParamsEl);
    }
    return roots;
  }
  // Legacy mode
  const roots: HTMLElement[] = [controls];
  if (shaderMenu) {
    roots.push(shaderMenu.particleParamsEl, shaderMenu.postParamsEl);
  }
  return roots;
}

/** Get all elements with [data-section] across all active roots */
function getSectionElements(): NodeListOf<HTMLElement> | HTMLElement[] {
  if (usingWindows && windowManager) {
    const elements: HTMLElement[] = [];
    for (const root of windowManager.getAllContentElements()) {
      elements.push(...root.querySelectorAll<HTMLElement>("[data-section]"));
    }
    return elements;
  }
  return controls.querySelectorAll<HTMLElement>("[data-section]");
}

function saveSettings(sim: Simulation) {
  const data: Record<string, string> = {};
  for (const root of getSettingsRoots()) {
    for (const el of root.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-setting]")) {
      const key = el.dataset.setting!;
      if (el instanceof HTMLSelectElement) {
        data[key] = el.value;
      } else if (el.type === "checkbox") {
        data[key] = String((el as HTMLInputElement).checked);
      } else if (el.type === "radio") {
        if ((el as HTMLInputElement).checked) data[key] = el.value;
      } else {
        data[key] = el.value;
      }
    }
  }
  const sections: Record<string, string> = {};
  for (const el of getSectionElements()) {
    sections[el.dataset.section!] = String(el.classList.contains("open"));
  }
  const ledgerPanels = document.getElementById("ledger-panels");
  const ledgerOpen = ledgerPanels?.classList.contains("open") ?? false;
  const mutedAttr = document.getElementById("ledger-organisms")?.dataset.mutedOrganisms;
  const mutedOrganisms = mutedAttr ? mutedAttr.split(",").filter(Boolean) : undefined;
  const envelope: StoredSettings = { version: sim.settingsVersion, data, sections, ledgerOpen, mutedOrganisms };
  localStorage.setItem(STORAGE_KEY_SETTINGS + sim.name, JSON.stringify(envelope));
}

function loadSettings(sim: Simulation) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_SETTINGS + sim.name);
    if (!raw) return;
    const parsed: StoredSettings | Record<string, string> = JSON.parse(raw);

    // Migrate legacy format (plain data object without envelope)
    const envelope: StoredSettings =
      "data" in parsed && typeof (parsed as StoredSettings).data === "object"
        ? (parsed as StoredSettings)
        : { data: parsed as Record<string, string> };

    // Invalidate if version doesn't match
    if (envelope.version !== sim.settingsVersion) {
      localStorage.removeItem(STORAGE_KEY_SETTINGS + sim.name);
      return;
    }

    const data = envelope.data;
    for (const root of getSettingsRoots()) {
      for (const el of root.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-setting]")) {
        const key = el.dataset.setting!;
        if (!(key in data)) continue;
        // Audio requires a user gesture — don't auto-enable from saved state
        if (key === "soundEnabled") continue;

        if (el instanceof HTMLSelectElement) {
          el.value = data[key];
          el.dispatchEvent(new Event("change", { bubbles: true }));
        } else if (el.type === "checkbox") {
          (el as HTMLInputElement).checked = data[key] === "true";
          el.dispatchEvent(new Event("change", { bubbles: true }));
        } else if (el.type === "radio") {
          (el as HTMLInputElement).checked = el.value === data[key];
          if ((el as HTMLInputElement).checked) el.dispatchEvent(new Event("change", { bubbles: true }));
        } else {
          el.value = data[key];
          el.dispatchEvent(new Event("input", { bubbles: true }));
        }
      }
    }

    const sections = envelope.sections;
    if (sections) {
      for (const el of getSectionElements()) {
        const key = el.dataset.section!;
        if (key in sections) {
          el.classList.toggle("open", sections[key] === "true");
        }
      }
    }

    if (envelope.ledgerOpen != null) {
      const ledgerPanels = document.getElementById("ledger-panels");
      const ledgerBackdrop = document.getElementById("ledger-backdrop");
      ledgerPanels?.classList.toggle("open", envelope.ledgerOpen);
      ledgerBackdrop?.classList.toggle("open", envelope.ledgerOpen);
    }

    if (envelope.mutedOrganisms?.length) {
      const osmEl = document.getElementById("ledger-organisms");
      if (osmEl) osmEl.dataset.mutedOrganisms = envelope.mutedOrganisms.join(",");
    }
  } catch { /* ignore corrupt data */ }
}

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const controls = document.getElementById("controls")!;

// Window system layer
const windowLayer = document.createElement("div");
windowLayer.id = "window-layer";
Object.assign(windowLayer.style, { position: "fixed", inset: "0", pointerEvents: "none", zIndex: "100" });
document.body.appendChild(windowLayer);

let windowManager: WindowManager | null = null;
let fabToggle: FabToggle | null = null;
let usingWindows = false; // true when current sim uses getWindows()

let gpu: GpuContext | null = null;
let active: Simulation | null = null;
let animFrame = 0;
let lastTime = 0;
let controlsListener: (() => void) | null = null;

let shaderMenu: ShaderMenu | null = null;
let paramListener: (() => void) | null = null;
let ledgerListener: (() => void) | null = null;
let resizeTimer = 0;

function resize() {
  const container = canvas.parentElement!;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;

  if (active && gpu) {
    if (active.resize) {
      active.resize(gpu, canvas.width, canvas.height);
    } else {
      active.setup(gpu, canvas.width, canvas.height);
    }
  }
}

function debouncedResize() {
  clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(resize, 100);
}

const FIXED_DT = 1 / 60;
const MAX_STEPS_PER_FRAME = 3;
let accumulator = 0;
let paused = false;

window.addEventListener("sim-pause", ((e: CustomEvent<{ paused: boolean }>) => {
  paused = e.detail.paused;
  if (!paused) {
    // Reset timing so we don't process a huge elapsed spike on resume
    lastTime = performance.now();
    accumulator = 0;
  }
}) as EventListener);

function loop(time: number) {
  if (paused) {
    animFrame = requestAnimationFrame(loop);
    return;
  }

  const elapsed = Math.min((time - lastTime) / 1000, MAX_STEPS_PER_FRAME * FIXED_DT);
  lastTime = time;

  if (active && gpu) {
    accumulator += elapsed;
    let stepped = false;
    while (accumulator >= FIXED_DT) {
      active.update(FIXED_DT);
      accumulator -= FIXED_DT;
      stepped = true;
    }
    if (stepped) {
      active.draw(gpu);
    }
  }

  animFrame = requestAnimationFrame(loop);
}

function selectSimulation(sim: Simulation) {
  if (!gpu) return;

  if (active) {
    cancelAnimationFrame(animFrame);
    active.teardown?.();
  }

  active = sim;
  accumulator = 0;
  paused = false;

  try {
    active.setup(gpu, canvas.width, canvas.height);
  } catch (e) {
    controls.innerHTML = "";
    throw e;
  }

  // Build controls after setup so particles exist
  controls.innerHTML = "";

  // Check if simulation supports the new window system
  if (sim.getWindows) {
    usingWindows = true;

    // Clear previous window manager
    windowManager?.clearAll();
    windowManager = new WindowManager(windowLayer, sim.name);

    // Register all windows from the simulation
    for (const def of sim.getWindows()) {
      const container = document.createElement("div");
      def.build(container);
      attachNumberScroll(container);
      windowManager.register({
        id: def.id,
        title: def.title,
        icon: def.icon,
        category: def.category,
        defaultVisible: def.defaultVisible,
        defaultPosition: def.defaultPosition,
        defaultWidth: def.defaultWidth,
      }, container);
    }

    // Build or update FAB toggle
    // Remove old FAB if switching simulations
    if (fabToggle) {
      document.querySelector(".fab-container")?.remove();
    }
    fabToggle = new FabToggle(document.body, windowManager);
    fabToggle.build();
    windowManager.applyLayout();

    // Hide legacy controls wrapper
    const wrapper = document.getElementById("controls-wrapper");
    if (wrapper) wrapper.classList.add("empty");
  } else {
    usingWindows = false;
    // Legacy fallback: use old buildControls
    sim.buildControls?.(controls);
    attachNumberScroll(controls);
  }

  // Set up shader menu if simulation supports shader switching
  const hasSwitchMethods = "switchParticleShader" in sim && "switchPostShader" in sim;
  if (hasSwitchMethods) {
    const s = sim as any;
    const rebuildParticleParams = () => {
      if (shaderMenu && typeof s.getParticleShaderParams === "function") {
        shaderMenu.setParticleParams(s.getParticleShaderParams());
      }
    };
    const rebuildPostParams = () => {
      if (shaderMenu && typeof s.getPostShaderParams === "function") {
        shaderMenu.setPostParams(s.getPostShaderParams());
      }
    };

    // Always create a fresh ShaderMenu when switching simulations
    shaderMenu = new ShaderMenu({
      onParticleEffectChange: (id) => { s.switchParticleShader(id); rebuildParticleParams(); },
      onPostEffectChange: (id) => { s.switchPostShader(id); rebuildPostParams(); },
    }, s.getActiveParticleEffectId(), s.getActivePostEffectId());

    // In window mode, build the shader menu into the shaders window
    if (usingWindows && windowManager) {
      const shadersContent = windowManager.getContentElement("shaders");
      if (shadersContent) {
        shaderMenu.buildInto(shadersContent);
      }
    }

    // Listen for restored settings updating effects
    s.onParticleEffectChanged = (id: string) => {
      shaderMenu?.syncSelections(id, s.getActivePostEffectId());
      rebuildParticleParams();
    };
    s.onPostEffectChanged = (id: string) => {
      shaderMenu?.syncSelections(s.getActiveParticleEffectId(), id);
      rebuildPostParams();
    };

    // Populate shader param panels
    rebuildParticleParams();
    rebuildPostParams();
  } else {
    shaderMenu = null;
  }

  // Restore saved values (after param panels are populated so their inputs exist)
  loadSettings(sim);

  // Save settings whenever a control changes
  if (controlsListener) {
    controls.removeEventListener("input", controlsListener);
    controls.removeEventListener("change", controlsListener);
    controls.removeEventListener("click", controlsListener);
  }
  controlsListener = () => { saveSettings(sim); };

  if (usingWindows && windowManager) {
    // Attach save listeners to all window content elements
    for (const el of windowManager.getAllContentElements()) {
      el.addEventListener("input", controlsListener);
      el.addEventListener("change", controlsListener);
      el.addEventListener("click", controlsListener);
    }
  } else {
    controls.addEventListener("input", controlsListener);
    controls.addEventListener("change", controlsListener);
    controls.addEventListener("click", controlsListener);
  }

  // Also save when shader param panel values change
  if (shaderMenu) {
    if (paramListener) {
      shaderMenu.particleParamsEl.removeEventListener("input", paramListener);
      shaderMenu.postParamsEl.removeEventListener("input", paramListener);
    }
    paramListener = () => { saveSettings(sim); };
    shaderMenu.particleParamsEl.addEventListener("input", paramListener);
    shaderMenu.postParamsEl.addEventListener("input", paramListener);
  }

  // Save when ledger open/close or muted organisms change
  const ledgerPanelsEl = document.getElementById("ledger-panels");
  const ledgerOrganismsEl = document.getElementById("ledger-organisms");
  if (ledgerPanelsEl) {
    if (ledgerListener) ledgerListener();
    const saveCb = () => { saveSettings(sim); };
    const panelsObs = new MutationObserver(saveCb);
    panelsObs.observe(ledgerPanelsEl, { attributes: true, attributeFilter: ["class"] });
    let organismsObs: MutationObserver | null = null;
    if (ledgerOrganismsEl) {
      organismsObs = new MutationObserver(saveCb);
      organismsObs.observe(ledgerOrganismsEl, { attributes: true, attributeFilter: ["data-muted-organisms"] });
    }
    ledgerListener = () => { panelsObs.disconnect(); organismsObs?.disconnect(); };
  }
  lastTime = performance.now();
  animFrame = requestAnimationFrame(loop);

}

window.addEventListener("resize", debouncedResize);

// --- Async WebGPU initialization ---
async function init() {
  if (!navigator.gpu) {
    document.body.innerHTML =
      "<p style='color:red;padding:2em'>WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, or another WebGPU-capable browser.</p>";
    throw new Error("WebGPU not supported");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    document.body.innerHTML =
      "<p style='color:red;padding:2em'>Failed to get WebGPU adapter. Your GPU may not be supported.</p>";
    throw new Error("No WebGPU adapter");
  }

  const device = await adapter.requestDevice();
  const canvasContext = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();

  canvasContext.configure({ device, format, alphaMode: "opaque" });

  gpu = { device, canvasContext, format, canvas };

  resize();

  selectSimulation(simulations[0]);

  // Wallpaper Engine integration — hide UI and wire up property listener
  if (
    (window as any).wallpaperPropertyListener !== undefined ||
    new URLSearchParams(location.search).has("wallpaper")
  ) {
    // @ts-ignore — wallpaper-engine is gitignored; only exists locally
    const { init: initBridge } = await import(
      /* @vite-ignore */ "../wallpaper-engine/bridge"
    );
    initBridge();
  }
}

init();
