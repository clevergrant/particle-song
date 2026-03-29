export interface GpuContext {
  device: GPUDevice;
  canvasContext: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

export interface WindowDefinition {
  readonly id: string;
  readonly title: string;
  readonly icon: string;
  readonly category?: string;
  readonly defaultVisible?: boolean;
  readonly defaultPosition?: { readonly x: number; readonly y: number };
  readonly defaultWidth?: number;
  build(container: HTMLElement): void;
}

export interface Simulation {
  /** Display name shown in the sidebar */
  name: string;

  /** Called once when the simulation is selected */
  setup(gpu: GpuContext, width: number, height: number): void;

  /** Called when the canvas is resized (optional — defaults to full setup) */
  resize?(gpu: GpuContext, width: number, height: number): void;

  /** Called every frame. dt is delta time in seconds */
  update(dt: number): void;

  /** Called every frame after update */
  draw(gpu: GpuContext): void;

  /** Declare floating window panels for this simulation */
  getWindows?(): WindowDefinition[];

  /** Called to populate the sidebar controls panel (legacy fallback) */
  buildControls?(container: HTMLElement): void;

  /**
   * Datetime version string for saved settings (e.g. "2026-03-24").
   * Changing this invalidates any previously stored localStorage data
   * for this simulation, forcing a fresh start with new defaults.
   */
  settingsVersion?: string;

  /** Called when the simulation is deselected (optional cleanup) */
  teardown?(): void;
}
