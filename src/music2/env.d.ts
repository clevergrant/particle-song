// Vite's `?worker&url` suffix compiles the referenced file as a worker
// bundle (ES-module) and returns the URL of the emitted chunk. Used by
// `audio-graph.ts` to load `voice-processor.ts` into an AudioWorklet.
declare module "*?worker&url" {
	const url: string
	export default url
}
