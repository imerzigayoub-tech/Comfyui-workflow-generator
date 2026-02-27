# ComfyUI Workflow Generator

Web app that converts a text prompt into a ComfyUI workflow JSON using your provider API key (BYOK).

## Features
- Prompt-only workflow generation via BYOK provider API (`OpenRouter`, `OpenAI`, `Google`)
- Workflow topology is generated from your prompt intent (no template selector)
- Pipeline is: provider draft -> server compiler -> strict runnable workflow JSON
- Optional inline controls in prompt:
  - `--ckpt your_model.safetensors`
- BYOK mode: choose `OpenRouter`, `OpenAI`, or `Google Gemini` and provide your API key
  - Provider generates a full ComfyUI workflow JSON from the prompt
- Download generated `workflow.json`
  - Download now exports node-based `workflow-ui.json` for ComfyUI canvas import

## Run
1. Open terminal in this folder:
   ```powershell
   cd C:\Users\pc\OneDrive\Bureau\Dev\comfyui-workflow-generator
   ```
2. Start server:
   ```powershell
   node server.js
   ```
3. Open:
   - `http://localhost:3000`

Note: generation is BYOK-only. Local `server.js` serves the frontend only.

## Import into ComfyUI
1. Generate and download `workflow-ui.json` from this app.
2. In ComfyUI, use workflow load/import and select that file.

## Notes
- Default checkpoint is `sd_xl_base_1.0.safetensors`.
- Override per prompt with `--ckpt ...` or globally with env var `DEFAULT_CKPT_NAME`.
- Upscale template default model is `4x-UltraSharp.pth` (also in `lib/workflow.js`).
- BYOK latency/reliability env vars:
  - `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`)
  - `LLM_TIMEOUT_MS` (default `25000`)
  - `LLM_MAX_TOKENS` (default `2600`)

## Vercel Deploy
- This repo includes a Vercel serverless route at `api/generate.js`.
- Frontend calls `/api/generate` and works on Vercel without running `server.js`.
- Provider selection is sent from the UI (`openrouter`, `openai`, `google`).
