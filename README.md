# ComfyUI Workflow Generator

Simple local app that converts a text prompt into a ComfyUI workflow JSON.

## Features
- Prompt to workflow JSON with multiple templates:
  - `txt2img_fast`
  - `txt2img_refine`
  - `txt2img_upscale`
  - `img2img` (includes `LoadImage` + `VAEEncode`)
  - `upscale` (includes `UpscaleModelLoader` + `ImageUpscaleWithModel`)
- Supports inline prompt controls:
  - `--neg ...` negative prompt
  - `--steps N` sampling steps
  - `--cfg N` guidance scale
  - `--ar W:H` aspect ratio (auto-sized around 1024)
  - `--mode txt2img|img2img|upscale`
  - `--ckpt your_model.safetensors`
  - `--denoise 0.65` (img2img)
  - `--upscale 2` (upscale)
  - `--image input.png` (img2img/upscale source image name)
- Optional BYOK mode: choose `OpenRouter`, `OpenAI`, or `Google Gemini` and provide your API key
  - In BYOK mode, the provider generates a full ComfyUI workflow JSON from the prompt
  - BYOK execution is available on deployed `/api/generate` (Vercel serverless route)
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

Note: local `server.js` is for local parser workflows. API-key BYOK generation runs through deployed serverless API.

## Import into ComfyUI
1. Generate and download `workflow.json` from this app.
2. In ComfyUI, use workflow load/import and select that file.

## Notes
- Default checkpoint is `sd_xl_base_1.0.safetensors`.
- Override per prompt with `--ckpt ...` or globally with env var `DEFAULT_CKPT_NAME`.
- Upscale template default model is `4x-UltraSharp.pth` (also in `lib/workflow.js`).

## Vercel Deploy
- This repo includes a Vercel serverless route at `api/generate.js`.
- Frontend calls `/api/generate` and works on Vercel without running `server.js`.
- Provider selection is sent from the UI (`local`, `openrouter`, `openai`, `google`).
- Auto template routing picks different workflow graphs from prompt intent (detail/refine/upscale/edit).
