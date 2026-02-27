# ComfyUI Workflow Generator

Simple local app that converts a text prompt into a ComfyUI workflow JSON.

## Features
- Prompt to workflow JSON with multiple templates:
  - `txt2img`
  - `img2img` (includes `LoadImage` + `VAEEncode`)
  - `upscale` (includes `UpscaleModelLoader` + `ImageUpscaleWithModel`)
- Supports inline prompt controls:
  - `--neg ...` negative prompt
  - `--steps N` sampling steps
  - `--cfg N` guidance scale
  - `--ar W:H` aspect ratio (auto-sized around 1024)
  - `--mode txt2img|img2img|upscale`
  - `--denoise 0.65` (img2img)
  - `--upscale 2` (upscale)
  - `--image input.png` (img2img/upscale source image name)
- Optional BYOK mode: choose `OpenRouter`, `OpenAI`, or `Google Gemini` and provide your API key
- Download generated `workflow.json`

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

## Import into ComfyUI
1. Generate and download `workflow.json` from this app.
2. In ComfyUI, use workflow load/import and select that file.

## Notes
- Default checkpoint is `v1-5-pruned-emaonly.safetensors`.
- Change this in `lib/workflow.js` (`ckpt_name`) to match your installed model filename.
- Upscale template default model is `4x-UltraSharp.pth` (also in `lib/workflow.js`).

## Vercel Deploy
- This repo includes a Vercel serverless route at `api/generate.js`.
- Frontend calls `/api/generate` and works on Vercel without running `server.js`.
- Provider selection is sent from the UI (`local`, `openrouter`, `openai`, `google`).
