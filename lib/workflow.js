function randomSeed() {
  return Math.floor(Math.random() * 2147483647);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function parseArgs(prompt) {
  const raw = (prompt || "").trim();

  const negMatch = raw.match(/--neg\s+([^\n]+?)(?=\s--|$)/i);
  const stepsMatch = raw.match(/--steps\s+(\d{1,3})/i);
  const cfgMatch = raw.match(/--cfg\s+([0-9]+(?:\.[0-9]+)?)/i);
  const arMatch = raw.match(/--ar\s+(\d+)\s*:\s*(\d+)/i);
  const modeMatch = raw.match(/--mode\s+(txt2img|img2img|upscale)/i);
  const denoiseMatch = raw.match(/--denoise\s+([0-9]+(?:\.[0-9]+)?)/i);
  const upscaleMatch = raw.match(/--upscale\s+([0-9]+(?:\.[0-9]+)?)/i);
  const imageMatch = raw.match(/--image\s+([^\n]+?)(?=\s--|$)/i);

  let cleanPrompt = raw
    .replace(/--neg\s+([^\n]+?)(?=\s--|$)/gi, "")
    .replace(/--steps\s+\d{1,3}/gi, "")
    .replace(/--cfg\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--ar\s+\d+\s*:\s*\d+/gi, "")
    .replace(/--mode\s+(txt2img|img2img|upscale)/gi, "")
    .replace(/--denoise\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--upscale\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--image\s+([^\n]+?)(?=\s--|$)/gi, "")
    .trim();

  if (!cleanPrompt) {
    cleanPrompt = "masterpiece, detailed, cinematic lighting";
  }

  const steps = stepsMatch ? clamp(Number(stepsMatch[1]), 1, 80) : 30;
  const cfg = cfgMatch ? clamp(Number(cfgMatch[1]), 1, 20) : 7;
  const denoise = denoiseMatch ? clamp(Number(denoiseMatch[1]), 0.05, 1) : 0.65;
  const upscaleFactor = upscaleMatch ? clamp(Number(upscaleMatch[1]), 1.5, 4) : 2;

  let width = 1024;
  let height = 1024;

  if (arMatch) {
    const w = Number(arMatch[1]);
    const h = Number(arMatch[2]);
    if (w > 0 && h > 0) {
      const base = 1024;
      if (w >= h) {
        width = base;
        height = Math.round((base * h) / w / 64) * 64;
      } else {
        height = base;
        width = Math.round((base * w) / h / 64) * 64;
      }
      width = Math.max(512, width);
      height = Math.max(512, height);
    }
  }

  return {
    prompt: cleanPrompt,
    negativePrompt: negMatch ? negMatch[1].trim() : "blurry, low quality, artifacts, deformed",
    steps,
    cfg,
    width,
    height,
    template: modeMatch ? modeMatch[1].toLowerCase() : "auto",
    denoise,
    upscaleFactor,
    sourceImage: imageMatch ? imageMatch[1].trim() : "input.png"
  };
}

function resolveTemplate(inputTemplate, prompt) {
  const template = (inputTemplate || "auto").toLowerCase();
  if (template !== "auto") {
    return template;
  }

  const text = (prompt || "").toLowerCase();
  if (text.includes("upscale") || text.includes("enhance") || text.includes("super-resolution")) {
    return "upscale";
  }
  if (text.includes("img2img") || text.includes("variation") || text.includes("restyle") || text.includes("edit")) {
    return "img2img";
  }
  return "txt2img";
}

function txt2imgWorkflow(parsed) {
  return {
    "1": {
      inputs: { ckpt_name: "v1-5-pruned-emaonly.safetensors" },
      class_type: "CheckpointLoaderSimple",
      _meta: { title: "Load Checkpoint" }
    },
    "2": {
      inputs: { text: parsed.prompt, clip: ["1", 1] },
      class_type: "CLIPTextEncode",
      _meta: { title: "Positive Prompt" }
    },
    "3": {
      inputs: { text: parsed.negativePrompt, clip: ["1", 1] },
      class_type: "CLIPTextEncode",
      _meta: { title: "Negative Prompt" }
    },
    "4": {
      inputs: { width: parsed.width, height: parsed.height, batch_size: 1 },
      class_type: "EmptyLatentImage",
      _meta: { title: "Empty Latent Image" }
    },
    "5": {
      inputs: {
        seed: randomSeed(),
        steps: parsed.steps,
        cfg: parsed.cfg,
        sampler_name: "euler",
        scheduler: "normal",
        denoise: 1,
        model: ["1", 0],
        positive: ["2", 0],
        negative: ["3", 0],
        latent_image: ["4", 0]
      },
      class_type: "KSampler",
      _meta: { title: "KSampler" }
    },
    "6": {
      inputs: { samples: ["5", 0], vae: ["1", 2] },
      class_type: "VAEDecode",
      _meta: { title: "VAE Decode" }
    },
    "7": {
      inputs: { filename_prefix: "ComfyUI_Txt2Img", images: ["6", 0] },
      class_type: "SaveImage",
      _meta: { title: "Save Image" }
    }
  };
}

function img2imgWorkflow(parsed) {
  return {
    "1": {
      inputs: { ckpt_name: "v1-5-pruned-emaonly.safetensors" },
      class_type: "CheckpointLoaderSimple",
      _meta: { title: "Load Checkpoint" }
    },
    "2": {
      inputs: { image: parsed.sourceImage, upload: "image" },
      class_type: "LoadImage",
      _meta: { title: "Load Source Image" }
    },
    "3": {
      inputs: { pixels: ["2", 0], vae: ["1", 2] },
      class_type: "VAEEncode",
      _meta: { title: "VAE Encode" }
    },
    "4": {
      inputs: { text: parsed.prompt, clip: ["1", 1] },
      class_type: "CLIPTextEncode",
      _meta: { title: "Positive Prompt" }
    },
    "5": {
      inputs: { text: parsed.negativePrompt, clip: ["1", 1] },
      class_type: "CLIPTextEncode",
      _meta: { title: "Negative Prompt" }
    },
    "6": {
      inputs: {
        seed: randomSeed(),
        steps: parsed.steps,
        cfg: parsed.cfg,
        sampler_name: "euler",
        scheduler: "normal",
        denoise: parsed.denoise,
        model: ["1", 0],
        positive: ["4", 0],
        negative: ["5", 0],
        latent_image: ["3", 0]
      },
      class_type: "KSampler",
      _meta: { title: "KSampler (Img2Img)" }
    },
    "7": {
      inputs: { samples: ["6", 0], vae: ["1", 2] },
      class_type: "VAEDecode",
      _meta: { title: "VAE Decode" }
    },
    "8": {
      inputs: { filename_prefix: "ComfyUI_Img2Img", images: ["7", 0] },
      class_type: "SaveImage",
      _meta: { title: "Save Image" }
    }
  };
}

function upscaleWorkflow(parsed) {
  return {
    "1": {
      inputs: { image: parsed.sourceImage, upload: "image" },
      class_type: "LoadImage",
      _meta: { title: "Load Source Image" }
    },
    "2": {
      inputs: { model_name: "4x-UltraSharp.pth" },
      class_type: "UpscaleModelLoader",
      _meta: { title: "Load Upscale Model" }
    },
    "3": {
      inputs: { image: ["1", 0], upscale_model: ["2", 0] },
      class_type: "ImageUpscaleWithModel",
      _meta: { title: "AI Upscale" }
    },
    "4": {
      inputs: { image: ["3", 0], upscale_method: "bicubic", scale_by: parsed.upscaleFactor },
      class_type: "ImageScaleBy",
      _meta: { title: "Resize By Factor" }
    },
    "5": {
      inputs: { filename_prefix: "ComfyUI_Upscale", images: ["4", 0] },
      class_type: "SaveImage",
      _meta: { title: "Save Image" }
    }
  };
}

function buildWorkflowFromParsed(parsed, templateInput) {
  const template = resolveTemplate(templateInput || parsed.template, parsed.prompt);

  if (template === "img2img") {
    return { template, workflow: img2imgWorkflow(parsed) };
  }
  if (template === "upscale") {
    return { template, workflow: upscaleWorkflow(parsed) };
  }
  return { template: "txt2img", workflow: txt2imgWorkflow(parsed) };
}

function buildWorkflow(promptText, templateInput) {
  const parsed = parseArgs(promptText);
  return buildWorkflowFromParsed(parsed, templateInput);
}

module.exports = {
  parseArgs,
  resolveTemplate,
  buildWorkflow,
  buildWorkflowFromParsed
};
