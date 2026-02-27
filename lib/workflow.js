function randomSeed() {
  return Math.floor(Math.random() * 2147483647);
}

function parseArgs(prompt) {
  const raw = (prompt || "").trim();

  const negMatch = raw.match(/--neg\s+([^\n]+?)(?=\s--|$)/i);
  const stepsMatch = raw.match(/--steps\s+(\d{1,3})/i);
  const cfgMatch = raw.match(/--cfg\s+([0-9]+(?:\.[0-9]+)?)/i);
  const arMatch = raw.match(/--ar\s+(\d+)\s*:\s*(\d+)/i);

  let cleanPrompt = raw
    .replace(/--neg\s+([^\n]+?)(?=\s--|$)/gi, "")
    .replace(/--steps\s+\d{1,3}/gi, "")
    .replace(/--cfg\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--ar\s+\d+\s*:\s*\d+/gi, "")
    .trim();

  if (!cleanPrompt) {
    cleanPrompt = "masterpiece, detailed, cinematic lighting";
  }

  const steps = stepsMatch ? Math.min(80, Math.max(1, Number(stepsMatch[1]))) : 30;
  const cfg = cfgMatch ? Math.min(20, Math.max(1, Number(cfgMatch[1]))) : 7;

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
    height
  };
}

function buildWorkflowFromParsed(parsed) {
  return {
    "1": {
      inputs: {
        ckpt_name: "v1-5-pruned-emaonly.safetensors"
      },
      class_type: "CheckpointLoaderSimple",
      _meta: { title: "Load Checkpoint" }
    },
    "2": {
      inputs: {
        text: parsed.prompt,
        clip: ["1", 1]
      },
      class_type: "CLIPTextEncode",
      _meta: { title: "Positive Prompt" }
    },
    "3": {
      inputs: {
        text: parsed.negativePrompt,
        clip: ["1", 1]
      },
      class_type: "CLIPTextEncode",
      _meta: { title: "Negative Prompt" }
    },
    "4": {
      inputs: {
        width: parsed.width,
        height: parsed.height,
        batch_size: 1
      },
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
      inputs: {
        samples: ["5", 0],
        vae: ["1", 2]
      },
      class_type: "VAEDecode",
      _meta: { title: "VAE Decode" }
    },
    "7": {
      inputs: {
        filename_prefix: "ComfyUI_Generated",
        images: ["6", 0]
      },
      class_type: "SaveImage",
      _meta: { title: "Save Image" }
    }
  };
}

function buildWorkflow(promptText) {
  return buildWorkflowFromParsed(parseArgs(promptText));
}

module.exports = {
  parseArgs,
  buildWorkflow,
  buildWorkflowFromParsed
};
