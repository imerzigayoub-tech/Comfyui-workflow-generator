const { apiWorkflowToUiWorkflow } = require("../lib/workflow");

const DEFAULT_CKPT_NAME = process.env.DEFAULT_CKPT_NAME || "sd_xl_base_1.0.safetensors";
const DEFAULT_OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || "openai/gpt-4o-mini";
const REQUEST_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 60000);
const LLM_MAX_TOKENS = Number(process.env.LLM_MAX_TOKENS || 900);

const SCHEDULERS = new Set(["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"]);
const DEFAULT_NEG = "blurry, low quality, artifacts, deformed";

function normalizeApiKey(raw) {
  const value = String(raw || "").trim();
  if (!value) return "";
  return value.replace(/^Bearer\s+/i, "").trim();
}

async function fetchWithTimeout(url, options, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error && error.name === "AbortError") {
      throw new Error(`Provider request timed out after ${timeoutMs}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function extractJsonObject(text) {
  if (!text) {
    throw new Error("Provider returned empty response.");
  }

  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("Provider did not return valid JSON.");
  }

  return JSON.parse(text.slice(start, end + 1));
}

function validateWorkflow(workflow) {
  if (!workflow || typeof workflow !== "object" || Array.isArray(workflow)) {
    throw new Error("Generated workflow is not an object.");
  }

  const nodeIds = Object.keys(workflow);
  if (nodeIds.length < 3) {
    throw new Error("Generated workflow has too few nodes.");
  }

  let linkRefCount = 0;
  for (const id of nodeIds) {
    const node = workflow[id];
    if (!node || typeof node !== "object") {
      throw new Error(`Invalid node at id ${id}.`);
    }
    if (!node.class_type || typeof node.class_type !== "string") {
      throw new Error(`Node ${id} missing class_type.`);
    }
    if (!node.inputs || typeof node.inputs !== "object") {
      throw new Error(`Node ${id} missing inputs.`);
    }
    for (const value of Object.values(node.inputs || {})) {
      if (Array.isArray(value) && value.length === 2) {
        linkRefCount += 1;
      }
    }
  }

  if (linkRefCount === 0) {
    throw new Error("Generated workflow has no links between nodes.");
  }

  return workflow;
}

function workflowSystemPrompt() {
  return [
    "You are generating a compact workflow PLAN from the user prompt.",
    "Return JSON only, no markdown.",
    "Return ONLY this object shape:",
    "{ intent, prompt, negativePrompt, steps, cfg, width, height, denoise, sourceImage, upscaleFactor, ckpt_name }",
    "intent must be one of: txt2img, txt2img_refine, img2img, upscale.",
    "Do NOT output full workflow nodes.",
    "Keep values concise and valid."
  ].join(" ");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function parsePromptHints(prompt) {
  const raw = String(prompt || "").trim();

  const ckptMatch = raw.match(/--ckpt\s+([^\n]+?)(?=\s--|$)/i);
  const negMatch = raw.match(/--neg\s+([^\n]+?)(?=\s--|$)/i);
  const stepsMatch = raw.match(/--steps\s+(\d{1,3})/i);
  const cfgMatch = raw.match(/--cfg\s+([0-9]+(?:\.[0-9]+)?)/i);
  const denoiseMatch = raw.match(/--denoise\s+([0-9]+(?:\.[0-9]+)?)/i);
  const upscaleMatch = raw.match(/--upscale\s+([0-9]+(?:\.[0-9]+)?)/i);
  const imageMatch = raw.match(/--image\s+([^\n]+?)(?=\s--|$)/i);
  const arMatch = raw.match(/--ar\s+(\d+)\s*:\s*(\d+)/i);

  let cleanPrompt = raw
    .replace(/--ckpt\s+([^\n]+?)(?=\s--|$)/gi, "")
    .replace(/--neg\s+([^\n]+?)(?=\s--|$)/gi, "")
    .replace(/--steps\s+\d{1,3}/gi, "")
    .replace(/--cfg\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--denoise\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--upscale\s+[0-9]+(?:\.[0-9]+)?/gi, "")
    .replace(/--image\s+([^\n]+?)(?=\s--|$)/gi, "")
    .replace(/--ar\s+\d+\s*:\s*\d+/gi, "")
    .trim();

  if (!cleanPrompt) cleanPrompt = "masterpiece, detailed, cinematic lighting";

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
    negativePrompt: negMatch ? negMatch[1].trim() : DEFAULT_NEG,
    steps: stepsMatch ? clamp(Number(stepsMatch[1]), 1, 80) : 30,
    cfg: cfgMatch ? clamp(Number(cfgMatch[1]), 1, 20) : 7,
    denoise: denoiseMatch ? clamp(Number(denoiseMatch[1]), 0.05, 1) : 0.65,
    width,
    height,
    sourceImage: imageMatch ? imageMatch[1].trim() : "input.png",
    upscaleFactor: upscaleMatch ? clamp(Number(upscaleMatch[1]), 1.5, 4) : 2,
    ckptName: ckptMatch ? ckptMatch[1].trim() : DEFAULT_CKPT_NAME
  };
}

function detectIntent(prompt, draft) {
  const text = String(prompt || "").toLowerCase();
  const draftIntent = String(draft?.intent || "").toLowerCase();
  if (["txt2img", "txt2img_refine", "img2img", "upscale"].includes(draftIntent)) {
    return draftIntent;
  }

  const draftWorkflow = draft?.workflow && typeof draft.workflow === "object" ? draft.workflow : draft;
  if (draftWorkflow && typeof draftWorkflow === "object") {
    const nodeTypes = new Set(Object.values(draftWorkflow).map(node => node?.class_type));
    if (nodeTypes.has("UpscaleModelLoader") || nodeTypes.has("ImageUpscaleWithModel")) return "upscale";
    if (nodeTypes.has("LoadImage") || nodeTypes.has("VAEEncode")) return "img2img";
  }

  if (text.includes("upscale") || text.includes("super resolution") || text.includes("enhance")) return "upscale";
  if (text.includes("img2img") || text.includes("restyle") || text.includes("edit") || text.includes("variation") || text.includes("inpaint")) return "img2img";
  if (text.includes("refine") || text.includes("hires") || text.includes("2 pass") || text.includes("two pass")) return "txt2img_refine";
  return "txt2img";
}

function readFirstNodeInput(workflow, classType, inputName) {
  if (!workflow || typeof workflow !== "object") return undefined;
  const ids = Object.keys(workflow).sort((a, b) => Number(a) - Number(b));
  for (const id of ids) {
    const node = workflow[id];
    if (node?.class_type === classType && node.inputs && node.inputs[inputName] !== undefined) {
      return node.inputs[inputName];
    }
  }
  return undefined;
}

function compileTxt2Img(params, refine = false) {
  const base = {
    "1": { inputs: { ckpt_name: params.ckptName }, class_type: "CheckpointLoaderSimple", _meta: { title: "Load Checkpoint" } },
    "2": { inputs: { text: params.prompt, clip: ["1", 1] }, class_type: "CLIPTextEncode", _meta: { title: "Positive Prompt" } },
    "3": { inputs: { text: params.negativePrompt, clip: ["1", 1] }, class_type: "CLIPTextEncode", _meta: { title: "Negative Prompt" } },
    "4": { inputs: { width: params.width, height: params.height, batch_size: 1 }, class_type: "EmptyLatentImage", _meta: { title: "Empty Latent" } },
    "5": {
      inputs: {
        seed: Math.floor(Math.random() * 2147483647),
        control_after_generate: "randomize",
        steps: params.steps,
        cfg: params.cfg,
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
    }
  };

  if (!refine) {
    base["6"] = { inputs: { samples: ["5", 0], vae: ["1", 2] }, class_type: "VAEDecode", _meta: { title: "VAE Decode" } };
    base["7"] = { inputs: { filename_prefix: "ComfyUI_BYOK_Txt2Img", images: ["6", 0] }, class_type: "SaveImage", _meta: { title: "Save Image" } };
    return base;
  }

  base["6"] = { inputs: { samples: ["5", 0], vae: ["1", 2] }, class_type: "VAEDecode", _meta: { title: "Decode Base" } };
  base["7"] = { inputs: { pixels: ["6", 0], vae: ["1", 2] }, class_type: "VAEEncode", _meta: { title: "Re-Encode" } };
  base["8"] = {
    inputs: {
      seed: Math.floor(Math.random() * 2147483647),
      control_after_generate: "randomize",
      steps: Math.max(10, Math.round(params.steps * 0.45)),
      cfg: Math.max(4, params.cfg - 1),
      sampler_name: "euler",
      scheduler: "normal",
      denoise: 0.35,
      model: ["1", 0],
      positive: ["2", 0],
      negative: ["3", 0],
      latent_image: ["7", 0]
    },
    class_type: "KSampler",
    _meta: { title: "Refine KSampler" }
  };
  base["9"] = { inputs: { samples: ["8", 0], vae: ["1", 2] }, class_type: "VAEDecode", _meta: { title: "Decode Refined" } };
  base["10"] = { inputs: { filename_prefix: "ComfyUI_BYOK_Txt2Img_Refine", images: ["9", 0] }, class_type: "SaveImage", _meta: { title: "Save Image" } };
  return base;
}

function compileImg2Img(params) {
  return {
    "1": { inputs: { ckpt_name: params.ckptName }, class_type: "CheckpointLoaderSimple", _meta: { title: "Load Checkpoint" } },
    "2": { inputs: { image: params.sourceImage, upload: "image" }, class_type: "LoadImage", _meta: { title: "Load Image" } },
    "3": { inputs: { pixels: ["2", 0], vae: ["1", 2] }, class_type: "VAEEncode", _meta: { title: "VAE Encode" } },
    "4": { inputs: { text: params.prompt, clip: ["1", 1] }, class_type: "CLIPTextEncode", _meta: { title: "Positive Prompt" } },
    "5": { inputs: { text: params.negativePrompt, clip: ["1", 1] }, class_type: "CLIPTextEncode", _meta: { title: "Negative Prompt" } },
    "6": {
      inputs: {
        seed: Math.floor(Math.random() * 2147483647),
        control_after_generate: "randomize",
        steps: params.steps,
        cfg: params.cfg,
        sampler_name: "euler",
        scheduler: "normal",
        denoise: params.denoise,
        model: ["1", 0],
        positive: ["4", 0],
        negative: ["5", 0],
        latent_image: ["3", 0]
      },
      class_type: "KSampler",
      _meta: { title: "KSampler" }
    },
    "7": { inputs: { samples: ["6", 0], vae: ["1", 2] }, class_type: "VAEDecode", _meta: { title: "VAE Decode" } },
    "8": { inputs: { filename_prefix: "ComfyUI_BYOK_Img2Img", images: ["7", 0] }, class_type: "SaveImage", _meta: { title: "Save Image" } }
  };
}

function compileUpscale(params) {
  return {
    "1": { inputs: { image: params.sourceImage, upload: "image" }, class_type: "LoadImage", _meta: { title: "Load Image" } },
    "2": { inputs: { model_name: "4x-UltraSharp.pth" }, class_type: "UpscaleModelLoader", _meta: { title: "Load Upscale Model" } },
    "3": { inputs: { image: ["1", 0], upscale_model: ["2", 0] }, class_type: "ImageUpscaleWithModel", _meta: { title: "AI Upscale" } },
    "4": { inputs: { image: ["3", 0], upscale_method: "bicubic", scale_by: params.upscaleFactor }, class_type: "ImageScaleBy", _meta: { title: "Scale Image" } },
    "5": { inputs: { filename_prefix: "ComfyUI_BYOK_Upscale", images: ["4", 0] }, class_type: "SaveImage", _meta: { title: "Save Image" } }
  };
}

function compileWorkflowFromPromptAndDraft(prompt, draftObject) {
  const hints = parsePromptHints(prompt);
  const draftWorkflow = draftObject?.workflow && typeof draftObject.workflow === "object" ? draftObject.workflow : draftObject;

  const merged = {
    ...hints,
    prompt: String(draftObject?.prompt || readFirstNodeInput(draftWorkflow, "CLIPTextEncode", "text") || hints.prompt),
    negativePrompt: String(draftObject?.negativePrompt || hints.negativePrompt),
    steps: clamp(Number(draftObject?.steps || readFirstNodeInput(draftWorkflow, "KSampler", "steps") || hints.steps), 1, 80),
    cfg: clamp(Number(draftObject?.cfg || readFirstNodeInput(draftWorkflow, "KSampler", "cfg") || hints.cfg), 1, 20),
    denoise: clamp(Number(draftObject?.denoise || readFirstNodeInput(draftWorkflow, "KSampler", "denoise") || hints.denoise), 0.05, 1),
    width: clamp(Number(draftObject?.width || hints.width), 512, 1536),
    height: clamp(Number(draftObject?.height || hints.height), 512, 1536),
    upscaleFactor: clamp(Number(draftObject?.upscaleFactor || hints.upscaleFactor), 1.5, 4),
    sourceImage: String(draftObject?.sourceImage || hints.sourceImage),
    ckptName: String(draftObject?.ckpt_name || hints.ckptName)
  };

  const intent = detectIntent(prompt, draftObject);
  if (intent === "upscale") {
    return { intent, workflow: compileUpscale(merged) };
  }
  if (intent === "img2img") {
    return { intent, workflow: compileImg2Img(merged) };
  }
  if (intent === "txt2img_refine") {
    return { intent, workflow: compileTxt2Img(merged, true) };
  }
  return { intent: "txt2img", workflow: compileTxt2Img(merged, false) };
}

function normalizeRuntimeDefaults(workflow, preferredCkptName) {
  const output = {};
  for (const [id, node] of Object.entries(workflow || {})) {
    const inputs = { ...(node.inputs || {}) };

    if (node.class_type === "CheckpointLoaderSimple") {
      inputs.ckpt_name = preferredCkptName || inputs.ckpt_name || DEFAULT_CKPT_NAME;
    }

    if (node.class_type === "KSampler") {
      const cfgNum = Number(inputs.cfg);
      const denoiseNum = Number(inputs.denoise);
      const stepsNum = Number(inputs.steps);
      const seedNum = Number(inputs.seed);

      if (!Number.isFinite(seedNum)) inputs.seed = Math.floor(Math.random() * 2147483647);
      if (!Number.isFinite(stepsNum)) inputs.steps = 30;
      if (!Number.isFinite(cfgNum)) inputs.cfg = 7;
      if (!Number.isFinite(denoiseNum)) inputs.denoise = 1;

      const samplerName = String(inputs.sampler_name || "").trim();
      const schedulerName = String(inputs.scheduler || "").trim();

      if (!SCHEDULERS.has(schedulerName) && SCHEDULERS.has(samplerName)) {
        inputs.scheduler = samplerName;
        inputs.sampler_name = "euler";
      } else {
        if (!SCHEDULERS.has(schedulerName)) inputs.scheduler = "normal";
        if (!samplerName || SCHEDULERS.has(samplerName)) inputs.sampler_name = "euler";
      }

      if (!inputs.control_after_generate) {
        inputs.control_after_generate = "randomize";
      }
    }

    output[String(id)] = {
      ...node,
      inputs
    };
  }
  return output;
}

async function generateWithOpenAI(prompt, apiKey) {
  const normalizedKey = normalizeApiKey(apiKey);
  const response = await fetchWithTimeout("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${normalizedKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4.1-mini",
      max_output_tokens: LLM_MAX_TOKENS,
      input: [
        { role: "system", content: workflowSystemPrompt() },
        { role: "user", content: prompt || "" }
      ]
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  return extractJsonObject(data.output_text || "");
}

async function generateWithOpenRouter(prompt, apiKey, model = DEFAULT_OPENROUTER_MODEL) {
  const normalizedKey = normalizeApiKey(apiKey);
  const modelsToTry = [model, "openai/gpt-4o-mini"];
  let lastError = null;

  for (const modelName of modelsToTry) {
    try {
      const response = await fetchWithTimeout("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${normalizedKey}`,
          "X-API-Key": normalizedKey,
          "Content-Type": "application/json",
          "HTTP-Referer": process.env.OPENROUTER_SITE_URL || "http://localhost",
          "X-Title": process.env.OPENROUTER_APP_NAME || "ComfyUI Workflow Generator"
        },
        body: JSON.stringify({
          model: modelName,
          messages: [
            { role: "system", content: workflowSystemPrompt() },
            { role: "user", content: prompt || "" }
          ],
          temperature: 0.2,
          max_tokens: LLM_MAX_TOKENS,
          response_format: { type: "json_object" }
        })
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
      }

      const data = await response.json();
      return extractJsonObject(data.choices?.[0]?.message?.content || "");
    } catch (error) {
      lastError = error;
      if (!/timed out/i.test(String(error?.message || ""))) {
        throw error;
      }
    }
  }

  throw lastError || new Error("BYOK request failed for OpenRouter.");
}

async function generateWithGoogle(prompt, apiKey) {
  const normalizedKey = normalizeApiKey(apiKey);
  const response = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(normalizedKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        generationConfig: { temperature: 0.2, maxOutputTokens: LLM_MAX_TOKENS },
        contents: [
          {
            role: "user",
            parts: [
              {
                text: `${workflowSystemPrompt()}\n\nUser prompt:\n${prompt || ""}`
              }
            ]
          }
        ]
      })
    }
  );

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  return extractJsonObject(data.candidates?.[0]?.content?.parts?.[0]?.text || "");
}

async function generateDraftWithProvider(provider, prompt, apiKey, options = {}) {
  if (provider === "openai") {
    return generateWithOpenAI(prompt, apiKey);
  }
  if (provider === "google") {
    return generateWithGoogle(prompt, apiKey);
  }
  return generateWithOpenRouter(prompt, apiKey, options.openrouterModel);
}

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed." });
    return;
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    const prompt = body.prompt || "";
    const provider = (body.provider || "openrouter").toLowerCase();
    const openrouterModel = body.openrouterModel || DEFAULT_OPENROUTER_MODEL;
    const apiKey =
      body.apiKey ||
      req.headers["x-openrouter-api-key"] ||
      req.headers["x-openai-api-key"] ||
      req.headers["x-google-api-key"] ||
      "";

    if (!apiKey) {
      res.status(400).json({ error: "API key is required. Provide a key and choose openrouter, openai, or google." });
      return;
    }

    if (provider === "local") {
      res.status(400).json({ error: "provider=local is disabled. Use openrouter, openai, or google." });
      return;
    }

    const draft = await generateDraftWithProvider(provider, prompt, apiKey, { openrouterModel });
    const { intent, workflow } = compileWorkflowFromPromptAndDraft(prompt, draft);
    const normalized = validateWorkflow(normalizeRuntimeDefaults(workflow, parsePromptHints(prompt).ckptName));
    const workflowUi = apiWorkflowToUiWorkflow(normalized);

    res.status(200).json({
      workflow: normalized,
      workflowUi,
      draft,
      mode: "byok-compiled",
      provider,
      intent
    });
  } catch (error) {
    const message = error.message || "Invalid request body.";
    if (/timed out/i.test(message)) {
      res.status(504).json({ error: message });
      return;
    }
    res.status(400).json({ error: message });
  }
};
