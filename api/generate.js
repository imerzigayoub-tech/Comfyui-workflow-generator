const { parseArgs, apiWorkflowToUiWorkflow } = require("../lib/workflow");
const DEFAULT_CKPT_NAME = process.env.DEFAULT_CKPT_NAME || "sd_xl_base_1.0.safetensors";
const SCHEDULERS = new Set(["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"]);

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
  }

  let linkRefCount = 0;
  for (const id of nodeIds) {
    const node = workflow[id];
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

function normalizeWorkflowShape(candidate) {
  if (
    candidate &&
    typeof candidate === "object" &&
    !Array.isArray(candidate) &&
    candidate.workflow &&
    typeof candidate.workflow === "object" &&
    !Array.isArray(candidate.workflow)
  ) {
    return candidate.workflow;
  }
  return candidate;
}

function workflowSystemPrompt() {
  return [
    "You generate ComfyUI workflow JSON from user intent.",
    "Return ONLY one JSON object (no markdown, no explanation).",
    "The object must be a valid ComfyUI workflow graph: keys are node ids as strings.",
    "Each node must have: class_type, inputs, _meta.title.",
    "IMPORTANT: linked inputs MUST use exact reference format: [\"<node_id>\", <output_index>].",
    "Example: KSampler inputs.model = [\"1\", 0], positive = [\"2\", 0].",
    "Prefer common built-in nodes only.",
    "Use one of these structures depending on intent:",
    "- txt2img: CheckpointLoaderSimple, CLIPTextEncode (pos/neg), EmptyLatentImage, KSampler, VAEDecode, SaveImage",
    "- img2img: include LoadImage and VAEEncode before KSampler",
    "- upscale: include LoadImage, UpscaleModelLoader, ImageUpscaleWithModel, SaveImage",
    "If uncertain, choose txt2img.",
    "Set reasonable defaults for steps/cfg/size/seed.",
    "Output strictly JSON only."
  ].join(" ");
}

function normalizeLinkValue(value) {
  if (Array.isArray(value) && value.length === 2) {
    return [String(value[0]), Number(value[1]) || 0];
  }

  if (value && typeof value === "object") {
    const nodeId = value.node ?? value.node_id ?? value.id ?? value.from;
    const slot = value.output ?? value.slot ?? value.index ?? value.out;
    if (nodeId !== undefined && slot !== undefined) {
      return [String(nodeId), Number(slot) || 0];
    }
  }

  if (typeof value === "string") {
    const match = value.match(/^\s*(\d+)\s*[:|,]\s*(\d+)\s*$/);
    if (match) {
      return [match[1], Number(match[2]) || 0];
    }
  }

  return value;
}

function normalizeWorkflowLinks(workflow) {
  const normalized = {};
  for (const [id, node] of Object.entries(workflow || {})) {
    const inputs = {};
    for (const [name, value] of Object.entries(node.inputs || {})) {
      inputs[name] = normalizeLinkValue(value);
    }
    normalized[String(id)] = {
      ...node,
      inputs
    };
  }
  return normalized;
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

const REQUIRED_LINK_INPUTS = {
  CLIPTextEncode: ["clip"],
  KSampler: ["model", "positive", "negative", "latent_image"],
  VAEDecode: ["samples", "vae"],
  SaveImage: ["images"],
  VAEEncode: ["pixels", "vae"],
  ImageUpscaleWithModel: ["image", "upscale_model"],
  ImageScaleBy: ["image"]
};

function isValidLinkRef(value, nodeIdsSet) {
  if (!(Array.isArray(value) && value.length === 2)) {
    return false;
  }
  const fromId = String(value[0]);
  const slot = Number(value[1]);
  return nodeIdsSet.has(fromId) && Number.isFinite(slot) && slot >= 0;
}

function hasBrokenLinks(workflow) {
  const nodeIds = Object.keys(workflow || {});
  const nodeIdsSet = new Set(nodeIds);

  for (const id of nodeIds) {
    const node = workflow[id];
    const inputs = node.inputs || {};

    for (const value of Object.values(inputs)) {
      if (Array.isArray(value) && !isValidLinkRef(value, nodeIdsSet)) {
        return true;
      }
    }

    const required = REQUIRED_LINK_INPUTS[node.class_type] || [];
    for (const inputName of required) {
      if (!isValidLinkRef(inputs[inputName], nodeIdsSet)) {
        return true;
      }
    }
  }

  return false;
}

function sortIds(ids) {
  return [...ids].sort((a, b) => Number(a) - Number(b));
}

function findNodeIdsByType(workflow, type) {
  return sortIds(Object.keys(workflow || {}).filter(id => workflow[id]?.class_type === type));
}

function pickNodeId(workflow, type, mode = "last") {
  const ids = findNodeIdsByType(workflow, type);
  if (!ids.length) return null;
  return mode === "first" ? ids[0] : ids[ids.length - 1];
}

function setLinkInput(workflow, toId, inputName, fromId, outputIndex) {
  if (!toId || !fromId) return;
  const node = workflow[toId];
  if (!node || !node.inputs) return;
  node.inputs[inputName] = [String(fromId), Number(outputIndex) || 0];
}

function autoRepairLinks(workflow) {
  const repaired = JSON.parse(JSON.stringify(workflow || {}));
  const ids = sortIds(Object.keys(repaired));

  const ckptId = pickNodeId(repaired, "CheckpointLoaderSimple");
  const emptyLatentId = pickNodeId(repaired, "EmptyLatentImage");
  const vaeEncodeId = pickNodeId(repaired, "VAEEncode");
  const loadImageId = pickNodeId(repaired, "LoadImage");
  const upscaleModelId = pickNodeId(repaired, "UpscaleModelLoader");
  const upscaleImageId = pickNodeId(repaired, "ImageUpscaleWithModel");
  const imageScaleId = pickNodeId(repaired, "ImageScaleBy");
  const vaeDecodeId = pickNodeId(repaired, "VAEDecode");
  const ksamplerId = pickNodeId(repaired, "KSampler");
  const saveImageId = pickNodeId(repaired, "SaveImage");

  const clipIds = findNodeIdsByType(repaired, "CLIPTextEncode");
  const positiveClipId = clipIds.find(id => /positive/i.test(String(repaired[id]?._meta?.title || ""))) || clipIds[0] || null;
  const negativeClipId = clipIds.find(id => /negative/i.test(String(repaired[id]?._meta?.title || ""))) || clipIds[1] || clipIds[0] || null;

  for (const id of ids) {
    const node = repaired[id];
    if (!node || !node.inputs) continue;

    if (node.class_type === "CLIPTextEncode") {
      setLinkInput(repaired, id, "clip", ckptId, 1);
    }

    if (node.class_type === "KSampler") {
      setLinkInput(repaired, id, "model", ckptId, 0);
      setLinkInput(repaired, id, "positive", positiveClipId, 0);
      setLinkInput(repaired, id, "negative", negativeClipId, 0);
      setLinkInput(repaired, id, "latent_image", vaeEncodeId || emptyLatentId, 0);
    }

    if (node.class_type === "VAEEncode") {
      setLinkInput(repaired, id, "pixels", loadImageId, 0);
      setLinkInput(repaired, id, "vae", ckptId, 2);
    }

    if (node.class_type === "VAEDecode") {
      setLinkInput(repaired, id, "samples", ksamplerId, 0);
      setLinkInput(repaired, id, "vae", ckptId, 2);
    }

    if (node.class_type === "ImageUpscaleWithModel") {
      setLinkInput(repaired, id, "image", loadImageId || vaeDecodeId, 0);
      setLinkInput(repaired, id, "upscale_model", upscaleModelId, 0);
    }

    if (node.class_type === "ImageScaleBy") {
      setLinkInput(repaired, id, "image", upscaleImageId || loadImageId || vaeDecodeId, 0);
    }

    if (node.class_type === "SaveImage") {
      setLinkInput(repaired, id, "images", vaeDecodeId || imageScaleId || upscaleImageId, 0);
    }
  }

  if (saveImageId) {
    setLinkInput(
      repaired,
      saveImageId,
      "images",
      pickNodeId(repaired, "VAEDecode") || pickNodeId(repaired, "ImageScaleBy") || pickNodeId(repaired, "ImageUpscaleWithModel"),
      0
    );
  }

  return repaired;
}

async function generateWithOpenAI(prompt, apiKey) {
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4.1-mini",
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
  return validateWorkflow(extractJsonObject(data.output_text || ""));
}

async function generateWithOpenRouter(prompt, apiKey, model = process.env.OPENROUTER_MODEL || "openrouter/auto") {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": process.env.OPENROUTER_SITE_URL || "http://localhost",
      "X-Title": process.env.OPENROUTER_APP_NAME || "ComfyUI Workflow Generator"
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: workflowSystemPrompt() },
        { role: "user", content: prompt || "" }
      ],
      temperature: 0.3
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  const text = data.choices?.[0]?.message?.content || "";
  return validateWorkflow(extractJsonObject(text));
}

async function generateWithGoogle(prompt, apiKey) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        generationConfig: { temperature: 0.3 },
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
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  return validateWorkflow(extractJsonObject(text));
}

async function generateWorkflowWithProvider(provider, prompt, apiKey, options = {}) {
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
    const openrouterModel = body.openrouterModel || process.env.OPENROUTER_MODEL || "openrouter/auto";
    const apiKey =
      body.apiKey ||
      req.headers["x-openrouter-api-key"] ||
      req.headers["x-openai-api-key"] ||
      req.headers["x-google-api-key"] ||
      "";

    if (!apiKey) {
      res.status(400).json({
        error: "API key is required. Provide a key and choose openrouter, openai, or google."
      });
      return;
    }

    if (provider === "local") {
      res.status(400).json({
        error: "provider=local is disabled. Use openrouter, openai, or google."
      });
      return;
    }

    const parsedPrompt = parseArgs(prompt);
    const workflowApi = normalizeWorkflowShape(
      await generateWorkflowWithProvider(provider, prompt, apiKey, { openrouterModel })
    );
    let workflowCandidate = validateWorkflow(
      normalizeRuntimeDefaults(normalizeWorkflowLinks(workflowApi), parsedPrompt.ckptName || DEFAULT_CKPT_NAME)
    );
    if (hasBrokenLinks(workflowCandidate)) {
      workflowCandidate = validateWorkflow(autoRepairLinks(workflowCandidate));
      if (hasBrokenLinks(workflowCandidate)) {
        res.status(422).json({
          error: "Provider returned incomplete links and auto-repair could not resolve all required connections."
        });
        return;
      }
    }

    const workflow = workflowCandidate;
    const workflowUi = apiWorkflowToUiWorkflow(workflow);
    res.status(200).json({ workflow, workflowUi, mode: "byok-generated", provider, template: "llm-generated" });
  } catch (error) {
    res.status(400).json({ error: error.message || "Invalid request body." });
  }
};
