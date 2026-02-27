const { buildWorkflow, parseArgs, resolveTemplate, apiWorkflowToUiWorkflow } = require("../lib/workflow");

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

async function generateWithOpenRouter(prompt, apiKey) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
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

async function generateWorkflowWithProvider(provider, prompt, apiKey) {
  if (provider === "openai") {
    return generateWithOpenAI(prompt, apiKey);
  }
  if (provider === "google") {
    return generateWithGoogle(prompt, apiKey);
  }
  return generateWithOpenRouter(prompt, apiKey);
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
    const requestedTemplate = (body.template || "auto").toLowerCase();
    const apiKey =
      body.apiKey ||
      req.headers["x-openrouter-api-key"] ||
      req.headers["x-openai-api-key"] ||
      req.headers["x-google-api-key"] ||
      "";

    if (apiKey && provider !== "local") {
      const workflowApi = normalizeWorkflowShape(await generateWorkflowWithProvider(provider, prompt, apiKey));
      const workflow = validateWorkflow(workflowApi);
      const workflowUi = apiWorkflowToUiWorkflow(workflow);
      res.status(200).json({ workflow, workflowUi, mode: "byok-generated", provider, template: "llm-generated" });
      return;
    }

    const parsed = parseArgs(prompt);
    const { template, workflow } = buildWorkflow(prompt, requestedTemplate);
    const workflowUi = apiWorkflowToUiWorkflow(workflow);
    res.status(200).json({
      workflow,
      workflowUi,
      mode: "local-parser",
      parsed,
      template: template || resolveTemplate(requestedTemplate, parsed.prompt)
    });
  } catch (error) {
    res.status(400).json({ error: error.message || "Invalid request body." });
  }
};
