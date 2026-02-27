const { apiWorkflowToUiWorkflow } = require("../lib/workflow");
const DEFAULT_OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || "openai/gpt-4o-mini";
const DEFAULT_OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434";
const DEFAULT_OLLAMA_MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b";
const REQUEST_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 120000);
const LLM_MAX_TOKENS = Number(process.env.LLM_MAX_TOKENS || 1200);

function normalizeApiKey(raw) {
  const value = String(raw || "").trim();
  if (!value) return "";
  return value.replace(/^Bearer\s+/i, "").replace(/^['\"]|['\"]$/g, "").trim();
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

function workflowSystemPrompt() {
  return [
    "Generate a ComfyUI API workflow from the user prompt.",
    "Return JSON only.",
    "Output format must be: { intent, workflow_api }",
    "workflow_api must be ComfyUI API prompt format: object keyed by node id strings.",
    "Each node must include: class_type, inputs, _meta.title.",
    "Linked inputs must be in format: [\"<node_id>\", <output_index>].",
    "No markdown, no explanation, no extra text.",
    "Use node names and widgets that are compatible with ComfyUI defaults."
  ].join(" ");
}

function validateWorkflowUi(workflowUi) {
  if (!workflowUi || typeof workflowUi !== "object" || Array.isArray(workflowUi)) {
    throw new Error("workflow_ui is not an object.");
  }

  if (!Array.isArray(workflowUi.nodes) || workflowUi.nodes.length < 3) {
    throw new Error("workflow_ui.nodes is missing or too small.");
  }

  if (!Array.isArray(workflowUi.links) || workflowUi.links.length < 2) {
    throw new Error("workflow_ui.links is missing or too small.");
  }

  const nodeIds = new Set(workflowUi.nodes.map(node => Number(node?.id)).filter(Number.isFinite));
  if (!nodeIds.size) {
    throw new Error("workflow_ui.nodes has invalid ids.");
  }

  for (const link of workflowUi.links) {
    if (!Array.isArray(link) || link.length < 6) {
      throw new Error("workflow_ui has invalid link shape.");
    }
    const fromId = Number(link[1]);
    const toId = Number(link[3]);
    if (!nodeIds.has(fromId) || !nodeIds.has(toId)) {
      throw new Error("workflow_ui.links references unknown node ids.");
    }
  }

  return workflowUi;
}

function validateWorkflowApi(workflowApi) {
  if (!workflowApi || typeof workflowApi !== "object" || Array.isArray(workflowApi)) {
    throw new Error("workflow_api is not an object.");
  }
  const ids = Object.keys(workflowApi);
  if (ids.length < 3) {
    throw new Error("workflow_api has too few nodes.");
  }
  for (const id of ids) {
    const node = workflowApi[id];
    if (!node || typeof node !== "object") throw new Error(`workflow_api node ${id} is invalid.`);
    if (!node.class_type) throw new Error(`workflow_api node ${id} missing class_type.`);
    if (!node.inputs || typeof node.inputs !== "object") throw new Error(`workflow_api node ${id} missing inputs.`);
  }
  return workflowApi;
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
  if (!normalizedKey) {
    throw new Error("OpenRouter API key is empty.");
  }

  const modelsToTry = [model, "openai/gpt-4o-mini"];
  let lastError = null;

  for (const candidateModel of modelsToTry) {
    try {
      const response = await fetchWithTimeout("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${normalizedKey}`,
          "X-API-Key": normalizedKey,
          "x-api-key": normalizedKey,
          "Content-Type": "application/json",
          "HTTP-Referer": process.env.OPENROUTER_SITE_URL || "http://localhost",
          "X-Title": process.env.OPENROUTER_APP_NAME || "ComfyUI Workflow Generator"
        },
        body: JSON.stringify({
          model: candidateModel,
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
        if (response.status === 401) {
          throw new Error(`OpenRouter auth failed (401). Check key/provider. Body: ${body.slice(0, 220)}`);
        }
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

  throw lastError || new Error("OpenRouter request failed.");
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
            parts: [{ text: `${workflowSystemPrompt()}\n\nUser prompt:\n${prompt || ""}` }]
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

async function generateWithOllama(prompt, options = {}) {
  const baseUrl = String(options.ollamaBaseUrl || DEFAULT_OLLAMA_BASE_URL).replace(/\/+$/, "");
  const model = options.ollamaModel || DEFAULT_OLLAMA_MODEL;
  const response = await fetchWithTimeout(`${baseUrl}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt: `${workflowSystemPrompt()}\n\nUser prompt:\n${prompt || ""}`,
      format: "json",
      stream: false,
      options: {
        temperature: 0.2
      }
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  return extractJsonObject(data.response || "");
}

async function generateWithProvider(provider, prompt, apiKey, options = {}) {
  if (provider === "openai") return generateWithOpenAI(prompt, apiKey);
  if (provider === "google") return generateWithGoogle(prompt, apiKey);
  if (provider === "ollama") return generateWithOllama(prompt, options);
  return generateWithOpenRouter(prompt, apiKey, options.openrouterModel || DEFAULT_OPENROUTER_MODEL);
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
    const ollamaModel = body.ollamaModel || DEFAULT_OLLAMA_MODEL;
    const ollamaBaseUrl = body.ollamaBaseUrl || DEFAULT_OLLAMA_BASE_URL;
    const apiKey =
      body.apiKey ||
      req.headers["x-openrouter-api-key"] ||
      req.headers["x-openai-api-key"] ||
      req.headers["x-google-api-key"] ||
      "";

    if (provider !== "ollama" && !apiKey) {
      res.status(400).json({ error: "API key is required. Provide a key and choose openrouter, openai, or google." });
      return;
    }

    if (provider === "local") {
      res.status(400).json({ error: "provider=local is disabled. Use openrouter, openai, google, or ollama." });
      return;
    }

    const result = await generateWithProvider(provider, prompt, apiKey, { openrouterModel, ollamaModel, ollamaBaseUrl });
    const workflowApi = validateWorkflowApi(result.workflow_api || result.workflow || result);
    const workflowUi = validateWorkflowUi(apiWorkflowToUiWorkflow(workflowApi));

    res.status(200).json({
      workflow: workflowApi,
      workflowUi,
      draft: result,
      mode: "byok-direct",
      provider,
      intent: result.intent || "unknown"
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
