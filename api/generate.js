const { buildWorkflow, parseArgs, buildWorkflowFromParsed } = require("../lib/workflow");

const PARAM_JSON_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    prompt: { type: "string" },
    negativePrompt: { type: "string" },
    steps: { type: "integer", minimum: 1, maximum: 80 },
    cfg: { type: "number", minimum: 1, maximum: 20 },
    width: { type: "integer", minimum: 512, maximum: 1536 },
    height: { type: "integer", minimum: 512, maximum: 1536 }
  },
  required: ["prompt", "negativePrompt", "steps", "cfg", "width", "height"]
};

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

async function parseWithOpenAI(prompt, apiKey) {
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4.1-mini",
      input: [
        {
          role: "system",
          content:
            "Convert a user image prompt into ComfyUI generation params. Keep intent, return only valid JSON."
        },
        {
          role: "user",
          content: prompt || ""
        }
      ],
      text: {
        format: {
          type: "json_schema",
          name: "comfy_params",
          strict: true,
          schema: PARAM_JSON_SCHEMA
        }
      }
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  return extractJsonObject(data.output_text || "");
}

async function parseWithOpenRouter(prompt, apiKey) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "Return ONLY a JSON object with keys: prompt, negativePrompt, steps, cfg, width, height. No markdown."
        },
        {
          role: "user",
          content: prompt || ""
        }
      ],
      temperature: 0.2
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  const text = data.choices?.[0]?.message?.content || "";
  return extractJsonObject(text);
}

async function parseWithGoogle(prompt, apiKey) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        generationConfig: { temperature: 0.2 },
        contents: [
          {
            role: "user",
            parts: [
              {
                text:
                  "Return ONLY a JSON object with keys: prompt, negativePrompt, steps, cfg, width, height. No markdown.\n\nPrompt:\n" +
                  (prompt || "")
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
  return extractJsonObject(text);
}

async function parseWithProvider(provider, prompt, apiKey) {
  if (provider === "openai") {
    return parseWithOpenAI(prompt, apiKey);
  }
  if (provider === "google") {
    return parseWithGoogle(prompt, apiKey);
  }
  return parseWithOpenRouter(prompt, apiKey);
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
    const apiKey =
      body.apiKey ||
      req.headers["x-openrouter-api-key"] ||
      req.headers["x-openai-api-key"] ||
      req.headers["x-google-api-key"] ||
      "";

    if (apiKey && provider !== "local") {
      const parsed = await parseWithProvider(provider, prompt, apiKey);
      const workflow = buildWorkflowFromParsed(parsed);
      res.status(200).json({ workflow, mode: "byok", provider });
      return;
    }

    const workflow = buildWorkflow(prompt);
    const parsed = parseArgs(prompt);
    res.status(200).json({ workflow, mode: "local-parser", parsed });
  } catch (error) {
    res.status(400).json({ error: error.message || "Invalid request body." });
  }
};
