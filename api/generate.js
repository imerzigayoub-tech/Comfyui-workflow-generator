const { buildWorkflow, parseArgs, buildWorkflowFromParsed } = require("../lib/workflow");

async function parseWithOpenAI(prompt, apiKey) {
  const schema = {
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
          schema
        }
      }
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`BYOK request failed (${response.status}): ${body.slice(0, 300)}`);
  }

  const data = await response.json();
  const text = data.output_text || "";
  return JSON.parse(text);
}

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed." });
    return;
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    const prompt = body.prompt || "";
    const apiKey = body.apiKey || req.headers["x-openai-api-key"] || "";

    if (apiKey) {
      const parsed = await parseWithOpenAI(prompt, apiKey);
      const workflow = buildWorkflowFromParsed(parsed);
      res.status(200).json({ workflow, mode: "byok" });
      return;
    }

    const workflow = buildWorkflow(prompt);
    const parsed = parseArgs(prompt);
    res.status(200).json({ workflow, mode: "local-parser", parsed });
  } catch (error) {
    res.status(400).json({ error: error.message || "Invalid request body." });
  }
};
