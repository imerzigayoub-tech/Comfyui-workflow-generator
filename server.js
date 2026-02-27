const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");

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
    height,
  };
}

function buildWorkflow(promptText) {
  const parsed = parseArgs(promptText);

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

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body)
  });
  res.end(body);
}

function serveStatic(req, res) {
  const parsedUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const pathname = parsedUrl.pathname === "/" ? "/index.html" : parsedUrl.pathname;
  const relativePath = path.normalize(pathname).replace(/^([/\\])+/, "");
  const filePath = path.resolve(PUBLIC_DIR, relativePath);

  if (!filePath.startsWith(path.resolve(PUBLIC_DIR))) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const mime = {
      ".html": "text/html; charset=utf-8",
      ".css": "text/css; charset=utf-8",
      ".js": "application/javascript; charset=utf-8"
    }[ext] || "application/octet-stream";

    res.writeHead(200, { "Content-Type": mime });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/api/generate") {
    let body = "";
    req.on("data", chunk => {
      body += chunk;
      if (body.length > 1_000_000) {
        req.destroy();
      }
    });

    req.on("end", () => {
      try {
        const payload = JSON.parse(body || "{}");
        const workflow = buildWorkflow(payload.prompt || "");
        sendJson(res, 200, { workflow });
      } catch (error) {
        sendJson(res, 400, { error: "Invalid JSON body." });
      }
    });
    return;
  }

  if (req.method === "GET") {
    serveStatic(req, res);
    return;
  }

  res.writeHead(405);
  res.end("Method not allowed");
});

server.listen(PORT, () => {
  console.log(`ComfyUI workflow generator running at http://localhost:${PORT}`);
});
