const http = require("http");
const fs = require("fs");
const path = require("path");

const { buildWorkflow } = require("./lib/workflow");

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");

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
        const requestedTemplate = (payload.template || "auto").toLowerCase();
        const { template, workflow } = buildWorkflow(payload.prompt || "", requestedTemplate);
        sendJson(res, 200, { workflow, mode: "local-parser", template });
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
