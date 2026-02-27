const promptInput = document.getElementById("prompt");
const output = document.getElementById("output");
const providerInput = document.getElementById("provider");
const templateInput = document.getElementById("template");
const apiKeyInput = document.getElementById("apiKey");
const generateBtn = document.getElementById("generateBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusText = document.getElementById("status");

let latestWorkflow = null;
let latestWorkflowApi = null;

async function generateWorkflow() {
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptInput.value,
        provider: providerInput.value,
        template: templateInput.value,
        apiKey: apiKeyInput.value.trim()
      })
    });

    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }

    const data = await response.json();
    latestWorkflowApi = data.workflow || null;
    latestWorkflow = data.workflowUi || data.workflow || null;
    output.textContent = JSON.stringify(latestWorkflow, null, 2);
    statusText.textContent = `Mode: ${data.mode || "unknown"} | Template: ${data.template || "unknown"}`;
    downloadBtn.disabled = false;
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
    statusText.textContent = "Mode: error | Template: n/a";
    latestWorkflow = null;
    downloadBtn.disabled = true;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate JSON";
  }
}

function downloadWorkflow() {
  if (!latestWorkflow) {
    return;
  }

  const blob = new Blob([JSON.stringify(latestWorkflow, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "workflow-ui.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

generateBtn.addEventListener("click", generateWorkflow);
downloadBtn.addEventListener("click", downloadWorkflow);
