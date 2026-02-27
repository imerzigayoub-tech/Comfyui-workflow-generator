const promptInput = document.getElementById("prompt");
const output = document.getElementById("output");
const generateBtn = document.getElementById("generateBtn");
const downloadBtn = document.getElementById("downloadBtn");

let latestWorkflow = null;

async function generateWorkflow() {
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptInput.value })
    });

    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }

    const data = await response.json();
    latestWorkflow = data.workflow;
    output.textContent = JSON.stringify(latestWorkflow, null, 2);
    downloadBtn.disabled = false;
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
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
  link.download = "workflow.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

generateBtn.addEventListener("click", generateWorkflow);
downloadBtn.addEventListener("click", downloadWorkflow);
