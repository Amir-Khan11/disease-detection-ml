const messages = document.getElementById("messages");
const form = document.getElementById("composer-form");
const input = document.getElementById("message-input");
const promptPills = document.getElementById("prompt-pills");
const chatColumn = document.getElementById("chat-column");
const heroCard = document.getElementById("hero-card");
const assistantTemplate = document.getElementById("assistant-template");
const userTemplate = document.getElementById("user-template");
const typingTemplate = document.getElementById("typing-template");
const summaryDialog = document.getElementById("summary-dialog");
const plotsDialog = document.getElementById("plots-dialog");
const summaryContent = document.getElementById("summary-content");

function scrollToBottom(force = false) {
  window.requestAnimationFrame(() => {
    messages.scrollTo({
      top: messages.scrollHeight,
      behavior: force ? "auto" : "smooth",
    });
  });
}

function updatePromptVisibility() {
  if (chatColumn.classList.contains("initial-state")) {
    promptPills.classList.remove("hidden");
    return;
  }

  const nonTypingMessages = [...messages.children].filter(node => !node.classList.contains("typing-message"));
  const shouldShow = nonTypingMessages.length === 0;
  promptPills.classList.toggle("hidden", !shouldShow);
}

function setInitialState(isInitial) {
  chatColumn.classList.toggle("initial-state", isInitial);
  heroCard.classList.toggle("hidden", !isInitial);

  if (isInitial) {
    messages.innerHTML = "";
    promptPills.classList.remove("hidden");
    input.value = "";
    autoResize();
    scrollToBottom(true);
  }
}

function addUserMessage(text) {
  const node = userTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".user-bubble").textContent = text;
  messages.appendChild(node);
  updatePromptVisibility();
  scrollToBottom();
}

function createAssistantCard(payload) {
  const wrapper = document.createElement("div");
  wrapper.className = "assistant-layout";
  const topPrediction = payload.predictions?.[0];

  if (payload.intro) {
    const intro = document.createElement("p");
    intro.className = "assistant-intro";
    intro.textContent = payload.intro;
    wrapper.appendChild(intro);
  }

  if (topPrediction) {
    const hero = document.createElement("section");
    hero.className = "result-hero";
    hero.innerHTML = `
      <p class="result-label">Most Likely Match</p>
      <div class="result-disease">${toTitleCase(topPrediction.disease)}</div>
      <p class="result-confidence">${topPrediction.probability.toFixed(2)}% confidence</p>
    `;
    wrapper.appendChild(hero);
  }

  if (payload.recognizedSymptoms?.length) {
    const card = document.createElement("section");
    card.className = "inline-card";
    card.innerHTML = "<h3>Recognized Symptoms</h3>";

    const chipRow = document.createElement("div");
    chipRow.className = "chip-row";
    payload.recognizedSymptoms.forEach(symptom => {
      const chip = document.createElement("span");
      chip.className = "symptom-chip";
      chip.textContent = symptom;
      chipRow.appendChild(chip);
    });
    card.appendChild(chipRow);
    wrapper.appendChild(card);
  }

  if (payload.predictions?.length) {
    const card = document.createElement("section");
    card.className = "inline-card";
    card.innerHTML = "<h3>Prediction Confidence</h3>";

    const list = document.createElement("div");
    list.className = "prediction-list";
    payload.predictions.forEach(item => {
      const confidenceTone = item.probability >= 70 ? "high" : item.probability >= 40 ? "medium" : "low";
      const row = document.createElement("div");
      row.className = `prediction-row prediction-row-${confidenceTone}`;
      row.innerHTML = `
        <div class="prediction-meta">
          <span class="prediction-name">${toTitleCase(item.disease)}</span>
          <span class="prediction-value">${item.probability.toFixed(2)}%</span>
        </div>
        <div class="prediction-bar">
          <span style="width: ${Math.max(4, item.probability)}%"></span>
        </div>
      `;
      list.appendChild(row);
    });
    card.appendChild(list);
    wrapper.appendChild(card);
  }

  if (payload.severity || payload.advice || payload.clusterLabel !== undefined) {
    const card = document.createElement("section");
    card.className = "inline-card";
    card.innerHTML = "<h3>Analysis</h3>";

    const badge = document.createElement("div");
    badge.className = "analysis-badge";
    badge.textContent = "Clinical Review";
    card.appendChild(badge);

    if (payload.severity) {
      const severity = document.createElement("p");
      severity.className = "assistant-detail";
      severity.textContent = payload.severity;
      card.appendChild(severity);
    }

    if (payload.advice) {
      const advice = document.createElement("p");
      advice.className = "assistant-detail";
      advice.textContent = payload.advice;
      card.appendChild(advice);
    }

    if (payload.clusterLabel !== undefined) {
      const cluster = document.createElement("p");
      cluster.className = "assistant-detail";
      cluster.textContent = `KMeans cluster: Cluster ${payload.clusterLabel}`;
      card.appendChild(cluster);
    }

    wrapper.appendChild(card);
  }

  return wrapper;
}

function addAssistantMessage(payload) {
  const node = assistantTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".assistant-card").appendChild(createAssistantCard(payload));
  messages.appendChild(node);
  updatePromptVisibility();
  scrollToBottom();
}

function addPlainAssistantMessage(text) {
  addAssistantMessage({ intro: text });
}

function addTypingMessage() {
  const node = typingTemplate.content.firstElementChild.cloneNode(true);
  const card = node.querySelector(".typing-card");
  card.innerHTML = `
    <div class="loading-state">
      <span class="loading-spinner" aria-hidden="true"></span>
      <span class="loading-text">Analyzing symptoms...</span>
    </div>
  `;
  messages.appendChild(node);
  form.classList.add("is-loading");
  scrollToBottom();
  return node;
}

function removeTypingMessage(node) {
  if (node?.parentNode) {
    node.parentNode.removeChild(node);
  }
  form.classList.remove("is-loading");
}

function toTitleCase(text) {
  return text.replace(/\b\w/g, char => char.toUpperCase());
}

function normalizeDiagnosis(payload) {
  if (!payload.recognized_symptoms.length) {
    return {
      intro: "I could not confidently match known symptoms from that message.",
      advice: "Try including terms like fever, headache, cough, chills, rash, fatigue, vomiting, sore throat, or body pain.",
      severity: "No structured symptom pattern was detected from the current text.",
      recognizedSymptoms: [],
      predictions: [],
      clusterLabel: payload.cluster_label,
    };
  }

  return {
    intro: "I analyzed your symptom description and organized the most relevant model outputs below.",
    recognizedSymptoms: payload.recognized_symptoms,
    predictions: payload.predictions,
    advice: payload.advice,
    severity: payload.severity,
    clusterLabel: payload.cluster_label,
  };
}

async function sendMessage(message) {
  setInitialState(false);
  addUserMessage(message);
  const typingNode = addTypingMessage();

  try {
    const response = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const payload = await response.json();
    removeTypingMessage(typingNode);

    if (!response.ok) {
      addPlainAssistantMessage(payload.error || "Something went wrong while processing the request.");
      return;
    }

    addAssistantMessage(normalizeDiagnosis(payload));
  } catch (error) {
    removeTypingMessage(typingNode);
    addPlainAssistantMessage("The local server could not process that request. Please make sure the Python server is running.");
  }
}

async function loadSummary() {
  summaryContent.innerHTML = "<p>Loading ML summary...</p>";
  summaryDialog.showModal();

  try {
    const response = await fetch("/api/summary");
    const payload = await response.json();

    const distribution = Object.entries(payload.disease_counts)
      .map(([name, count]) => `<li>${toTitleCase(name)}: ${count}</li>`)
      .join("");

    summaryContent.innerHTML = `
      <div class="summary-cards">
        <div class="metric-card"><span>Dataset Rows</span><strong>${payload.dataset_rows}</strong></div>
        <div class="metric-card"><span>Training Rows</span><strong>${payload.train_rows}</strong></div>
        <div class="metric-card"><span>Testing Rows</span><strong>${payload.test_rows}</strong></div>
        <div class="metric-card"><span>Accuracy</span><strong>${payload.accuracy}%</strong></div>
        <div class="metric-card"><span>KMeans Clusters</span><strong>${payload.cluster_count}</strong></div>
      </div>
      <section>
        <h3>Class Distribution</h3>
        <ul class="distribution-list">${distribution}</ul>
      </section>
      <section>
        <h3>Dataset File</h3>
        <div class="dataset-path">${payload.dataset_path}</div>
      </section>
    `;
  } catch (error) {
    summaryContent.innerHTML = "<p>Could not load the ML summary.</p>";
  }
}

function autoResize() {
  input.style.height = "0px";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function insertPrompt(prompt) {
  input.value = prompt;
  autoResize();
  input.focus();
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }

  input.value = "";
  autoResize();
  await sendMessage(message);
});

input.addEventListener("input", autoResize);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => {
    setInitialState(true);
    insertPrompt(button.dataset.prompt);
  });
});

document.getElementById("new-chat").addEventListener("click", () => {
  setInitialState(true);
  input.focus();
});

const openSummaryButton = document.getElementById("open-summary");
if (openSummaryButton) {
  openSummaryButton.addEventListener("click", loadSummary);
}

document.getElementById("more-summary").addEventListener("click", loadSummary);

function openPlots() {
  plotsDialog.showModal();
}

document.getElementById("share-plots").addEventListener("click", openPlots);

document.querySelectorAll("[data-close]").forEach(button => {
  button.addEventListener("click", () => {
    document.getElementById(button.dataset.close).close();
  });
});

[summaryDialog, plotsDialog].forEach(dialog => {
  dialog.addEventListener("click", event => {
    if (event.target === dialog) {
      dialog.close();
    }
  });
});

setInitialState(true);
autoResize();
