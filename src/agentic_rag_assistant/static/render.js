/** Build DOM nodes without interpreting document or model text as HTML. */
export function element(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function iconButton(name, label, onClick) {
  const button = element("button", "icon-button");
  button.type = "button";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.append(icon(name));
  button.addEventListener("click", onClick);
  return button;
}

export function fileIcon(filename) {
  const node = element("span", `file-icon${filename.toLowerCase().endsWith(".pdf") ? " pdf" : ""}`);
  node.append(icon("file"));
  return node;
}

export function dateLabel(value) {
  return new Date(value).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  });
}

function assistantHeading(answered = true) {
  const heading = element("div", "assistant-heading");
  const mark = element("span", "brand-mark");
  mark.append(icon("spark"));
  heading.append(mark, element("span", "", "Agentic"));
  if (!answered) heading.append(element("span", "abstention-label", "Insufficient evidence"));
  return heading;
}

export function userMessage(question, timestamp) {
  const node = element("div", "user-message");
  node.append(element("div", "user-bubble", question));
  if (timestamp) node.append(element("span", "message-time", dateLabel(timestamp)));
  return node;
}

export function pendingMessage(question) {
  const pair = element("article", "message-pair pending-message");
  pair.setAttribute("aria-label", "Answer in progress");
  const dots = element("span", "loading-dots");
  for (let i = 0; i < 3; i++) dots.append(element("span"));
  const status = element("p", "", "Retrieving sources and preparing your answer");
  status.append(dots);
  pair.append(userMessage(question), assistantHeading(), status);
  return pair;
}

export function renderTurn(turn, notify) {
  const response = turn.response;
  const pair = element("article", "message-pair");
  pair.dataset.turn = turn.turn_number;
  pair.setAttribute("aria-label", `Conversation turn ${turn.turn_number}`);
  const answer = element("div", "answer-text");
  const sources = element("div", "source-list");
  const cards = new Map();
  for (const citation of response.citations) {
    const card = element("details", "source-card");
    const summary = element("summary");
    summary.append(element("span", "source-number", citation.source_id), icon("file"),
      element("span", "source-filename", citation.filename));
    if (citation.chunk.page_number) summary.append(element("span", "source-page", `Page ${citation.chunk.page_number}`));
    else summary.append(element("span", "source-page", "TXT"));
    card.append(summary, element("blockquote", "", citation.chunk.text));
    sources.append(card);
    cards.set(citation.source_id, card);
  }
  // Source numbers belong to this answer only; never link a different turn's sources.
  for (const part of response.answer.split(/(\[\d+\])/g)) {
    const matched = /^\[(\d+)\]$/.exec(part);
    const card = matched && cards.get(Number(matched[1]));
    if (card) {
      const marker = element("button", "citation-marker", part);
      marker.type = "button";
      marker.setAttribute("aria-label", `Show source ${matched[1]}`);
      marker.addEventListener("click", () => {
        card.open = true;
        card.scrollIntoView({ block: "nearest" });
        card.querySelector("summary").focus();
      });
      answer.append(marker);
    } else answer.append(document.createTextNode(part));
  }
  pair.append(userMessage(response.question, turn.created_at), assistantHeading(response.answered), answer);
  const symbols = { add: "+", subtract: "−", multiply: "×", divide: "÷" };
  for (const result of response.tool_results) {
    const row = element("div", "tool-result");
    row.append(icon("calc"), element("span", "", "Calculator"),
      element("strong", "", `${result.left} ${symbols[result.operation] || result.operation} ${result.right} = ${result.value}`));
    pair.append(row);
  }
  if (response.citations.length) pair.append(sources);
  const copy = iconButton("copy", "Copy answer", async () => {
    try { await navigator.clipboard.writeText(response.answer); notify("Answer copied"); }
    catch { notify("Copy is unavailable in this browser. Select the answer text to copy it."); }
  });
  copy.classList.add("copy-answer");
  pair.append(copy);
  return pair;
}
