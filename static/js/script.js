/**
 * CodeAlpha FAQ Chatbot - Frontend Controller
 * Handles user interactions, API requests, message rendering, accessibility, and UI states.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const chipsList = document.getElementById("chipsList");
  const statusPill = document.getElementById("statusPill");
  const faqCountPill = document.getElementById("faqCountPill");

  // State
  let isWaitingForResponse = false;

  // Curated initial sample questions representing major domains
  const defaultSuggestions = [
    "How do I get admission?",
    "Can I pay fees online?",
    "When are exams conducted?",
    "How can I get a scholarship?",
    "Are there sports scholarships?",
    "What is the minimum attendance required?",
    "What are the library opening hours?",
    "What amenities are in the hostel fee?",
    "When do campus placements start?",
    "I lost my student ID."
  ];

  /**
   * Helper to format current time (e.g. "10:35 AM")
   */
  function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  /**
   * Smoothly scrolls chat history to the bottom
   */
  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  /**
   * Escapes HTML to protect against XSS
   */
  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Renders the welcome card with introductory guidance, category chips, and pipeline diagram
   */
  function renderWelcomeCard() {
    const welcomeHtml = `
      <div class="welcome-card" id="welcomeCard">
        <div class="welcome-header">
          <div class="welcome-icon-box" aria-hidden="true">
            <svg class="icon-welcome" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 15h-2v-6h2zm0-8h-2V7h2z"></path>
            </svg>
          </div>
          <div class="welcome-header-text">
            <h2 class="welcome-title">Welcome to the University FAQ Assistant</h2>
            <p class="welcome-subtitle">
              Ask questions about admissions, fees, exams, scholarships, attendance, library services, hostels, internships, and placements.
            </p>
          </div>
        </div>

        <div class="welcome-attribution-badge">
          <svg class="icon-tiny" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          <span>Powered by a local FAQ knowledge base using NLP, TF-IDF, and cosine similarity.</span>
        </div>

        <div class="welcome-categories-section">
          <span class="categories-heading">Supported Campus Topics:</span>
          <div class="category-tags-grid">
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Admissions &amp; Criteria</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Tuition Fees &amp; Refunds</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Merit &amp; Sports Scholarships</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Exams &amp; Re-evaluations</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Attendance Regulations</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Hostels &amp; Curfew Rules</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Library &amp; Digital Papers</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Campus Placements</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Internships &amp; NOC</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Student Smart ID</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Certificates &amp; Transcripts</span>
            <span class="category-tag"><span class="tag-bullet" aria-hidden="true"></span>Academic Advising &amp; Tutoring</span>
          </div>
        </div>

        <div class="how-it-works-box">
          <div class="how-it-works-header">
            <div class="how-it-works-title">
              <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
              <span>How It Works</span>
            </div>
            <span class="pipeline-badge">NLP Pipeline</span>
          </div>

          <div class="pipeline-flow" role="list" aria-label="Chatbot NLP processing flow">
            <div class="flow-step" role="listitem">User Question</div>
            <div class="flow-arrow" aria-hidden="true">→</div>
            <div class="flow-step" role="listitem">NLP Preprocessing</div>
            <div class="flow-arrow" aria-hidden="true">→</div>
            <div class="flow-step" role="listitem">TF-IDF</div>
            <div class="flow-arrow" aria-hidden="true">→</div>
            <div class="flow-step" role="listitem">Cosine Similarity</div>
            <div class="flow-arrow" aria-hidden="true">→</div>
            <div class="flow-step" role="listitem">Best FAQ</div>
            <div class="flow-arrow" aria-hidden="true">→</div>
            <div class="flow-step" role="listitem">Answer</div>
          </div>

          <p class="pipeline-desc">
            Your question is processed and compared with the local FAQ knowledge base to find the most relevant answer.
          </p>
        </div>
      </div>
    `;
    chatMessages.innerHTML = welcomeHtml;
    chatMessages.scrollTop = 0;
  }

  /**
   * Loads sample questions into suggestion chips
   */
  async function loadSuggestions() {
    try {
      const response = await fetch("/api/sample-questions");
      if (response.ok) {
        const samples = await response.json();
        if (Array.isArray(samples) && samples.length > 0) {
          const remoteQuestions = samples.map(s => s.question);
          // Combine remote with curated common queries for variety
          const combined = Array.from(new Set([...defaultSuggestions.slice(0, 4), ...remoteQuestions]));
          renderChips(combined);
          return;
        }
      }
    } catch (err) {
      console.warn("Could not fetch remote suggestions, using default:", err);
    }
    renderChips(defaultSuggestions);
  }

  /**
   * Check API health and update metrics banner
   */
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        if (faqCountPill) {
          faqCountPill.textContent = `${data.faq_count || 36} FAQs`;
        }
        if (statusPill) {
          statusPill.innerHTML = '<span class="status-indicator-dot" aria-hidden="true"></span>Online';
          statusPill.title = `System Online • ${data.faq_count || 36} FAQs indexed across ${data.categories_count || 15} categories`;
        }
      }
    } catch (e) {
      if (statusPill) {
        statusPill.innerHTML = '<span class="status-indicator-dot offline" aria-hidden="true"></span>Offline';
      }
    }
  }

  /**
   * Renders suggestion chip buttons with keyboard accessibility
   */
  function renderChips(questions) {
    chipsList.innerHTML = "";
    questions.forEach(q => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip-btn";
      btn.textContent = q;
      btn.setAttribute("tabindex", "0");
      btn.setAttribute("aria-label", `Ask: ${q}`);

      const handleClick = () => {
        if (!isWaitingForResponse) {
          userInput.value = q;
          sendMessage(q);
        }
      };

      btn.addEventListener("click", handleClick);
      btn.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      });

      chipsList.appendChild(btn);
    });
  }

  /**
   * Appends a user message bubble
   */
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="avatar avatar-user" title="You" aria-label="You">You</div>
      <div class="bubble">
        <div class="message-text">${escapeHtml(text)}</div>
        <div class="message-meta user-meta">
          <span class="meta-time">${getCurrentTime()}</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Appends an active typing indicator
   */
  function showTypingIndicator() {
    const typingRow = document.createElement("div");
    typingRow.className = "message-row bot";
    typingRow.id = "typingIndicator";
    typingRow.setAttribute("aria-label", "Assistant is searching knowledge base");
    typingRow.innerHTML = `
      <div class="avatar avatar-bot" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect width="16" height="12" x="4" y="8" rx="2"></rect>
          <path d="M12 8V4H8"></path>
        </svg>
      </div>
      <div class="bubble typing-bubble">
        <span class="typing-dot" aria-hidden="true"></span>
        <span class="typing-dot" aria-hidden="true"></span>
        <span class="typing-dot" aria-hidden="true"></span>
        <span class="sr-only">Searching FAQ knowledge base...</span>
      </div>
    `;
    chatMessages.appendChild(typingRow);
    scrollToBottom();
  }

  /**
   * Removes typing indicator
   */
  function removeTypingIndicator() {
    const typing = document.getElementById("typingIndicator");
    if (typing) {
      typing.remove();
    }
  }

  /**
   * Formats answer text for readable fallback guidance when bullet suggestions are present
   */
  function formatBotAnswer(answerText, isFallback) {
    const escaped = escapeHtml(answerText);
    if (!isFallback) return escaped;

    // Check if the fallback contains a topic suggestions sentence
    const tryAskingMarker = "Try asking about:";
    const couldYouMarker = "Could you be more specific? You can ask about";

    if (escaped.includes(tryAskingMarker)) {
      const parts = escaped.split(tryAskingMarker);
      return `
        <div class="fallback-lead">${parts[0].trim()}</div>
        <div class="fallback-topics-box">
          <span class="fallback-topics-label">Try asking about:</span>
          <span class="fallback-topics-list">${parts[1].trim()}</span>
        </div>
      `;
    }

    if (escaped.includes(couldYouMarker)) {
      const parts = escaped.split("Could you be more specific?");
      return `
        <div class="fallback-lead">${parts[0].trim()}</div>
        <div class="fallback-topics-box">
          <span class="fallback-topics-label">Could you be more specific?</span>
          <span class="fallback-topics-list">${(parts[1] || "").replace("You can ask about", "Suggested domains:").trim()}</span>
        </div>
      `;
    }

    return escaped;
  }

  /**
   * Appends bot response bubble with category badge & cosine similarity rating
   */
  function appendBotMessage(data) {
    const row = document.createElement("div");
    row.className = "message-row bot";

    const isFallback = Boolean(data.is_fallback);
    const category = data.category || "General";
    const answer = data.answer || "";

    // Check if clarification is needed (ambiguity fallback) vs standard out-of-domain fallback
    const isAmbiguous = isFallback && answer.toLowerCase().includes("broad or ambiguous");

    const similarityPct = typeof data.confidence_pct === "number"
      ? data.confidence_pct
      : Math.round((data.confidence || 0) * 100);

    let confClass = "conf-med";
    let confLabel = `Similarity: ${similarityPct}%`;
    let categoryClass = "cat-standard";

    if (category === "Greeting" || category === "Gratitude") {
      confClass = "conf-conv";
      confLabel = "Conversational";
      categoryClass = "cat-conv";
    } else if (isAmbiguous) {
      confClass = "conf-ambiguous";
      confLabel = "Clarification Needed";
      categoryClass = "cat-fallback";
    } else if (isFallback) {
      confClass = "conf-fallback";
      confLabel = "Low Confidence";
      categoryClass = "cat-fallback";
    } else {
      if (similarityPct >= 50) {
        confClass = "conf-high";
      } else {
        confClass = "conf-med";
      }
    }

    // Matched FAQ canonical note
    const matchedNote = (data.matched_question && !isFallback && category !== "Greeting" && category !== "Gratitude")
      ? `
        <div class="meta-matched-faq">
          <span class="matched-faq-label">Matched FAQ:</span>
          <span class="matched-faq-text">"${escapeHtml(data.matched_question)}"</span>
        </div>
      `
      : "";

    const formattedAnswer = formatBotAnswer(answer, isFallback);

    row.innerHTML = `
      <div class="avatar avatar-bot" title="FAQ Assistant" aria-label="FAQ Assistant">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect width="16" height="12" x="4" y="8" rx="2"></rect>
          <path d="M12 8V4H8"></path>
        </svg>
      </div>
      <div class="bubble bot-bubble ${isFallback ? 'bubble-fallback' : ''}">
        <div class="message-text">${formattedAnswer}</div>
        <div class="message-meta">
          <div class="meta-badges">
            <span class="meta-badge meta-category ${categoryClass}">${escapeHtml(category)}</span>
            <span class="meta-badge meta-confidence ${confClass}">${confLabel}</span>
          </div>
          <span class="meta-time">${getCurrentTime()}</span>
        </div>
        ${matchedNote}
      </div>
    `;

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Appends network or server error notice
   */
  function appendErrorMessage(message) {
    const row = document.createElement("div");
    row.className = "message-row bot";
    row.innerHTML = `
      <div class="avatar avatar-bot avatar-error" aria-label="Error">!</div>
      <div class="bubble network-error-bubble">
        <div class="message-text">${escapeHtml(message)}</div>
        <div class="message-meta">
          <div class="meta-badges">
            <span class="meta-badge meta-category cat-fallback">System Notice</span>
          </div>
          <span class="meta-time">${getCurrentTime()}</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  /**
   * Sends the user query to the /api/chat endpoint
   */
  async function sendMessage(messageText) {
    const query = (messageText || userInput.value || "").trim();
    if (!query) {
      userInput.focus();
      return;
    }

    if (isWaitingForResponse) return;

    // Display user bubble
    appendUserMessage(query);
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;
    isWaitingForResponse = true;

    // Show animated typing indicator
    showTypingIndicator();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: query })
      });

      const data = await response.json();
      removeTypingIndicator();

      if (!response.ok) {
        appendErrorMessage(data.error || "Unable to retrieve an answer at this time.");
      } else {
        appendBotMessage(data);
      }
    } catch (error) {
      removeTypingIndicator();
      console.error("Fetch error:", error);
      appendErrorMessage("Cannot connect to server. Please check your network connection or server status.");
    } finally {
      userInput.disabled = false;
      sendBtn.disabled = false;
      isWaitingForResponse = false;
      userInput.focus();
    }
  }

  // Form submission handler
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });

  // Enter key support in input
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Clear chat button
  clearChatBtn.addEventListener("click", () => {
    if (confirm("Reset chat conversation history?")) {
      renderWelcomeCard();
      userInput.value = "";
      userInput.focus();
    }
  });

  // Initialize application
  renderWelcomeCard();
  loadSuggestions();
  checkHealth();
  userInput.focus();
});
