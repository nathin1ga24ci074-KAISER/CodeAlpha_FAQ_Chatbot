# CodeAlpha FAQ AI Assistant - Classical NLP Chatbot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask 3.x](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP%20Pipeline-yellow.svg)](https://www.nltk.org/)
[![Tests](https://img.shields.io/badge/Tests-28%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Demonstrable%20%26%20Complete-brightgreen.svg)]()

> **CodeAlpha Artificial Intelligence Internship — Task 2: FAQ Chatbot using NLP**
> An intelligent, self-contained FAQ Chatbot built using Python, Flask, NLTK, and Scikit-learn. It preprocesses natural language questions and matches them against a local demonstration University Student Services FAQ knowledge base using **TF-IDF Vectorization** and **Cosine Similarity**, running 100% locally with zero external API dependencies.

---

## Demonstration Knowledge Base Notice

> [!NOTE]
> **Academic / Demonstration Purpose**: The FAQ entries in this repository represent a **generalized, fictional demonstration dataset** designed specifically to evaluate NLP text processing, tokenization, lemmatization, and vector similarity algorithms. The policies, fees, dates, and requirements are illustrative and do not represent any real-world university or institution.

> [!IMPORTANT]
> **Classical NLP Implementation**: This project strictly uses **classical statistical Natural Language Processing (NLP)**: text cleaning, NLTK tokenization, stopword removal, WordNet lemmatization, Scikit-learn TF-IDF vectorization, and cosine similarity distance. **It does NOT use external Large Language Models (LLMs), Generative AI, OpenAI, Gemini, or third-party cloud APIs.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [NLP & Matching Pipeline](#nlp--matching-pipeline)
  - [1. Text Preprocessing](#1-text-preprocessing)
  - [2. Sublinear TF-IDF Vectorization](#2-sublinear-tf-idf-vectorization)
  - [3. Cosine Similarity Calculation](#3-cosine-similarity-calculation)
  - [4. FAQ Question Variants](#4-faq-question-variants)
  - [5. Ambiguity Handling & Candidate Margin](#5-ambiguity-handling--candidate-margin)
  - [6. Conversational Intent Handling](#6-conversational-intent-handling)
  - [7. Fallback & Low-Confidence Protection](#7-fallback--low-confidence-protection)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Running the Test Suite](#running-the-test-suite)
- [API Documentation](#api-documentation)
- [Example Questions & Behavior](#example-questions--behavior)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Internship Attribution](#internship-attribution)

---

## Project Overview

In university campus environments and customer support desks, staff members routinely answer hundreds of repetitive queries regarding admission deadlines, tuition fee payment methods, examination schedules, scholarship criteria, and hostel regulations.

This project provides an automated, lightweight, self-hosted AI FAQ assistant capable of understanding varied natural-language phrasings, evaluating semantic proximity against indexed FAQ items, and delivering precise answers with transparent confidence metrics.

---

## Problem Statement

Traditional keyword-based search algorithms fail when users paraphrase questions, use synonyms, or ask broad, ambiguous questions:
1. **Vocabulary Mismatch**: A user asking *"How do I enroll in college?"* should match the canonical FAQ *"How can I apply for admission to the university?"* despite differing keywords.
2. **False Positives from Weak Overlap**: Simple term matching might return a specific *Sports Scholarship* answer for a generic query like *"How can I get a scholarship?"*, or return an *Exam Re-evaluation* answer for *"When are exams conducted?"*.
3. **Ambiguity in Broad Queries**: Queries like *"What about fees?"* lack specific intent and should solicit clarification rather than guessing a specific payment portal answer.
4. **Out-of-Domain Hallucinations**: Irrelevant queries (e.g., recipes, jokes, astronomy) should trigger an informative fallback without fabricating answers.

The **CodeAlpha FAQ AI Assistant** solves these challenges using an end-to-end deterministic NLP pipeline with question variants, multi-candidate margin inspection, and keyword-guided disambiguation.

---

## Key Features

- **100% Local & Offline**: Runs entirely on your local machine with zero external cloud dependencies or API keys.
- **Comprehensive Knowledge Base**: 36 canonical FAQ topics spanning 15 campus service categories with **193 natural language question variants**.
- **Robust NLP Preprocessing**: Custom pipeline utilizing NLTK (lowercasing, punctuation stripping, tokenization, stopword filtering, dual-phase verb/noun lemmatization).
- **Sublinear TF-IDF Vectorizer**: Captures unigrams $(1, 1)$ and bigrams $(1, 2)$ to emphasize key domain compounds (e.g., `"tuition fee"`, `"id card"`, `"exam schedule"`).
- **Normalized Cosine Proximity**: Accurate vector angle measurement invariant to text length.
- **Ambiguity & Disambiguation Engine**:
  - Compares top two scoring candidates with a configurable margin ($\Delta = 0.08$).
  - Prevents over-specific matches when user queries are broad.
- **Conversational Handlers**: Recognizes and responds politely to greetings (*"Hi"*, *"Good morning"*) and gratitude (*"Thank you"*, *"Thanks"*).
- **Modern Academic UI**:
  - Accessible, responsive design with clear visual hierarchy.
  - Technically accurate `Similarity: <pct>%` badges.
  - Interactive "How It Works" visual pipeline flow.
  - Quick-start suggested question chips with keyboard support (`Enter` / `Space`).
  - Conversation reset ("Reset Chat") button.
- **Standardized REST API**: JSON endpoints for chat, health monitoring, categories, and sample questions.
- **Rigorous Automated Testing**: Complete test suite with **28 unit tests** covering data integrity, NLP stages, ambiguity checks, and HTTP routes.

---

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | Python 3.10+ / Flask 3.x | Lightweight WSGI web framework and REST API server |
| **NLP Preprocessing** | NLTK (`nltk`) | Tokenization, English stopword removal, WordNet lemmatizer |
| **Feature Extraction** | Scikit-learn (`scikit-learn`) | `TfidfVectorizer` with sublinear TF scaling and $(1, 2)$ n-grams |
| **Similarity Metric** | Scikit-learn (`metrics.pairwise`) | `cosine_similarity` for normalized vector distance calculation |
| **Numerical Processing**| NumPy (`numpy`) | Array indexing and multi-candidate score comparisons |
| **Data Storage** | JSON (`data/faqs.json`) | Portable, decoupled FAQ knowledge base with question variants |
| **Frontend UI** | Semantic HTML5, CSS3, Vanilla JS | Zero-dependency, responsive interface with SVG icons |
| **Test Framework** | Python `unittest` | Automated unit, regression, and integration testing |

---

## System Architecture

```text
                               +-----------------------------+
                               |        User Browser         |
                               | (HTML5 / CSS3 / Vanilla JS) |
                               +-----------------------------+
                                              |
                          JSON Request: POST /api/chat {"message": "..."}
                                              v
                               +-----------------------------+
                               |      Flask Application      |
                               |          (app.py)           |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |  NLP Preprocessing Pipeline |
                               |    (preprocessing.py)       |
                               | - Lowercase & Clean Text    |
                               | - Tokenization (NLTK)       |
                               | - Stopwords Filtering       |
                               | - WordNet Lemmatization     |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |     TF-IDF Vectorizer       |
                               |        (matcher.py)         |
                               | - 1-gram & 2-gram Features  |
                               | - Pre-fitted on 193 Variants|
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |  Cosine Similarity Engine   |
                               | - Inner Product with Matrix |
                               | - Sort Candidate Scores     |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | Ambiguity & Threshold Check |
                               | - Threshold Check (>= 0.25) |
                               | - Margin Check (s1 - s2)    |
                               | - Domain Keyword Guardrails |
                               +-----------------------------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
                 [Match Found]                               [No Match / Ambiguous]
                       v                                             v
        +-----------------------------+               +-----------------------------+
        |  Return Canonical Answer    |               |   Return Clarification or   |
        |  Category & Similarity %    |               |   Domain Fallback Message   |
        +-----------------------------+               +-----------------------------+
```

---

## NLP & Matching Pipeline

### 1. Text Preprocessing
Raw input strings undergo a 4-stage sequential transformation in [`chatbot/preprocessing.py`](file:///C:/Users/nathi/Desktop/CodeAlpha_FAQ_Chatbot/chatbot/preprocessing.py):
1. **Text Cleaning**: Replaces punctuation and special characters with spaces, converts characters to lowercase, and strips extra whitespace.
2. **Tokenization**: Uses `nltk.tokenize.word_tokenize` to isolate individual tokens.
3. **Stopword Removal**: Eliminates uninformative English function words (e.g., *"the"*, *"is"*, *"in"*, *"at"*, *"of"*, *"for"*).
4. **Dual-Phase Lemmatization**: Utilizes `nltk.stem.WordNetLemmatizer` to reduce tokens to dictionary root forms, first attempting verb lemmatization (`pos='v'`) followed by noun lemmatization (`pos='n'`).
   *Examples: `"applying"` $\rightarrow$ `"apply"`, `"fees"` $\rightarrow$ `"fee"`, `"registered"` $\rightarrow$ `"register"`, `"scholarships"` $\rightarrow$ `"scholarship"`.*

### 2. Sublinear TF-IDF Vectorization
The processed tokens are converted into numerical feature vectors using `TfidfVectorizer`:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Sublinear Term Frequency**: Replaces raw count $\text{TF}$ with $1 + \ln(\text{TF})$ to prevent frequent words from dominating similarity scores.
- **Inverse Document Frequency**: Penalizes terms that appear ubiquitously across many FAQ entries:
  $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
- **N-gram Modeling**: Analyzes both unigrams and bigrams $(1, 2)$ so that contextual phrases (such as `"pay fee"`, `"hostel fee"`, or `"id card"`) carry distinct feature weights.

### 3. Cosine Similarity Calculation
The query vector $\vec{u}$ is compared against all indexed FAQ entry vectors $\vec{v}_i$:
$$\text{Similarity}(\vec{u}, \vec{v}_i) = \cos(\theta) = \frac{\vec{u} \cdot \vec{v}_i}{\|\vec{u}\| \|\vec{v}_i\|} = \frac{\sum_{k} u_k v_{ik}}{\sqrt{\sum_{k} u_k^2} \sqrt{\sum_{k} v_{ik}^2}}$$

The metric outputs a normalized scalar value between $0.0$ (no overlap) and $1.0$ (identical vector orientation). Because cosine similarity measures the angle rather than vector magnitude, it remains invariant to document length.

### 4. FAQ Question Variants
To bridge vocabulary variations without retraining, each FAQ entry in [`data/faqs.json`](file:///C:/Users/nathi/Desktop/CodeAlpha_FAQ_Chatbot/data/faqs.json) contains multiple natural-language question variants:
```json
{
  "id": 1,
  "question": "How can I apply for admission to the university?",
  "variants": [
    "How do I get admission?",
    "How can I apply for admission?",
    "What is the admission process?",
    "How do I enroll in the university?",
    "Where can I submit my admission application?"
  ],
  "answer": "You can apply online through the university admissions portal...",
  "category": "Admissions"
}
```
During initialization, the vectorizer builds its vocabulary across all **193 variants**. When any variant matches, the system retrieves the canonical FAQ answer and displays the canonical question in the UI.

### 5. Ambiguity Handling & Candidate Margin
To prevent false-positive answers on broad queries, [`chatbot/matcher.py`](file:///C:/Users/nathi/Desktop/CodeAlpha_FAQ_Chatbot/chatbot/matcher.py) inspects the top two ranked candidates:
- If the difference between the top score $s_1$ and second score $s_2$ is less than the ambiguity margin ($\Delta = s_1 - s_2 < 0.08$), the query is recognized as potentially ambiguous.
- **Domain Guardrails**:
  - *"How can I get a scholarship?"* matches the general **Merit Scholarship** FAQ rather than the specific Sports Scholarship unless athletic/sports keywords are explicitly mentioned.
  - *"What about fees?"* is recognized as broad and triggers a polite clarification prompt rather than assuming an online payment intent.
  - *"When are exams conducted?"* matches the **Exam Schedule** FAQ and refuses to match Re-evaluation unless rechecking terms are present.

### 6. Conversational Intent Handling
Before vector search, the engine checks for conversational pleasantries using regex patterns:
- **Greetings**: `"hello"`, `"hi"`, `"hey"`, `"good morning"`, `"good afternoon"`, `"good evening"` $\rightarrow$ Returns a warm welcome introducing available student service topics.
- **Gratitude**: `"thank you"`, `"thanks"`, `"thank you so much"` $\rightarrow$ Returns a polite acknowledgement.

### 7. Fallback & Low-Confidence Protection
If the best cosine similarity score falls below the configured threshold (default `0.25`), the chatbot returns a structured fallback response:
```json
{
  "answer": "I'm sorry, I couldn't find a reliable answer to that question in my knowledge base. Try asking about admissions, courses, fees, scholarships, exams, attendance, library, hostels, internships, or campus placements.",
  "category": "Fallback",
  "confidence": 0.0,
  "confidence_pct": 0.0,
  "is_fallback": true,
  "matched_question": null
}
```

---

## Project Structure

```text
CodeAlpha_FAQ_Chatbot/
│
├── app.py                      # Flask web application & REST API routes
├── requirements.txt            # Minimal runtime dependencies
├── README.md                   # Comprehensive technical documentation
├── .gitignore                  # Git rules excluding venv, cache, OS files, logs
│
├── data/
│   └── faqs.json               # 36 canonical FAQs with 193 question variants
│
├── chatbot/
│   ├── __init__.py             # Chatbot package initialization
│   ├── preprocessing.py        # Text cleaning, NLTK tokenization, stopwords, lemmatization
│   └── matcher.py              # TF-IDF vectorizer, cosine similarity, ambiguity engine
│
├── templates/
│   └── index.html              # Modern, accessible semantic HTML5 chat interface
│
├── static/
│   ├── css/
│   │   └── style.css           # Responsive styles, typography, badges, pipeline diagram
│   └── js/
│       └── script.js           # Interactive UI, AJAX communication, suggestion chips
│
└── test_chatbot.py             # Complete automated test suite (28 unit tests)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- PowerShell, Command Prompt, or bash terminal.

### 1. Clone or Navigate to Project
```powershell
cd C:\Users\nathi\Desktop\CodeAlpha_FAQ_Chatbot
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(On macOS/Linux: `source venv/bin/activate`)*

### 3. Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Verify NLTK Data Corpora
NLTK packages (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) download automatically on first run. To pre-download them manually:
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Running the Application

Start the local Flask development server:

```powershell
python app.py
```

Expected terminal output:
```text
2026-10-04 08:00:00 [INFO] FAQChatbotApp: FAQMatcher initialized successfully with 36 FAQs.
Starting CodeAlpha FAQ Chatbot on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## Running the Test Suite

Run the full automated test suite using Python's built-in `unittest` runner:

```powershell
python -m unittest discover
```
or:
```powershell
python test_chatbot.py
```

### Expected Output
```text
............................
----------------------------------------------------------------------
Ran 28 tests in 0.117s

OK
```

### Test Suite Coverage
The 28 tests validate:
- **Data Integrity**: JSON schema, non-empty fields, valid IDs, category diversity, and variant presence.
- **NLP Preprocessing**: Lowercasing, punctuation stripping, tokenization, stopword removal, and verb/noun lemmatization.
- **Exact & Variant Matching**: Exact canonical questions, paraphrased variants, and synonym variations.
- **Intent Disambiguation**: Merit vs sports scholarships, exam scheduling vs re-evaluation, broad fee queries.
- **Conversational Handling**: Multiple greetings and gratitude expressions.
- **Fallback Triggering**: Out-of-domain prompts, random characters, empty input, and whitespace.
- **HTTP Endpoints**: GET `/api/health`, GET `/api/sample-questions`, GET `/api/categories`, and POST `/api/chat`.

---

## API Documentation

### 1. `GET /api/health`
Checks application health and reports indexed knowledge base statistics.

**Response `(200 OK)`**:
```json
{
  "ambiguity_margin": 0.08,
  "categories_count": 15,
  "faq_count": 36,
  "status": "healthy",
  "threshold": 0.25,
  "variants_count": 193
}
```

---

### 2. `POST /api/chat`
Submits a natural language query for NLP preprocessing and similarity matching.

**Request Header**:
`Content-Type: application/json`

**Request Body**:
```json
{
  "message": "How do I get admission?"
}
```

**Success Response `(200 OK — Confident Match)`**:
```json
{
  "answer": "You can apply online through the university admissions portal. Complete the registration form, upload the required academic transcripts and identity documents, and pay the non-refundable application fee.",
  "category": "Admissions",
  "confidence": 1.0,
  "confidence_pct": 100.0,
  "is_fallback": false,
  "matched_question": "How can I apply for admission to the university?"
}
```

**Clarification Response `(200 OK — Ambiguous Query)`**:
```json
{
  "answer": "Your question about fees is broad or ambiguous. Could you be more specific? You can ask about online tuition fee payment methods, refund policies, exam fees, or hostel fees.",
  "category": "Fallback",
  "confidence": 0.308,
  "confidence_pct": 30.8,
  "is_fallback": true,
  "matched_question": null
}
```

**Fallback Response `(200 OK — Out of Domain)`**:
```json
{
  "answer": "I'm sorry, I couldn't find a reliable answer to that question in my knowledge base. Try asking about admissions, courses, fees, scholarships, exams, attendance, library, hostels, internships, or campus placements.",
  "category": "Fallback",
  "confidence": 0.0,
  "confidence_pct": 0.0,
  "is_fallback": true,
  "matched_question": null
}
```

**Error Response `(400 Bad Request)`**:
```json
{
  "error": "The 'message' field cannot be empty or only whitespace."
}
```

---

### 3. `GET /api/sample-questions`
Returns curated sample questions across distinct categories for quick-start suggestion chips.

**Response `(200 OK)`**:
```json
[
  { "category": "Admissions", "question": "How can I apply for admission to the university?" },
  { "category": "Fees", "question": "How can I pay my tuition fees online?" },
  { "category": "Exams", "question": "When will the semester examination schedule be announced?" }
]
```

---

### 4. `GET /api/categories`
Returns all 15 indexed knowledge base topic categories.

**Response `(200 OK)`**:
```json
{
  "categories": [
    "Academic Support", "Admissions", "Attendance", "Certificates", "Contacts",
    "Courses", "Exams", "Fees", "Hostel", "Internships",
    "Library", "Placements", "Scholarships", "Student ID", "Timetables"
  ]
}
```

---

## Example Questions & Behavior

| Query | Category | Similarity | System Behavior |
| :--- | :--- | :--- | :--- |
| `"How can I apply for admission?"` | Admissions | 100.0% | Matches canonical admission FAQ via indexed variant |
| `"Can I pay fees online?"` | Fees | 100.0% | Matches online fee payment FAQ |
| `"How can I get a scholarship?"` | Scholarships | 100.0% | Matches general merit scholarship (not sports scholarship) |
| `"Are there sports scholarships?"` | Scholarships | 100.0% | Matches sports/extracurricular scholarship via intent check |
| `"When are exams conducted?"` | Exams | 100.0% | Matches exam schedule FAQ (not re-evaluation FAQ) |
| `"What about fees?"` | Fallback | 30.8% | Triggers clarification request due to ambiguity |
| `"I lost my student ID."` | Student ID | 100.0% | Matches replacement student ID card procedure |
| `"Hi there!"` | Greeting | 100.0% | Returns conversational welcome message |
| `"Thank you very much"` | Gratitude | 100.0% | Returns polite acknowledgement |
| `"What is quantum string theory?"`| Fallback | 0.0% | Refuses to fabricate; offers campus topic suggestions |

---

## Limitations

1. **Lexical/Bag-of-Words Representation**: TF-IDF relies on n-gram word overlaps and morphological stems. It cannot infer complex relational reasoning or resolve completely unfamiliar synonyms not present in the vocabulary or WordNet.
2. **Stateless Single-Turn Interaction**: The engine evaluates each query independently and does not track contextual state across multiple turns (e.g., following up with *"And what is the deadline for that?"*).
3. **Bounded Knowledge Scope**: The assistant is strictly limited to the topics defined in [`data/faqs.json`](file:///C:/Users/nathi/Desktop/CodeAlpha_FAQ_Chatbot/data/faqs.json).

---

## Future Enhancements

1. **Dense Vector Embeddings**: Integrate local transformer models (such as `sentence-transformers/all-MiniLM-L6-v2`) in a hybrid lexical-semantic pipeline.
2. **Contextual Session State**: Implement session tracking to resolve pronouns and follow-up inquiries.
3. **Admin Feedback & Curation**: Provide an administrative view to log unanswered queries and expand question variants continuously.
4. **Speech Synthesis & Recognition**: Utilize browser Web Speech APIs to support voice input and spoken audio responses.

---

## Internship Attribution

- **Organization**: [CodeAlpha](https://www.codealpha.tech/)
- **Internship**: Artificial Intelligence Internship
- **Task**: Task 2 — FAQ Chatbot using NLP
- **Intern / Author**: Nathin Gowda K G
- **Date**: October 2026
- **License**: Open-source educational project for portfolio and demonstration purposes.
