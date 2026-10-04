"""
CodeAlpha FAQ Chatbot - Flask Web Application
Provides REST API endpoints and web interface for NLP-based FAQ matching.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify

from chatbot.matcher import FAQMatcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FAQChatbotApp")

# Initialize Flask app
app = Flask(__name__)

# Load FAQ Matcher instance
try:
    matcher = FAQMatcher()
    logger.info("FAQMatcher initialized successfully with %d FAQs.", matcher.get_faq_count())
except Exception as e:
    logger.error("Failed to initialize FAQMatcher: %s", e)
    matcher = None


@app.route("/")
def index():
    """Render the main chatbot web interface."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint returning system status and FAQ knowledge base metrics.
    """
    if matcher is None:
        return jsonify({
            "status": "unhealthy",
            "error": "Matcher failed to initialize"
        }), 503

    return jsonify({
        "status": "healthy",
        "faq_count": matcher.get_faq_count(),
        "variants_count": matcher.get_variant_count(),
        "threshold": matcher.similarity_threshold,
        "ambiguity_margin": matcher.ambiguity_margin,
        "categories_count": len(matcher.get_categories())
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Primary chat endpoint for natural-language question matching.

    Request JSON:
        { "message": "How do I apply for admission?" }

    Response JSON:
        {
            "answer": "...",
            "category": "Admissions",
            "confidence": 0.87,
            "matched_question": "...",
            "is_fallback": false
        }
    """
    if matcher is None:
        return jsonify({
            "error": "Chatbot engine is currently unavailable."
        }), 503

    # Ensure JSON payload
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid request payload. Expected a JSON object with a 'message' field."
        }), 400

    # Extract and validate message field
    user_message = data.get("message")
    if user_message is None or not isinstance(user_message, str):
        return jsonify({
            "error": "The 'message' field is required and must be a string."
        }), 400

    user_message = user_message.strip()
    if not user_message:
        return jsonify({
            "error": "The 'message' field cannot be empty or only whitespace."
        }), 400

    if len(user_message) > 1000:
        return jsonify({
            "error": "The question is too long. Please limit your query to 1000 characters."
        }), 400

    try:
        match_result = matcher.match(user_message)
        logger.info(
            "Query: '%s' | Matched: '%s' | Confidence: %.4f | Category: '%s'",
            user_message,
            match_result.get("matched_question"),
            match_result.get("confidence", 0.0),
            match_result.get("category")
        )
        return jsonify(match_result), 200

    except Exception as exc:
        logger.exception("Error processing chat message: %s", exc)
        return jsonify({
            "error": "An internal server error occurred while processing your request."
        }), 500


@app.route("/api/sample-questions", methods=["GET"])
def sample_questions():
    """
    Returns curated sample questions to populate starter suggestion chips in the UI.
    """
    if matcher is None:
        return jsonify([]), 503

    samples = matcher.get_sample_questions(count=8)
    return jsonify(samples), 200


@app.route("/api/categories", methods=["GET"])
def categories():
    """Returns all FAQ topic categories."""
    if matcher is None:
        return jsonify([]), 503

    return jsonify({
        "categories": matcher.get_categories()
    }), 200


@app.errorhandler(404)
def not_found(e):
    """Handle 404 routes gracefully."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "API route not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    """Handle unexpected server errors gracefully."""
    return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    print(f"Starting CodeAlpha FAQ Chatbot on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
