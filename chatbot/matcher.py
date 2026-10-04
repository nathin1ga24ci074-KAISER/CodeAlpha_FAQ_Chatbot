"""
FAQ Matcher Module using TF-IDF Vectorization and Cosine Similarity
Compares incoming user questions with preprocessed FAQ entries and question variants.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_pipeline, clean_text, tokenize_and_normalize

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_ANSWER = (
    "I couldn't find a reliable answer for that in my knowledge base. "
    "Could you be more specific? You can ask about admissions, fees, scholarships, "
    "exams, attendance, library, hostel, internships, placements, or student ID."
)

AMBIGUOUS_FALLBACK_ANSWER = (
    "Your question seems a bit broad or ambiguous. Could you please provide more details "
    "or specify what you'd like to know? (For example, you can ask about admission requirements, "
    "fee payment methods, deadlines, scholarships, or exam schedules)."
)

GREETING_RESPONSE = (
    "Hello! I'm the University FAQ Assistant. How can I help you with admissions, "
    "fees, exams, scholarships, or other student services?"
)

GRATITUDE_RESPONSE = (
    "You're welcome! Feel free to ask another question."
)

# Deterministic intent keywords for specialized FAQs to prevent false-positive broad matches
FAQ_INTENT_REQUIREMENTS: Dict[int, Set[str]] = {
    # FAQ 13: Sports & Extracurricular Scholarship
    13: {
        "sport", "sports", "extracurricular", "athletic", "athletics",
        "cultural", "competition", "tournament", "athlete", "activity", "activities"
    },
    # FAQ 15: Re-evaluation or Rechecking of Exam Papers
    15: {
        "re-evaluation", "reevaluation", "evaluation", "rechecking", "recheck", "re-check",
        "paper", "papers", "copy", "copies", "re-eval", "reeval", "marks",
        "grade", "grades", "scorecard", "re-totaling", "retotaling", "re-assess", "reassess"
    },
}

# Generic domain head-words that are inherently ambiguous without action/qualifier intent
BROAD_GENERIC_KEYWORDS = {
    "fee", "fees", "course", "courses", "exam", "exams", "admission", "admissions",
    "hostel", "hostels", "scholarship", "scholarships", "placement", "placements"
}


class FAQMatcher:
    """
    NLP Question Matcher based on TF-IDF Vectorization and Cosine Similarity.
    Matches natural language user queries to a knowledge base of FAQs and question variants.
    """

    def __init__(
        self,
        faqs_path: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        ambiguity_margin: Optional[float] = None,
        high_confidence_threshold: Optional[float] = None,
    ):
        """
        Initialize the FAQMatcher with knowledge base, threshold, and ambiguity margin.

        Args:
            faqs_path: Path to faqs.json file. Defaults to data/faqs.json.
            similarity_threshold: Minimum cosine similarity score required for a match.
                                  Defaults to env var SIMILARITY_THRESHOLD or 0.25.
            ambiguity_margin: Minimum margin required between top and second candidate.
                              Defaults to env var AMBIGUITY_MARGIN or 0.08.
            high_confidence_threshold: Similarity score above which matches are considered definitive.
                                       Defaults to env var HIGH_CONFIDENCE_THRESHOLD or 0.60.
        """
        # Determine FAQ file path
        if faqs_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            faqs_path = os.path.join(base_dir, "data", "faqs.json")
        self.faqs_path = faqs_path

        # Determine threshold
        env_threshold = os.getenv("SIMILARITY_THRESHOLD")
        if similarity_threshold is not None:
            self.similarity_threshold = float(similarity_threshold)
        elif env_threshold is not None:
            self.similarity_threshold = float(env_threshold)
        else:
            self.similarity_threshold = 0.25

        # Determine ambiguity margin
        env_margin = os.getenv("AMBIGUITY_MARGIN")
        if ambiguity_margin is not None:
            self.ambiguity_margin = float(ambiguity_margin)
        elif env_margin is not None:
            self.ambiguity_margin = float(env_margin)
        else:
            self.ambiguity_margin = 0.08

        # Determine high confidence threshold
        env_high_conf = os.getenv("HIGH_CONFIDENCE_THRESHOLD")
        if high_confidence_threshold is not None:
            self.high_confidence_threshold = float(high_confidence_threshold)
        elif env_high_conf is not None:
            self.high_confidence_threshold = float(env_high_conf)
        else:
            self.high_confidence_threshold = 0.60

        # Internal storage
        self.faqs: List[Dict[str, Any]] = []
        self.corpus_entries: List[Tuple[int, str]] = []  # (faq_idx, text)
        self.faq_to_entries: Dict[int, List[int]] = {}   # faq_idx -> [corpus_entry_indices]
        self.preprocessed_corpus: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.faq_matrix = None

        # Load knowledge base and train vectorizer
        self._load_and_train()

    def _load_and_train(self) -> None:
        """Load FAQs from JSON, combine canonical and variant questions, and compute TF-IDF matrix."""
        if not os.path.exists(self.faqs_path):
            raise FileNotFoundError(f"FAQ knowledge base not found at: {self.faqs_path}")

        with open(self.faqs_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        if not self.faqs:
            raise ValueError("FAQ knowledge base is empty.")

        self.corpus_entries = []
        self.faq_to_entries = {}

        entry_idx = 0
        for faq_idx, item in enumerate(self.faqs):
            self.faq_to_entries[faq_idx] = []

            # Canonical question
            self.corpus_entries.append((faq_idx, item["question"]))
            self.faq_to_entries[faq_idx].append(entry_idx)
            entry_idx += 1

            # Variants
            for variant in item.get("variants", []):
                self.corpus_entries.append((faq_idx, variant))
                self.faq_to_entries[faq_idx].append(entry_idx)
                entry_idx += 1

        self.preprocessed_corpus = [
            preprocess_pipeline(text) for _, text in self.corpus_entries
        ]

        # Initialize TF-IDF Vectorizer with unigram and bigram features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        # Fit and compute the searchable TF-IDF matrix
        self.faq_matrix = self.vectorizer.fit_transform(self.preprocessed_corpus)

    def _check_conversational(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Detect and respond to greetings and gratitude before running FAQ vectorization.
        """
        cleaned = clean_text(user_query)
        if not cleaned:
            return None

        # Greeting check
        greeting_pattern = (
            r"^(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings|howdy)"
            r"(\s+(there|assistant|bot|team|all))?$"
        )
        if re.match(greeting_pattern, cleaned, re.IGNORECASE):
            return {
                "answer": GREETING_RESPONSE,
                "category": "Greeting",
                "confidence": 1.0,
                "confidence_pct": 100.0,
                "matched_question": "Greeting",
                "is_fallback": False,
            }

        # Gratitude check
        gratitude_pattern = (
            r"^(thank\s+you(\s+so\s+much|\s+very\s+much)?|thanks(\s+a\s+lot|\s+so\s+much|\s+again)?"
            r"|thx|many\s+thanks|much\s+appreciated|appreciate\s+it)$"
        )
        if re.match(gratitude_pattern, cleaned, re.IGNORECASE):
            return {
                "answer": GRATITUDE_RESPONSE,
                "category": "Gratitude",
                "confidence": 1.0,
                "confidence_pct": 100.0,
                "matched_question": "Gratitude",
                "is_fallback": False,
            }

        return None

    def match(self, user_query: str) -> Dict[str, Any]:
        """
        Find the best matching FAQ for a given natural-language user query.

        Args:
            user_query (str): The raw question asked by the user.

        Returns:
            dict containing:
                - answer (str)
                - category (str)
                - confidence (float: 0.0 to 1.0)
                - confidence_pct (float: 0.0 to 100.0)
                - matched_question (str or None)
                - is_fallback (bool)
        """
        if not user_query or not isinstance(user_query, str) or not user_query.strip():
            return {
                "answer": "Please ask a specific question so I can assist you.",
                "category": "Validation",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "matched_question": None,
                "is_fallback": True,
            }

        # Step 1: Check for basic conversational inputs (greetings / gratitude)
        conv_response = self._check_conversational(user_query)
        if conv_response is not None:
            return conv_response

        # Step 2: Preprocess user query
        processed_query = preprocess_pipeline(user_query)

        # If preprocessing left nothing (e.g. only punctuation or unknown symbols)
        if not processed_query.strip():
            return {
                "answer": DEFAULT_FALLBACK_ANSWER,
                "category": "Fallback",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "matched_question": None,
                "is_fallback": True,
            }

        # Step 3: Vectorize preprocessed query
        query_vector = self.vectorizer.transform([processed_query])

        # If query vector has all zeros (completely out of vocabulary)
        if query_vector.nnz == 0:
            return {
                "answer": DEFAULT_FALLBACK_ANSWER,
                "category": "Fallback",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "matched_question": None,
                "is_fallback": True,
            }

        # Step 4: Compute Cosine Similarity against all corpus questions and variants
        similarity_scores = cosine_similarity(query_vector, self.faq_matrix)[0]

        # Step 5: Aggregate scores per FAQ (maximum similarity among canonical question & variants)
        raw_lower = user_query.lower()
        query_tokens = set(tokenize_and_normalize(user_query))
        cleaned_words = set(clean_text(user_query).split())
        all_user_words = query_tokens.union(cleaned_words)

        faq_scores: Dict[int, float] = {}
        for faq_idx, entry_indices in self.faq_to_entries.items():
            max_score = float(max(similarity_scores[idx] for idx in entry_indices))

            # Apply intent keyword guards for specialized FAQs
            faq_id = self.faqs[faq_idx].get("id")
            if faq_id in FAQ_INTENT_REQUIREMENTS:
                required_keywords = FAQ_INTENT_REQUIREMENTS[faq_id]
                has_intent = bool(all_user_words & required_keywords) or any(
                    kw in raw_lower for kw in required_keywords
                )
                if not has_intent:
                    max_score = 0.0

            faq_scores[faq_idx] = max_score

        # Step 6: Rank FAQs by score
        sorted_faqs = sorted(faq_scores.items(), key=lambda item: item[1], reverse=True)
        top_idx, top_score = sorted_faqs[0]
        second_idx, second_score = sorted_faqs[1] if len(sorted_faqs) > 1 else (None, 0.0)

        # Step 7: Check against minimum similarity threshold
        if top_score < self.similarity_threshold:
            return {
                "answer": DEFAULT_FALLBACK_ANSWER,
                "category": "Fallback",
                "confidence": round(top_score, 4),
                "confidence_pct": round(top_score * 100, 1),
                "matched_question": None,
                "is_fallback": True,
            }

        # Step 8: Ambiguity & Broad-query Checks
        # Case A: Broad single-keyword query (e.g. "What about fees?")
        is_single_generic_keyword = (
            len(query_tokens) <= 1 and bool(all_user_words & BROAD_GENERIC_KEYWORDS)
        )
        if is_single_generic_keyword and top_score < self.high_confidence_threshold:
            return {
                "answer": AMBIGUOUS_FALLBACK_ANSWER,
                "category": "Fallback",
                "confidence": round(top_score, 4),
                "confidence_pct": round(top_score * 100, 1),
                "matched_question": None,
                "is_fallback": True,
            }

        # Case B: Top candidate vs second candidate ambiguity margin check
        if top_score < self.high_confidence_threshold and second_score > 0:
            margin = top_score - second_score
            if margin < self.ambiguity_margin:
                return {
                    "answer": AMBIGUOUS_FALLBACK_ANSWER,
                    "category": "Fallback",
                    "confidence": round(top_score, 4),
                    "confidence_pct": round(top_score * 100, 1),
                    "matched_question": None,
                    "is_fallback": True,
                }

        # Step 9: Confident Match - return canonical FAQ question
        matched_faq = self.faqs[top_idx]
        return {
            "answer": matched_faq["answer"],
            "category": matched_faq["category"],
            "confidence": round(top_score, 4),
            "confidence_pct": round(top_score * 100, 1),
            "matched_question": matched_faq["question"],
            "is_fallback": False,
        }

    def get_faq_count(self) -> int:
        """Return total number of loaded FAQs."""
        return len(self.faqs)

    def get_variant_count(self) -> int:
        """Return total number of question variants across all FAQs."""
        return len(self.corpus_entries) - len(self.faqs)

    def get_categories(self) -> List[str]:
        """Return unique sorted list of FAQ categories."""
        return sorted(list({faq["category"] for faq in self.faqs}))

    def get_sample_questions(self, count: int = 6) -> List[Dict[str, str]]:
        """Return a diverse sample of canonical questions from different categories."""
        seen_categories = set()
        samples = []
        for faq in self.faqs:
            if faq["category"] not in seen_categories:
                seen_categories.add(faq["category"])
                samples.append({
                    "category": faq["category"],
                    "question": faq["question"]
                })
            if len(samples) >= count:
                break
        return samples
