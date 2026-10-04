"""
Comprehensive Test Suite for CodeAlpha FAQ Chatbot
Tests FAQ data integrity, NLP preprocessing, matching engine, and Flask API.
"""

import json
import os
import unittest
from app import app
from chatbot.matcher import FAQMatcher
from chatbot.preprocessing import (
    clean_text,
    tokenize_words,
    remove_stopwords,
    lemmatize_token,
    preprocess_pipeline
)


class TestFAQKnowledgeBase(unittest.TestCase):
    """Verifies that faqs.json exists, is valid JSON, and has required fields."""

    def setUp(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.faqs_path = os.path.join(base_dir, "data", "faqs.json")

    def test_faqs_file_exists(self):
        self.assertTrue(os.path.exists(self.faqs_path), "faqs.json file does not exist.")

    def test_faqs_count_and_schema(self):
        with open(self.faqs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list, "faqs.json should contain a JSON array.")
        self.assertGreaterEqual(len(data), 30, "Knowledge base must have at least 30 FAQs.")

        required_keys = {"id", "category", "question", "answer"}
        categories = set()
        for idx, item in enumerate(data):
            self.assertTrue(
                required_keys.issubset(item.keys()),
                f"FAQ at index {idx} is missing required keys: {required_keys - set(item.keys())}"
            )
            self.assertTrue(item["question"].strip(), f"FAQ at index {idx} has an empty question.")
            self.assertTrue(item["answer"].strip(), f"FAQ at index {idx} has an empty answer.")
            categories.add(item["category"])

        # Check coverage across multiple student service categories
        self.assertGreaterEqual(len(categories), 8, "Expected coverage of at least 8 distinct categories.")


class TestNLPPreprocessing(unittest.TestCase):
    """Verifies each stage of the NLP text preprocessing pipeline."""

    def test_clean_text(self):
        raw = "Hello World! What's the fee payment schedule? (2026)"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "hello world what s the fee payment schedule 2026")
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)

    def test_tokenize_words(self):
        text = "student admission process"
        tokens = tokenize_words(text)
        self.assertEqual(tokens, ["student", "admission", "process"])

    def test_remove_stopwords(self):
        tokens = ["how", "can", "i", "apply", "for", "hostel"]
        filtered = remove_stopwords(tokens)
        self.assertIn("apply", filtered)
        self.assertIn("hostel", filtered)
        self.assertNotIn("how", filtered)
        self.assertNotIn("can", filtered)
        self.assertNotIn("for", filtered)

    def test_lemmatize_token(self):
        self.assertEqual(lemmatize_token("applying"), "apply")
        self.assertEqual(lemmatize_token("fees"), "fee")
        self.assertEqual(lemmatize_token("scholarships"), "scholarship")
        self.assertEqual(lemmatize_token("registered"), "register")

    def test_preprocess_pipeline(self):
        text = "How can I apply for admission to the university?"
        processed = preprocess_pipeline(text)
        self.assertIn("apply", processed)
        self.assertIn("admission", processed)
        self.assertIn("university", processed)


class TestFAQMatcher(unittest.TestCase):
    """Tests the TF-IDF Vectorization and Cosine Similarity matching engine."""

    @classmethod
    def setUpClass(cls):
        cls.matcher = FAQMatcher(similarity_threshold=0.25)

    def test_exact_question_match(self):
        query = "How can I apply for admission to the university?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Admissions")
        self.assertGreater(result["confidence"], 0.8)

    def test_paraphrased_question_match(self):
        query = "Where do I submit documents to apply for admission?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Admissions")
        self.assertGreaterEqual(result["confidence"], 0.25)

    def test_hostel_amenities_match(self):
        query = "What amenities and facilities are included in hostel fees?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Hostel")

    def test_irrelevant_query_triggers_fallback(self):
        query = "How do I bake a strawberry cheesecake?"
        result = self.matcher.match(query)
        self.assertTrue(result["is_fallback"])
        self.assertEqual(result["category"], "Fallback")
        self.assertIsNone(result["matched_question"])
        self.assertIn("couldn't find a reliable answer", result["answer"])

    def test_empty_query_handling(self):
        result = self.matcher.match("")
        self.assertTrue(result["is_fallback"])

        result_spaces = self.matcher.match("     ")
        self.assertTrue(result_spaces["is_fallback"])

    def test_admission_paraphrase_variant(self):
        query = "How do I get admission?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Admissions")
        self.assertGreaterEqual(result["confidence"], 0.8)
        self.assertEqual(result["matched_question"], "How can I apply for admission to the university?")

    def test_scholarship_ambiguity_not_sports(self):
        query = "How can I get a scholarship?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Scholarships")
        self.assertNotIn("sports", result["answer"].lower())
        self.assertNotIn("extracurricular", result["answer"].lower())
        self.assertEqual(result["matched_question"], "Are merit-based scholarships available for students?")

    def test_sports_scholarship_intent(self):
        query = "Are there sports scholarships?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Scholarships")
        self.assertIn("sports", result["answer"].lower())
        self.assertEqual(result["matched_question"], "Are there sports or extracurricular scholarships available?")

    def test_exam_schedule_vs_reevaluation(self):
        query = "When are exams conducted?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Exams")
        self.assertNotIn("re-evaluation", result["answer"].lower())
        self.assertNotIn("recheck", result["answer"].lower())
        self.assertEqual(result["matched_question"], "When will the semester examination schedule be announced?")

    def test_reevaluation_intent(self):
        query = "How do I apply for re-evaluation?"
        result = self.matcher.match(query)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["category"], "Exams")
        self.assertIn("re-evaluation", result["answer"].lower())
        self.assertEqual(result["matched_question"], "How can I apply for re-evaluation or rechecking of exam papers?")

    def test_ambiguous_fee_question(self):
        query = "What about fees?"
        result = self.matcher.match(query)
        self.assertTrue(result["is_fallback"])
        self.assertEqual(result["category"], "Fallback")
        self.assertIn("broad or ambiguous", result["answer"].lower())

    def test_greeting_conversational(self):
        for greeting in ["Hi", "hello", "hey", "good morning", "good evening"]:
            result = self.matcher.match(greeting)
            self.assertFalse(result["is_fallback"])
            self.assertEqual(result["category"], "Greeting")
            self.assertIn("Hello! I'm the University FAQ Assistant", result["answer"])

    def test_gratitude_conversational(self):
        for thanks in ["Thank you", "thanks", "thank you so much", "thanks a lot"]:
            result = self.matcher.match(thanks)
            self.assertFalse(result["is_fallback"])
            self.assertEqual(result["category"], "Gratitude")
            self.assertIn("You're welcome", result["answer"])

    def test_manual_test_cases_complete_suite(self):
        """Verifies all manual test cases A through M requested in specification."""
        cases = [
            ("How do I get admission?", "Admissions", False),
            ("How can I apply for admission?", "Admissions", False),
            ("Can I pay fees online?", "Fees", False),
            ("How can I get a scholarship?", "Scholarships", False),
            ("Are there sports scholarships?", "Scholarships", False),
            ("When are exams conducted?", "Exams", False),
            ("How do I apply for re-evaluation?", "Exams", False),
            ("What about fees?", "Fallback", True),
            ("I lost my student ID.", "Student ID", False),
            ("Tell me a joke.", "Fallback", True),
            ("Hi", "Greeting", False),
            ("Thank you", "Gratitude", False),
            ("asdfghjkl", "Fallback", True),
        ]
        for query, exp_cat, exp_fallback in cases:
            res = self.matcher.match(query)
            self.assertEqual(res["category"], exp_cat, f"Failed category for: {query}")
            self.assertEqual(res["is_fallback"], exp_fallback, f"Failed fallback for: {query}")


class TestFlaskAPI(unittest.TestCase):
    """Verifies Flask HTTP endpoints."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FAQ AI Assistant", response.data)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["faq_count"], 30)

    def test_sample_questions_endpoint(self):
        response = self.client.get("/api/sample-questions")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_chat_endpoint_valid_question(self):
        payload = {"message": "How can I pay my semester fee online?"}
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("answer", data)
        self.assertIn("category", data)
        self.assertIn("confidence", data)
        self.assertEqual(data["category"], "Fees")
        self.assertFalse(data["is_fallback"])

    def test_chat_endpoint_empty_message(self):
        payload = {"message": "   "}
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_chat_endpoint_missing_message(self):
        payload = {"text": "Hello"}
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_chat_endpoint_fallback(self):
        payload = {"message": "Quantum mechanics string theory astrophysics galaxy"}
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["is_fallback"])
        self.assertEqual(data["category"], "Fallback")


if __name__ == "__main__":
    unittest.main()
