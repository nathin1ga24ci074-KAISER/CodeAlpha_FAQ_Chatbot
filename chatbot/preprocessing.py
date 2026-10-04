"""
NLP Preprocessing Module for FAQ Chatbot
Handles text cleaning, tokenization, stopword removal, and lemmatization using NLTK.
"""

import re
import string
import logging
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# Ensure required NLTK resources are available
def ensure_nltk_resources():
    """Download required NLTK datasets quietly if not already present."""
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, pkg_name in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception as exc:
                logger.warning("Could not download NLTK package '%s': %s", pkg_name, exc)

# Initialize resources once
ensure_nltk_resources()

# Initialize Lemmatizer and Stopwords
_lemmatizer = WordNetLemmatizer()
try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
        "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
    }


def clean_text(text: str) -> str:
    """
    Lowercases text and strips punctuation, symbols, and extra whitespace.

    Args:
        text (str): Input raw text string.

    Returns:
        str: Cleaned lowercase string with punctuation removed.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Replace punctuation characters with space
    pattern = f"[{re.escape(string.punctuation)}]"
    cleaned = re.sub(pattern, " ", cleaned)

    # Collapse multiple whitespaces into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes text into individual words using NLTK word_tokenize with fallback.

    Args:
        text (str): Pre-cleaned text.

    Returns:
        List[str]: List of word tokens.
    """
    if not text:
        return []
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback to regex word boundary tokenization if NLTK fails
        tokens = re.findall(r"\b\w+\b", text)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Removes English stop words from the token list.

    Args:
        tokens (List[str]): List of word tokens.

    Returns:
        List[str]: Filtered list of non-stopword tokens.
    """
    return [token for token in tokens if token not in _stop_words and len(token) > 1]


def lemmatize_token(token: str) -> str:
    """
    Normalizes a token using WordNet lemmatizer.
    Applies verb lemmatization followed by noun lemmatization to handle inflections.

    Args:
        token (str): A single word token.

    Returns:
        str: Lemmatized base form of the token.
    """
    # Attempt verb reduction first (e.g. 'applying' -> 'apply', 'registered' -> 'register')
    verb_lemma = _lemmatizer.lemmatize(token, pos="v")
    if verb_lemma != token:
        return verb_lemma
    # Otherwise noun reduction (e.g. 'fees' -> 'fee', 'admissions' -> 'admission')
    return _lemmatizer.lemmatize(token, pos="n")


def tokenize_and_normalize(text: str) -> List[str]:
    """
    Executes the full preprocessing pipeline returning list of normalized tokens:
    1. Cleaning & Lowercasing
    2. Tokenization
    3. Stopword Removal
    4. Lemmatization

    Args:
        text (str): Raw input text.

    Returns:
        List[str]: Processed token list.
    """
    cleaned = clean_text(text)
    tokens = tokenize_words(cleaned)
    filtered = remove_stopwords(tokens)
    lemmatized = [lemmatize_token(tok) for tok in filtered]
    return lemmatized


def preprocess_pipeline(text: str) -> str:
    """
    Standard preprocessing pipeline producing a single normalized string suitable
    for TF-IDF vectorization.

    Args:
        text (str): Raw input text string.

    Returns:
        str: Space-separated normalized tokens.
    """
    tokens = tokenize_and_normalize(text)
    # If all tokens were filtered out (e.g. only stopwords were entered),
    # fall back to cleaned tokens so vectorizer has some non-empty input if possible
    if not tokens:
        raw_tokens = tokenize_words(clean_text(text))
        tokens = [lemmatize_token(t) for t in raw_tokens if len(t) > 1]
    return " ".join(tokens)
