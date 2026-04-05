"""
guardrails.py — NUST Bank query safety filter

Logic:
  1. Block prompt injections (always)
  2. Block clearly off-topic queries (cooking, sports, weather, etc.)
  3. Allow everything else — banking, social, follow-ups, comparisons
"""

import re

# ─── Block list: prompt injection attempts ────────────────────────
INJECTION_PATTERNS = [
    "ignore previous", "ignore all", "system prompt", "you are now",
    "new persona", "dev mode", "jailbreak", "rewrite everything",
    "forget your instructions", "disregard", "act as if",
    "pretend you are", "override",
]

# ─── Block list: clearly off-topic domains ────────────────────────
OFF_TOPIC_PATTERNS = [
    # food / cooking
    r"\b(recipe|cooking|cook|bake|boil|fry|ingredient|cuisine|meal|dish|food)\b",
    # sports
    r"\b(cricket|football|soccer|tennis|hockey|psl|ipl|fifa|match|tournament|stadium)\b",
    # weather
    r"\b(weather|forecast|temperature|rain|cloudy|sunny|humidity)\b",
    # entertainment
    r"\b(movie|film|song|drama|series|netflix|youtube|tiktok|instagram)\b",
    # medical (non-financial)
    r"\b(doctor|hospital|medicine|treatment|disease|symptom|diagnosis)\b",
]

# ─── Allow list: banking & financial keywords ─────────────────────
BANKING_KEYWORDS = [
    "bank", "account", "rate", "loan", "finance", "nust", "profit",
    "interest", "card", "sme", "insurance", "remit", "digital",
    "saving", "current", "deposit", "credit", "debit", "transfer",
    "payment", "salary", "cheque", "check", "branch", "atm", "fee",
    "charge", "balance", "statement", "open", "close", "apply",
    "eligibility", "requirement", "document", "kyc", "zakat",
    # account codes
    "lca", "naa", "nwa", "pwra", "rda", "vpca", "vpba", "nsda",
    "pls", "cda", "nma", "nada", "nadra", "nfda", "nsa", "nmf",
    "nsf", "nif", "nuf", "nfmf", "nfbf", "nhf", "nrf", "nmc",
    # product names
    "sahar", "waqaar", "asaan", "maximiser", "imarat", "ujala",
    "mortgage", "mastercard", "freelancer", "roshan", "hunarmand",
    "pakwatan", "champs", "fauri", "ujala", "rice",
]

# ─── Allow list: conversational / follow-up phrases ──────────────
CONVERSATIONAL_KEYWORDS = [
    "hi", "hello", "hey", "salam", "thanks", "thank you", "bye",
    "how are you", "who are you", "your name", "what can you do",
    "help me", "tell me", "show me", "explain", "describe",
    "what is", "what are", "how do", "how does", "can i", "can you",
    "compare", "difference", "which one", "best", "recommend",
    "elaborate", "detail", "list", "types", "options", "features",
    "yes", "no", "ok", "sure", "please", "above", "earlier",
    "previous", "remember", "what did you", "also", "and", "more",
]


def is_query_safe(query: str) -> bool:
    """
    Returns True if the query should be processed, False if it should be blocked.
    """
    if not query or not query.strip():
        return False

    q = query.lower().strip()

    # 1. Block prompt injections
    if any(pattern in q for pattern in INJECTION_PATTERNS):
        return False

    # 2. Block clearly off-topic
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q):
            return False

    # 3. Allow banking keywords
    if any(kw in q for kw in BANKING_KEYWORDS):
        return True

    # 4. Allow conversational / follow-up
    if any(kw in q for kw in CONVERSATIONAL_KEYWORDS):
        return True

    # 5. FIX: default ALLOW (not block).
    # A small model's response to an off-topic question is harmless.
    # Silently blocking valid questions destroys user trust.
    # Let the model decide how to respond to edge cases.
    return True