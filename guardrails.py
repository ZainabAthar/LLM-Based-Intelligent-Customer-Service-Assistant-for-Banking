# --- NUST BANK HYBRID GUARDRAILS ---

def is_query_safe(query: str) -> bool:
    """
    Hybrid guardrail:
    - ALLOW: social/conversational + banking queries
    - BLOCK: prompt injection + clearly off-topic queries
    - DEFAULT: block (with polite redirect from the AI)
    """
    if not query or not query.strip():
        return False
    
    q = query.lower().strip()
    
    # 1. BLOCK: Prompt injection
    injection_patterns = [
        "ignore previous", "system prompt", "you are now a", 
        "dev mode", "rewrite everything", "forget your instructions"
    ]
    if any(bad in q for bad in injection_patterns):
        return False

    # 2. ALLOW: Social & conversational (broad enough for follow-ups)
    social_patterns = [
        "hi", "hello", "hey", "name", "i am", "call me", "who are you",
        "thank", "thanks", "bye", "how are you", "remember", "told you",
        "what did", "above", "earlier", "previous",
        "yes", "no", "ok", "sure", "please", "help me", "tell me more",
        "can you explain", "could you explain", "show me",
        "compare", "difference", "which one", "best", "recommend",
        "elaborate", "detail", "list", "enlist", "types", "options"
    ]
    if any(k in q for k in social_patterns):
        return True

    # 3. ALLOW: Banking & financial terms
    banking_patterns = [
        "bank", "account", "rate", "loan", "finance", "nust", "profit", 
        "interest", "card", "sme", "insurance", "remit", "digital",
        "saving", "current", "deposit", "credit", "debit", "transfer",
        "payment", "salary", "cheque", "check", "branch", "atm",
        "lca", "naa", "nwa", "pwra", "rda", "vpca", "vpba", "nsda",
        "pls", "cda", "nma", "nada", "nadra", "nfda", "nsa", "nmf",
        "nsf", "nif", "nuf", "nfmf", "nfbf", "nhf", "nrf", "nmc",
        "sahar", "waqaar", "asaan", "maximiser", "imarat", "ujala",
        "mortgage", "mastercard", "freelancer", "roshan", "hunarmand"
    ]
    if any(k in q for k in banking_patterns):
        return True

    # 4. DEFAULT: Block off-topic (recipe, weather, sports, etc.)
    return False
