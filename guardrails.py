import re

# --- Safety Configuration ---
PII_PATTERNS = {
    "CNIC": r'\b\d{5}-\d{7}-\d{1}\b',
    "PHONE": r'(\+92|0)3\d{2}-\d{7}|\b0\d{2}-\d{7,8}\b',
    "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    "IBAN": r'\bPK\d{2}[A-Z]{4}[A-Z0-9]{16}\b',
    "ACCOUNT_NUMBER": r'\b\d{10,14}\b'
}

def check_pii(text):
    """Detect if any PII is present in the text."""
    found_pii = []
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            found_pii.append(label)
    return found_pii

def mask_output(text):
    """Mask any accidentally generated PII in the model output."""
    masked_text = text
    for label, pattern in PII_PATTERNS.items():
        masked_text = re.sub(pattern, f"<{label}_MASKED>", masked_text)
    return masked_text

def validate_response(query, context, response):
    """
    Final check on the assistant's response.
    1. Check for PII.
    2. (Placeholder) Check for hallucinations by comparing with context.
    """
    # Mask any PII that might have been generated or leaked
    safe_response = mask_output(response)
    
    # Simple check: If context is very small or irrelevant, flag it
    if len(context.strip()) < 10:
        return "I'm sorry, I don't have enough information in my database to answer that accurately."
        
    return safe_response

if __name__ == "__main__":
    test_text = "My account number is 123456789012 and my CNIC is 12345-1234567-1"
    print(f"Original: {test_text}")
    print(f"Masked: {mask_output(test_text)}")
    print(f"Detected PII: {check_pii(test_text)}")
