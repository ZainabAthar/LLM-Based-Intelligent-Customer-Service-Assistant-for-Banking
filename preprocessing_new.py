# enriched_preprocessing_v2.py
import pandas as pd
import re
import json
import hashlib
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
EXCEL_PATH = "product_info.xlsx"                     # Your Excel file
OUTPUT_JSONL = "preprocessed_data/enriched_bank_data.jsonl"

# Map sheet codes to full, user‑friendly account names
ACCOUNT_NAME_MAP = {
    "LCA": "Little Champs Account",
    "NAA": "NUST Asaan Account (NAA)",
    "NWA": "NUST Waqaar Account",
    "PWRA": "PakWatan Remittance Account",
    "RDA": "Roshan Digital Account",
    "VPCA": "Value Plus Current Account",
    "VP-BA": "Value Plus Business Account",
    "VPBA": "NUST Value Premium Business Account",
    "NSDA": "NUST Special Deposit Account",
    "PLS": "Profit and Loss Sharing Account (PLS)",
    "CDA": "Current Deposit Account",
    "NMA": "NUST Maximiser Account",
    "NADA": "NUST Asaan Digital Account",
    "NADRA": "NUST Asaan Digital Remittance Account",
    "NUST4Car": "NUST4Car Auto Finance",
    "ESFCA": "Exporters’ Special Foreign Currency Account",
    "NFDA": "NUST Freelancer Digital Account",
    "NSA": "NUST Sahar Accounts",
    "PF": "NUST Personal Finance",
    "NMC": "NUST Bank Mastercard",
    "NMF": "NUST Mortgage Finance",
    "NSF": "NUST Sahar Finance",
    "NIF": "NUST Imarat Finance",
    "NUF": "NUST Ujala Finance",
    "NFMF": "NUST Flour Mill Finance",
    "NFBF": "NUST Fauri Business Finance",
    "PMYB &ALS": "Prime Minister Youth Business & Agriculture Loan Scheme",
    "NRF": "NUST Rice Finance",
    "NHF": "NUST Hunarmand Finance",
    "Nust Life": "NUST Life Bancassurance Policy",
    "EFU Life": "EFU Life Bancassurance Policy",
    "Jubilee Life": "Jubilee Life Bancassurance Policy",
    "HOME REMITTANCE": "Home Remittance Services",
}
DEFAULT_NAME = "NUST Bank Product"

# Sheets to skip (no Q&A content)
SKIP_SHEETS = ["Main", "Rate Sheet July 1 2024", "Sheet1"]

# PII patterns (same as your guardrails)
PII_PATTERNS = {
    "CNIC": r'\b\d{5}-\d{7}-\d{1}\b',
    "PHONE": r'(\+92|0)3\d{2}-\d{7}|\b0\d{2}-\d{7,8}\b',
    "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    "IBAN": r'\bPK\d{2}[A-Z]{4}[A-Z0-9]{16}\b',
    "ACCOUNT_NUMBER": r'\b\d{10,14}\b',
}
# ----------------------------------------------------------------------

def anonymize_text(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"<{label}_MASKED>", text)
    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_question(text: str) -> bool:
    """Heuristic to detect a question line."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    # Ends with '?' or starts with typical question words
    if t.endswith('?'):
        return True
    lower = t.lower()
    if lower.startswith(('what', 'how', 'why', 'when', 'where', 'can', 'is', 'are', 'does', 'do', 'which')):
        return True
    return False

def extract_qa_from_sheet(df, sheet_name):
    """
    Extract Q&A pairs by looking at each row's full text (all columns combined).
    - If a row looks like a question, it starts a new pair.
    - Subsequent non‑question rows are appended to the current answer.
    Works for any column layout.
    """
    qa_pairs = []
    current_q = None
    current_a_lines = []

    for idx, row in df.iterrows():
        # Combine all non‑empty cells in this row into one string
        cells = [clean_text(str(cell)) for cell in row if pd.notna(cell) and str(cell).strip()]
        if not cells:
            continue
        row_text = " ".join(cells)

        # Skip the "Main" marker (appears in some sheets)
        if row_text.lower() == "main":
            continue

        if is_question(row_text):
            # Save previous pair if any
            if current_q and current_a_lines:
                answer = " ".join(current_a_lines).strip()
                if answer:
                    qa_pairs.append((current_q, answer))
            # Start new question
            current_q = row_text
            current_a_lines = []
        else:
            # Non‑question → part of the current answer
            if current_q:
                current_a_lines.append(row_text)

    # Last pair
    if current_q and current_a_lines:
        answer = " ".join(current_a_lines).strip()
        if answer:
            qa_pairs.append((current_q, answer))

    return qa_pairs

def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_excel():
    xl = pd.ExcelFile(EXCEL_PATH)
    all_records = []
    seen_hashes = set()

    for sheet in xl.sheet_names:
        if sheet in SKIP_SHEETS:
            print(f"Skipping sheet: {sheet}")
            continue

        print(f"Processing: {sheet}")
        df = xl.parse(sheet, header=None)
        qa_list = extract_qa_from_sheet(df, sheet)

        if not qa_list:
            print(f"  Warning: No Q&A pairs found for {sheet}")
            continue

        # Get friendly account name
        account_name = ACCOUNT_NAME_MAP.get(sheet, DEFAULT_NAME)

        for q, a in qa_list:
            # Build enriched content
            enriched = f"[Account: {account_name}]\nQuestion: {q}\nAnswer: {a}"
            # Anonymise any PII
            enriched = anonymize_text(enriched)
            doc_id = generate_hash(enriched)

            if doc_id in seen_hashes:
                continue
            seen_hashes.add(doc_id)

            record = {
                "hash_id": doc_id,
                "metadata": {
                    "source": EXCEL_PATH,
                    "sheet": sheet,
                    "account_name": account_name,
                    "question": q
                },
                "content": enriched,
                "structured_data": {
                    "question": q,
                    "answer": a,
                    "account": account_name
                }
            }
            all_records.append(record)

    # Write output
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\n✅ Finished. Extracted {len(all_records)} Q&A pairs.")
    print(f"Output saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    process_excel()