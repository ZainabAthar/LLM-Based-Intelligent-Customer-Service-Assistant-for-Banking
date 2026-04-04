# preprocessing_new.py
import pandas as pd
import re
import json
import hashlib
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
EXCEL_PATH = "product_info.xlsx"                     # Your Excel file
OUTPUT_JSONL = "preprocessed_data/bank_data_master_test.jsonl"
RATE_SHEET_JSONL = "preprocessed_data/rate_sheet_only_test.jsonl"

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
    "Jubilee Life ": "Jubilee Life Bancassurance Policy",
    "HOME REMITTANCE": "Home Remittance Services",
}
DEFAULT_NAME = "NUST Bank Product"

# Sheets to skip (no Q&A content)
SKIP_SHEETS = ["Main", "Rate Sheet July 1 2024", "Sheet1"]

# PII patterns
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
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    if t.endswith('?'):
        return True
    lower = t.lower()
    if lower.startswith(('what', 'how', 'why', 'when', 'where', 'can', 'is', 'are', 'does', 'do', 'which')):
        return True
    return False

def rows_to_markdown_table(rows):
    """
    Convert a list of rows (each a list of strings) into a Markdown table.
    Assumes the first row is the header. Rows are padded to equal length.
    """
    if not rows:
        return ""
    num_cols = max(len(r) for r in rows)
    # Pad each row to num_cols
    norm_rows = [r + [''] * (num_cols - len(r)) for r in rows]
    # Create header row (first row)
    header = "| " + " | ".join(norm_rows[0]) + " |"
    # Separator
    separator = "|" + "|".join([" --- " for _ in range(num_cols)]) + "|"
    # Body rows (remaining)
    body_rows = ["| " + " | ".join(row) + " |" for row in norm_rows[1:]]
    return "\n".join([header, separator] + body_rows)

def extract_qa_from_sheet(df, sheet_name):
    """
    Extract Q&A pairs. Preserves table structure:
    - Each row is read as a list of cells (all non‑empty columns).
    - Non‑question rows are collected as lists of cells.
    - If the answer consists of multiple rows and at least one row has multiple cells,
      it is formatted as a Markdown table; otherwise joined as plain text.
    """
    qa_pairs = []
    current_q = None
    current_answer_rows = []   # list of list of strings (each inner list is a row of cells)

    for idx, row in df.iterrows():
        # Collect non‑empty cells in this row
        cells = [clean_text(str(cell)) for cell in row if pd.notna(cell) and str(cell).strip()]
        if not cells:
            continue
        row_text = " ".join(cells)

        # Skip the "Main" marker
        if row_text.lower() == "main":
            continue

        if is_question(row_text):
            # Save previous Q&A
            if current_q and current_answer_rows:
                # Determine if this answer looks like a table
                if len(current_answer_rows) > 1 and any(len(r) > 1 for r in current_answer_rows):
                    answer = rows_to_markdown_table(current_answer_rows)
                else:
                    answer = " ".join(" ".join(r) for r in current_answer_rows)
                qa_pairs.append((current_q, answer))
            # Start new question
            current_q = row_text
            current_answer_rows = []
        else:
            # This row is part of the answer – store the raw cell list
            if current_q:
                current_answer_rows.append(cells)

    # Last pair
    if current_q and current_answer_rows:
        if len(current_answer_rows) > 1 and any(len(r) > 1 for r in current_answer_rows):
            answer = rows_to_markdown_table(current_answer_rows)
        else:
            answer = " ".join(" ".join(r) for r in current_answer_rows)
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

        account_name = ACCOUNT_NAME_MAP.get(sheet, DEFAULT_NAME)

        for q, a in qa_list:
            enriched = f"[Account: {account_name}]\nQuestion: {q}\nAnswer: {a}"
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

    # Merge rate sheet data if exists
    if Path(RATE_SHEET_JSONL).exists():
        print(f"Merging specialized rate sheet data from {RATE_SHEET_JSONL}...")
        with open(RATE_SHEET_JSONL, 'r', encoding='utf-8') as rf:
            for line in rf:
                rate_rec = json.loads(line)
                if rate_rec.get("hash_id") not in seen_hashes:
                    all_records.append(rate_rec)
                    seen_hashes.add(rate_rec.get("hash_id"))
        print(f"  Merged records. Total now: {len(all_records)}")
    else:
        print(f"  Warning: {RATE_SHEET_JSONL} not found. Skipping merge.")

    # Write output
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\nFinished. Extracted {len(all_records)} Q&A pairs.")
    print(f"Output saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    process_excel()