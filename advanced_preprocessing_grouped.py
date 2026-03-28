import pandas as pd
import re
import os
import json
import hashlib
from datetime import datetime
import argparse

# --- Configuration ---
SENSITIVE_PATTERNS = {
    "CNIC": r'\b\d{5}-\d{7}-\d{1}\b',
    "PHONE": r'(\+92|0)3\d{2}-\d{7}|\b0\d{2}-\d{7,8}\b',
    "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    "IBAN": r'\bPK\d{2}[A-Z]{4}[A-Z0-9]{16}\b',
    "ACCOUNT_NUMBER": r'\b\d{10,14}\b',
    "CREDIT_CARD": r'\b(?:\d[ -]*?){13,16}\b',
    "SECRET_KEYS": r'(?i)(password|secret|key|token|api_key)\s*[:=]\s*[^\s]+'
}


def generate_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def expand_units(text):
    if not isinstance(text, str):
        text = str(text)
    
    def handle_k(match):
        val = match.group(1)
        try:
            expanded = int(float(val) * 1000)
            return f"{expanded:,}"
        except:
            return match.group(0)
    
    text = re.sub(r'\b(\d+\.?\d*)K\b', handle_k, text, flags=re.IGNORECASE)
    
    def handle_m(match):
        val = match.group(1)
        try:
            expanded = int(float(val) * 1000000)
            return f"{expanded:,}"
        except:
            return match.group(0)
    
    text = re.sub(r'\b(\d+\.?\d*)M\b', handle_m, text, flags=re.IGNORECASE)
    return text


def anonymize_text(text):
    if not isinstance(text, str):
        text = str(text)
    for label, pattern in SENSITIVE_PATTERNS.items():
        text = re.sub(pattern, f"<{label}_MASKED>", text)
    return text


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# import question‑detection helpers from retrieval if available
try:
    from retrieval import looks_like_question
except ImportError:
    # fallback simple heuristic
    def looks_like_question(s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        return t.endswith('?') or t.lower().startswith(
            ('what', 'how', 'why', 'when', 'where', 'can', 'is', 'are')
        )


def process_bank_data_rowwise(file_path):
    """Original behaviour: one record per non‑empty row."""
    xl = pd.ExcelFile(file_path)
    all_records = []
    seen_hashes = set()

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, header=0 if False else None)
        df = df.fillna("")
        df.columns = [str(c).strip() for c in df.columns]

        last_valid_col = "Description"
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            semantic_parts = []
            for col, val in row_dict.items():
                col_name = str(col).strip()
                val_str = str(val).strip()
                if not val_str or val_str.lower() in ["nan", ""] or val_str == "0.0":
                    continue
                val_str = expand_units(val_str)
                cleaned = clean_text(anonymize_text(val_str))
                if re.match(r'^0\.\d+$', cleaned):
                    try:
                        pct = float(cleaned) * 100
                        cleaned = f"{pct:.2f}%"
                    except:
                        pass
                semantic_parts.append(f"{col_name}: {cleaned}")
            if not semantic_parts:
                continue
            content = f"BANK RECORD (Chapter: {sheet_name})\n" + " | ".join(semantic_parts)
            h = generate_hash(content)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            all_records.append({
                "hash_id": h,
                "metadata": {"source": os.path.basename(file_path),
                             "sheet": sheet_name,
                             "row": int(idx) + 1,
                             "processed_at": datetime.now().isoformat()},
                "content": content
            })
    return all_records


def process_bank_data_grouped(file_path, mode='sheet'):
    """Group the spreadsheet into bigger documents.
    mode: 'sheet' = one document per sheet
          'qa'    = question+answer pairs extracted from each sheet

    We also deduplicate records by hash so that repeated questions/answers
    or sheet-level concatenations don't generate collisions when indexed.
    """
    xl = pd.ExcelFile(file_path)
    records = []
    seen_hashes = set()

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, header=None).fillna("")
        # convert each row to a line of text
        lines = []
        for _, r in df.iterrows():
            vals = [str(v).strip() for v in r.tolist() if pd.notna(v) and str(v).strip()]
            if vals:
                lines.append(" | ".join(vals))

        if mode == 'sheet':
            # include sheet name explicitly so embeddings know which account
            content = f"Account: {sheet_name}\nSHEET: {sheet_name}\n" + "\n".join(lines)
            h = generate_hash(content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                records.append({
                    "hash_id": h,
                    "metadata": {"source": os.path.basename(file_path), "sheet": sheet_name},
                    "content": content
                })
        elif mode == 'qa':
            current_q = None
            current_a = []
            for line in lines:
                if looks_like_question(line):
                    if current_q and current_a:
                        cont = f"Account: {sheet_name}\nQ: {current_q}\nA: {' '.join(current_a)}"
                        h = generate_hash(cont)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            records.append({
                                "hash_id": h,
                                "metadata": {"source": os.path.basename(file_path), "sheet": sheet_name},
                                "content": cont
                            })
                    current_q = line
                    current_a = []
                else:
                    if current_q:
                        current_a.append(line)
            if current_q and current_a:
                cont = f"Account: {sheet_name}\nQ: {current_q}\nA: {' '.join(current_a)}"
                h = generate_hash(cont)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    records.append({
                        "hash_id": h,
                        "metadata": {"source": os.path.basename(file_path), "sheet": sheet_name},
                        "content": cont
                    })
        else:
            raise ValueError(f"unknown mode {mode}")
    return records


def save_output(records, output_path_jsonl, output_path_json):
    os.makedirs(os.path.dirname(output_path_jsonl), exist_ok=True)
    with open(output_path_jsonl, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(output_path_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Success! Generated unique records: {len(records)}")
    print(f"Saved to: {output_path_jsonl} and {output_path_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="product_info.xlsx", help="Input Excel file")
    ap.add_argument("--out", default="preprocessed_data/bank_data_advanced_qa.jsonl",
                    help="Output JSONL path")
    ap.add_argument("--mode", choices=["row","sheet","qa"], default="row",
                    help="Granularity: row=original, sheet=one doc per sheet, qa=extract question/answer pairs")
    args = ap.parse_args()

    out_json = os.path.splitext(args.out)[0] + ".json"

    if args.mode == "row":
        records = process_bank_data_rowwise(args.xlsx)
    else:
        records = process_bank_data_grouped(args.xlsx, mode=args.mode)
    if records:
        save_output(records, args.out, out_json)
