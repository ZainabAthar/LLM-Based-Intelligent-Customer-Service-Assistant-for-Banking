import pandas as pd
import re
import os
import json
import hashlib
from datetime import datetime

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
    """Convert 18K to 18,000 and 1.5M to 1,500,000 for easier LLM understanding."""
    if not isinstance(text, str):
        text = str(text)
    
    # Handle K (Thousands)
    def handle_k(match):
        val = match.group(1)
        try:
            expanded = int(float(val) * 1000)
            return f"{expanded:,}"
        except: return match.group(0)
        
    text = re.sub(r'\b(\d+\.?\d*)K\b', handle_k, text, flags=re.IGNORECASE)
    
    # Handle M (Millions)
    def handle_m(match):
        val = match.group(1)
        try:
            expanded = int(float(val) * 1000000)
            return f"{expanded:,}"
        except: return match.group(0)
        
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

def find_best_header_row(file_path, sheet_name):
    """Scan first 20 rows to find the most likely header row. 
    Improved to consider rows with more text-like (non-numeric) entries."""
    xl = pd.ExcelFile(file_path)
    df_temp = xl.parse(sheet_name, header=None, nrows=20)
    
    best_row_idx = 0
    max_score = 0
    
    for i, row in df_temp.iterrows():
        non_empty = row.dropna()
        if non_empty.empty:
            continue
            
        # Score based on number of string-type columns (headers are usually strings)
        score = sum(1 for x in non_empty if isinstance(x, str) and len(str(x).strip()) > 1)
        
        if score > max_score:
            max_score = score
            best_row_idx = i
            
    return best_row_idx

def process_bank_data_robust(file_path):
    if not os.path.exists(file_path):
        print(f" Error: File {file_path} not found.")
        return None

    xl = pd.ExcelFile(file_path)
    all_records = []
    seen_hashes = set()

    for sheet_name in xl.sheet_names:
        print(f"Robustly parsing sheet: {sheet_name}...")
        
        # 1. Find the best header row
        header_row = find_best_header_row(file_path, sheet_name)
        
        # 2. Parse with the detected header
        try:
            df = xl.parse(sheet_name, header=header_row)
            df = df.fillna("")
            
            # Clean up column names (remove 'Unnamed', trim spaces)
            df.columns = [str(c).strip() for c in df.columns]
            df.columns = [c if "Unnamed" not in c else f"Info_{i}" for i, c in enumerate(df.columns)]
        except Exception as e:
            print(f" Error parsing {sheet_name}: {e}")
            continue

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            semantic_parts = []
            last_valid_col = "Description"
            
            for col, val in row_dict.items():
                col_name = str(col).strip()
                val_str = str(val).strip()
                
                # Force remove any residual "Unnamed" or "Info_" strings
                is_unnamed = "Unnamed" in col_name or "Info_" in col_name
                current_label = last_valid_col if is_unnamed else col_name
                
                if not is_unnamed:
                    last_valid_col = col_name

                if not val_str or val_str.lower() in ["nan", ""] or val_str == "0.0":
                    continue
                
                # 1. Expand K/M units first
                val_str = expand_units(val_str)
                
                # 2. Anonymize/Clean
                anonymized = anonymize_text(val_str)
                cleaned = clean_text(anonymized)
                
                # 3. Handle rates (0.15 -> 15.00%)
                if re.match(r'^0\.\d+$', cleaned):
                    try:
                        pct = float(cleaned) * 100
                        cleaned = f"{pct:.2f}%"
                    except: pass

                semantic_parts.append(f"{current_label}: {cleaned}")

            if not semantic_parts:
                continue

            # Full text combined for embedding - semi-structured for small models
            content = f"BANK RECORD (Chapter: {sheet_name})\n" + " | ".join(semantic_parts)
            
            content_hash = generate_hash(content)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            record = {
                "hash_id": content_hash,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "sheet": sheet_name,
                    "row": int(idx + header_row + 1),
                    "processed_at": datetime.now().isoformat()
                },
                "content": content
            }
            all_records.append(record)

    return all_records

def save_output(records, output_path_jsonl, output_path_json):
    os.makedirs(os.path.dirname(output_path_jsonl), exist_ok=True)
    
    # Save JSONL
    with open(output_path_jsonl, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    # Save standard JSON for legacy compatibility
    with open(output_path_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    print(f" Success! Generated unique records: {len(records)}")
    print(f" Saved to: {output_path_jsonl} and {output_path_json}")

if __name__ == "__main__":
    INPUT_FILE = "product_info.xlsx"
    OUT_JSONL = "Submission/preprocessed_data/bank_data_advanced.jsonl"
    OUT_JSON = "Submission/preprocessed_data/bank_data_clean.json"
    
    records = process_bank_data_robust(INPUT_FILE)
    if records:
        save_output(records, OUT_JSONL, OUT_JSON)
