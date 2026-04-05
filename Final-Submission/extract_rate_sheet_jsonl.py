import pandas as pd
import json
import hashlib
import os

# Precise Column Mapping from JSON Dump
# Savings: Payout=Index 1, Rate=Index 3
# Term: Tenor=Index 5, Payout=Index 6, Rate=Index 8

EXCEL_FILE = "product_info.xlsx"
OUTPUT_JSONL = "preprocessed_data/rate_sheet_only.jsonl"
SHEET_NAME_INDEX = 1 

def get_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def format_as_percent(val):
    if pd.isna(val) or val == "null" or val == None: return None
    try:
        f_val = float(val)
        return f"{f_val * 100:.2f}%" if f_val < 1.0 else f"{f_val:.2f}%"
    except:
        return None

def extract():
    print(f"Reading Excel Sheet Index {SHEET_NAME_INDEX}...")
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME_INDEX, header=None)
    data = df.values
    
    records = []
    curr_sav = None
    curr_term = None
    
    for i in range(len(data)):
        row = data[i]
        
        # --- LEFT SIDE: SAVINGS (Indices 1 and 3) ---
        # Note: Name and Payout share the same column (1)
        left_val = str(row[1]).strip() if len(row) > 1 and not pd.isna(row[1]) else None
        left_rate = row[3] if len(row) > 3 else None
        
        if left_val:
            # Header detection
            if "Account" in left_val or "Savings" in left_val or "LCA" in left_val or "Pensioners" in left_val:
                if "Profit Payment" not in left_val:
                    curr_sav = left_val
            
            # Data detection
            rate_s = format_as_percent(left_rate)
            if curr_sav and rate_s and left_val not in ["Profit Payment", curr_sav]:
                q = f"What is the profit rate and frequency for {curr_sav}?"
                a = f"The {curr_sav} offers a profit rate of {rate_s}, with profit paid {left_val}."
                records.append({"acc": curr_sav, "q": q, "a": a})

        # --- RIGHT SIDE: TERM DEPOSITS (Indices 5, 6, 8) ---
        term_val = str(row[5]).strip() if len(row) > 5 and not pd.isna(row[5]) else None
        term_payout = str(row[6]).strip() if len(row) > 6 and not pd.isna(row[6]) else None
        term_rate = row[8] if len(row) > 8 else None
        
        if term_val:
            # Header detection
            if "SNDR" in term_val or "Term Deposit" in term_val or "Maximiser" in term_val:
                if term_val not in ["Tenor", "Payout"]:
                    curr_term = term_val
            
            # Data detection
            rate_t = format_as_percent(term_rate)
            if curr_term and rate_t and term_val not in ["Tenor", "Payout", "Profit Rate", curr_term]:
                q = f"What is the profit rate for {curr_term} ({term_val} tenor)?"
                pay_info = f" with {term_payout} payout" if term_payout != "nan" else ""
                a = f"The profit rate for {curr_term} with a {term_val} tenor{pay_info} is {rate_t}."
                records.append({"acc": curr_term, "q": q, "a": a})

    # Export to JSONL
    final_output = []
    for r in records:
        text = f"[Account: {r['acc']}]\nQuestion: {r['q']}\nAnswer: {r['a']}"
        final_output.append({
            "hash_id": get_hash(text),
            "metadata": {
                "source": os.path.basename(EXCEL_FILE),
                "account_name": r["acc"],
                "question": r["q"]
            },
            "content": text,
            "structured_data": {"question": r["q"], "answer": r["a"], "account": r["acc"]}
        })

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in final_output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"SUCCESS: Extracted {len(final_output)} records to {OUTPUT_JSONL}")

if __name__ == "__main__":
    extract()
