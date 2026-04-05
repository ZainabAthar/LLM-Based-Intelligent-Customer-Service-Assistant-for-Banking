"""
preprocessing_new.py — NUST Bank Data Preprocessing
Rebuilt from scratch after reading the actual Excel structure.

HOW THE EXCEL IS STRUCTURED (observed across all 33 content sheets):
  - Row 0:      Sheet title (skip)
  - "Main" cell: navigation marker (skip)
  - Question rows: single cell containing a question string
  - Answer rows:  one or more rows — collect ALL non-noise cells from ALL columns
  - Multi-column rows: some sheets put parallel variants side by side
                       e.g. "New Business: X | Existing Business: Y"
  - "#REF!" cells: broken Excel refs — skip
  - Rate table headers: "Profit Payment", "Profit Rate", "Tenor" etc. — skip
"""

import pandas as pd
import re
import json
import hashlib
from pathlib import Path

# ─────────────────────────────────────────────
EXCEL_PATH   = "product_info.xlsx"
OUTPUT_JSONL = "preprocessed_data/bank_data_master.jsonl"

SKIP_SHEETS = {"Main", "Rate Sheet July 1 2024", "Sheet1"}

ACCOUNT_NAME_MAP = {
    "LCA":           "Little Champs Account",
    "NAA":           "NUST Asaan Account",
    "NWA":           "NUST Waqaar Account",
    "PWRA":          "PakWatan Remittance Account",
    "RDA":           "Roshan Digital Account",
    "VPCA":          "Value Plus Current Account",
    "VP-BA":         "Value Plus Business Account",
    "VPBA":          "NUST Value Premium Business Account",
    "NSDA":          "NUST Special Deposit Account",
    "PLS":           "Profit and Loss Sharing Account",
    "CDA":           "Current Deposit Account",
    "NMA":           "NUST Maximiser Account",
    "NADA":          "NUST Asaan Digital Account",
    "NADRA":         "NUST Asaan Digital Remittance Account",
    "NUST4Car":      "NUST4Car Auto Finance",
    "ESFCA":         "Exporters Special Foreign Currency Account",
    "NFDA":          "NUST Freelancer Digital Account",
    "NSA":           "NUST Sahar Accounts",
    "PF":            "NUST Personal Finance",
    "NMC":           "NUST Bank Mastercard",
    "NMF":           "NUST Mortgage Finance",
    "NSF":           "NUST Sahar Finance",
    "NIF":           "NUST Imarat Finance",
    "NUF":           "NUST Ujala Finance",
    "NFMF":          "NUST Flour Mill Finance",
    "NFBF":          "NUST Fauri Business Finance",
    "PMYB &ALS":     "Prime Minister Youth Business & Agriculture Loan Scheme",
    "NRF":           "NUST Rice Finance",
    "NHF":           "NUST Hunarmand Finance",
    "Nust Life":     "NUST Life Bancassurance Policy",
    "EFU Life":      "EFU Life Bancassurance Policy",
    "Jubilee Life ": "Jubilee Life Bancassurance Policy",
    "HOME REMITTANCE": "Home Remittance Services",
}

PII_PATTERNS = {
    "CNIC":           r'\b\d{5}-\d{7}-\d\b',
    "PHONE":          r'(\+92|0)3\d{2}-\d{7}|\b0\d{2}-\d{7,8}\b',
    "IBAN":           r'\bPK\d{2}[A-Z]{4}[A-Z0-9]{16}\b',
}

# Table header cells that are not real answer content
NOISE_TOKENS = {
    "main", "profit payment", "profit rate", "payout", "tenor", "tenure",
    "savings", "term deposit", "currency", "pkr", "usd", "gbp", "eur",
    "* for current account only", "for current account only",
    "#ref!", "nan", "",
}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def clean(val) -> str:
    if val is None:
        return ""
    s = str(val).replace("\xa0", " ").replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_noise(text: str) -> bool:
    t = text.lower().strip().rstrip("*").strip()
    return t in NOISE_TOKENS


def is_question(text: str) -> bool:
    t = text.strip()
    if len(t) < 8:
        return False
    if t.endswith("?"):
        return True
    low = t.lower()
    q_starters = (
        "what", "how", "why", "when", "where", "can i", "can the", "can a",
        "is ", "are ", "does", "do ", "which", "who ", "will ", "was ",
        "in which", "for which", "any ", "is there", "are there",
        "does the", "does nust",
    )
    return any(low.startswith(s) for s in q_starters)


def anonymize(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"<{label}_MASKED>", text)
    return text


def make_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_answer_lines(raw: str) -> str:
    """Normalize bullet chars and extra whitespace in answer text."""
    lines = raw.split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Normalize various bullet chars to dash
        line = re.sub(r"^[·•o]\s+", "- ", line)
        # Clean up numbered list noise like "1.\xa0\xa0 text" → "1. text"
        line = re.sub(r"^(\d+)\.\s+", r"\1. ", line)
        out.append(line)
    # Collapse 3+ blank lines
    result = "\n".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ─────────────────────────────────────────────
#  Core extractor
# ─────────────────────────────────────────────

def extract_qa(df: pd.DataFrame, account_name: str) -> list:
    """
    Walk the dataframe row by row.
    Question detected → start collecting answer rows until next question.
    All non-noise cells in each answer row are collected.
    Multi-column answer rows are joined with " | " to preserve parallel structure.
    """
    qa_pairs   = []
    current_q  = None
    ans_parts  = []

    def flush():
        if current_q and ans_parts:
            full_ans = "\n".join(ans_parts).strip()
            if len(full_ans) > 4:
                qa_pairs.append((current_q, full_ans))

    for _, row in df.iterrows():
        cells = []
        for val in row:
            c = clean(val)
            if c and not is_noise(c) and c != account_name:
                cells.append(c)

        if not cells:
            continue

        first = cells[0]

        # Skip "Main" navigation markers
        if first.lower() == "main":
            continue

        if is_question(first):
            flush()
            current_q = first
            ans_parts = []
            # If there are extra cells on the same row as the question,
            # they are usually table headers (e.g. "Savings | Term Deposit") — skip
        else:
            if current_q is None:
                continue  # preamble before first question

            if len(cells) == 1:
                ans_parts.append(cells[0])
            else:
                # Multi-cell row: join with " | " to preserve parallel data
                ans_parts.append(" | ".join(cells))

    flush()
    return qa_pairs


# ─────────────────────────────────────────────
#  Rate sheet extractor
# ─────────────────────────────────────────────

def extract_rate_sheet(xl: pd.ExcelFile) -> list:
    records = []
    try:
        df = xl.parse(1, header=None)
    except Exception as e:
        print(f"  Warning: Could not parse rate sheet: {e}")
        return records

    data = df.values

    def fmt_pct(v):
        try:
            f = float(v)
            if f == 0:
                return None
            return f"{f * 100:.2f}%" if f < 1.0 else f"{f:.2f}%"
        except:
            return None

    curr_sav  = None
    curr_term = None

    for row in data:
        row_s = [clean(v) for v in row]

        # LEFT side: savings (col 1 = name/payout, col 3 = rate)
        lname = row_s[1] if len(row_s) > 1 else ""
        lrate = row_s[3] if len(row_s) > 3 else ""

        if lname and not is_noise(lname):
            if any(k in lname for k in ("Account", "Savings", "LCA", "Pensioners", "Bachat", "PLS")):
                if "Profit" not in lname and "Rate" not in lname:
                    curr_sav = lname
            r = fmt_pct(lrate)
            if curr_sav and r and lname not in (curr_sav, "") and not is_noise(lname):
                q = f"What is the profit rate for {curr_sav}?"
                a = f"The {curr_sav} offers a profit rate of **{r}**, with profit paid **{lname}**."
                text = f"[Account: {curr_sav}]\nQuestion: {q}\nAnswer: {a}"
                records.append({
                    "hash_id": make_hash(text),
                    "metadata": {"source": EXCEL_PATH, "sheet": "Rate Sheet",
                                 "account_name": curr_sav, "question": q},
                    "content": text,
                    "structured_data": {"question": q, "answer": a, "account": curr_sav},
                })

        # RIGHT side: term deposits (col 5 = tenor, col 6 = payout, col 8 = rate)
        tname   = row_s[5] if len(row_s) > 5 else ""
        tpayout = row_s[6] if len(row_s) > 6 else ""
        trate   = row_s[8] if len(row_s) > 8 else ""

        if tname and not is_noise(tname):
            if any(k in tname for k in ("SNDR", "Term Deposit", "Maximiser", "Bachat", "Waqaar", "Sahar")):
                if tname not in ("Tenor", "Payout"):
                    curr_term = tname
            r = fmt_pct(trate)
            if curr_term and r and tname not in (curr_term, "Tenor", "Payout", "Profit Rate", "") and not is_noise(tname):
                payout_str = f" with {tpayout} payout" if tpayout not in ("", "nan") else ""
                q = f"What is the profit rate for {curr_term} with {tname} tenor?"
                a = f"The profit rate for **{curr_term}** ({tname} tenor{payout_str}) is **{r}**."
                text = f"[Account: {curr_term}]\nQuestion: {q}\nAnswer: {a}"
                records.append({
                    "hash_id": make_hash(text),
                    "metadata": {"source": EXCEL_PATH, "sheet": "Rate Sheet",
                                 "account_name": curr_term, "question": q},
                    "content": text,
                    "structured_data": {"question": q, "answer": a, "account": curr_term},
                })

    print(f"  Rate sheet: {len(records)} rate records")
    return records


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def process_excel():
    xl          = pd.ExcelFile(EXCEL_PATH)
    all_records = []
    seen_hashes = set()
    total_bad   = 0

    for sheet in xl.sheet_names:
        if sheet.strip() in SKIP_SHEETS:
            print(f"Skipping:    {sheet}")
            continue

        account_name = ACCOUNT_NAME_MAP.get(sheet, sheet.strip())
        print(f"Processing:  {sheet:<20} → {account_name}")

        df      = xl.parse(sheet, header=None)
        qa_list = extract_qa(df, account_name)

        if not qa_list:
            print(f"  No Q&A pairs found")
            continue

        count = 0
        for q, raw_a in qa_list:
            a = clean_answer_lines(raw_a)

            # Skip records where answer is suspiciously short and looks like a mis-parse
            if len(a) < 8:
                total_bad += 1
                print(f"  SKIP short answer: Q={q[:60]} | A={a}")
                continue

            content = anonymize(
                f"[Account: {account_name}]\nQuestion: {q}\nAnswer: {a}"
            )
            h = make_hash(content)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            all_records.append({
                "hash_id": h,
                "metadata": {
                    "source":       EXCEL_PATH,
                    "sheet":        sheet,
                    "account_name": account_name,
                    "question":     q,
                },
                "content": content,
                "structured_data": {
                    "question": q,
                    "answer":   a,
                    "account":  account_name,
                },
            })
            count += 1
        print(f"  → {count} records")

    # Rate sheet
    print("\nProcessing rate sheet...")
    for rec in extract_rate_sheet(xl):
        if rec["hash_id"] not in seen_hashes:
            all_records.append(rec)
            seen_hashes.add(rec["hash_id"])

    # Save
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"   Total records saved : {len(all_records)}")
    print(f"   Bad records skipped : {total_bad}")
    print(f"   Output file         : {OUTPUT_JSONL}")


if __name__ == "__main__":
    process_excel()