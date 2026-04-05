"""
document_parser.py — NUST Bank Document Parser

Converts uploaded files (PDF, DOCX, TXT, plain text) into the
bank_data_master.jsonl schema so they can be ingested via the
/ingest endpoint and become instantly searchable.

Supported input formats:
  - .pdf   → extracted via pdfplumber
  - .docx  → extracted via python-docx
  - .txt   → read directly
  - .json  → assumed already in bank schema, passed through

Usage (standalone):
  python document_parser.py --file my_faq.pdf --account "New Savings Account"

Usage (as a module — called by api_service.py):
  from document_parser import parse_file_to_records
  records = parse_file_to_records(file_bytes, filename, account_name)
"""

import re
import json
import hashlib
import argparse
from pathlib import Path
from io import BytesIO


# ─────────────────────────────────────────────
#  Install check helpers
# ─────────────────────────────────────────────

def _pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")


def _docx_lib():
    try:
        import docx
        return docx
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")


# ─────────────────────────────────────────────
#  Text extractors
# ─────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdfplumber = _pdfplumber()
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t.strip())
    return "\n\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    docx = _docx_lib()
    doc = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1").strip()


# ─────────────────────────────────────────────
#  Text → Q&A chunker
# ─────────────────────────────────────────────

QUESTION_STARTERS = (
    "what", "how", "why", "when", "where", "can i", "can the", "can a",
    "is ", "are ", "does", "do ", "which", "who ", "will ", "was ",
    "in which", "for which", "any ", "is there", "are there",
    "does the", "does nust",
)


def _is_question(line: str) -> bool:
    t = line.strip()
    if len(t) < 8:
        return False
    if t.endswith("?"):
        return True
    low = t.lower()
    return any(low.startswith(s) for s in QUESTION_STARTERS)


def _make_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(text: str) -> str:
    text = re.sub(r"\xa0|\u200b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text_into_qa(raw_text: str, account_name: str) -> list:
    """
    Strategy 1 — Q&A detection:
      If the text contains explicit question lines (ending with ? or
      starting with question words), pair each question with the
      paragraphs that follow it until the next question.

    Strategy 2 — paragraph chunking (fallback):
      If no questions are detected, split into paragraphs and treat
      every paragraph as a standalone answer with a generated question.
    """
    lines = [_clean(l) for l in raw_text.splitlines() if _clean(l)]

    # ── Strategy 1: Q&A detection ──────────────────────────────────
    qa_pairs = []
    current_q = None
    ans_lines = []

    def flush():
        if current_q and ans_lines:
            answer = " ".join(ans_lines).strip()
            if len(answer) > 20:
                qa_pairs.append((current_q, answer))

    for line in lines:
        if _is_question(line):
            flush()
            current_q = line
            ans_lines = []
        else:
            if current_q:
                ans_lines.append(line)
    flush()

    if qa_pairs:
        return qa_pairs

    # ── Strategy 2: paragraph chunking ────────────────────────────
    # Group lines into paragraphs (split on blank lines)
    paragraphs = []
    current_para = []
    for line in lines:
        if line:
            current_para.append(line)
        else:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))

    # Turn each paragraph into a Q&A
    for i, para in enumerate(paragraphs):
        if len(para) < 30:
            continue
        # Generate a simple question from the first sentence
        first_sentence = re.split(r"[.!]", para)[0].strip()
        if len(first_sentence) > 15:
            q = f"What does {account_name} say about: {first_sentence[:80]}?"
        else:
            q = f"What is the information in section {i + 1} about {account_name}?"
        qa_pairs.append((q, para))

    return qa_pairs


# ─────────────────────────────────────────────
#  Main public function
# ─────────────────────────────────────────────

def parse_file_to_records(
    file_bytes: bytes,
    filename: str,
    account_name: str,
) -> list:
    """
    Parse a file into a list of bank_data_master.jsonl-compatible records.

    Args:
        file_bytes:   Raw bytes of the uploaded file.
        filename:     Original filename (used to detect extension).
        account_name: The product/account name to tag records with.

    Returns:
        List of dicts, each matching the bank record schema.
    """
    ext = Path(filename).suffix.lower()

    # ── Extract raw text ───────────────────────────────────────────
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        raw_text = extract_text_from_docx(file_bytes)
    elif ext in (".txt", ".md"):
        raw_text = extract_text_from_txt(file_bytes)
    elif ext in (".json", ".jsonl"):
        # Already structured — try to pass through directly
        return _passthrough_jsonl(file_bytes, filename)
    else:
        # Try plain text as fallback
        raw_text = extract_text_from_txt(file_bytes)

    if not raw_text.strip():
        return []

    # ── Chunk into Q&A pairs ───────────────────────────────────────
    qa_pairs = chunk_text_into_qa(raw_text, account_name)

    if not qa_pairs:
        return []

    # ── Build records ──────────────────────────────────────────────
    records = []
    seen_hashes = set()

    for question, answer in qa_pairs:
        content = f"[Account: {account_name}]\nQuestion: {question}\nAnswer: {answer}"
        hash_id = _make_hash(content)

        if hash_id in seen_hashes:
            continue
        seen_hashes.add(hash_id)

        records.append({
            "hash_id": hash_id,
            "metadata": {
                "source":       filename,
                "sheet":        "Uploaded",
                "account_name": account_name,
                "question":     question,
            },
            "content": content,
            "structured_data": {
                "question": question,
                "answer":   answer,
                "account":  account_name,
            },
        })

    return records


def _passthrough_jsonl(file_bytes: bytes, filename: str) -> list:
    """
    For .json/.jsonl uploads: validate each line has required fields
    and pass through valid records unchanged.
    """
    records = []
    text = file_bytes.decode("utf-8")

    # Handle both JSON array and JSONL (one object per line)
    text = text.strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
            lines = [json.dumps(item) for item in items]
        except Exception:
            lines = text.splitlines()
    else:
        lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # Auto-generate hash_id if missing
            if "hash_id" not in rec:
                content = rec.get("content", json.dumps(rec))
                rec["hash_id"] = _make_hash(content)
            # Ensure metadata exists
            if "metadata" not in rec:
                rec["metadata"] = {
                    "source": filename,
                    "sheet": "Uploaded",
                    "account_name": rec.get("structured_data", {}).get("account", "Unknown"),
                    "question": rec.get("structured_data", {}).get("question", ""),
                }
            if "content" not in rec:
                sd = rec.get("structured_data", {})
                rec["content"] = (
                    f"[Account: {rec['metadata'].get('account_name', 'Unknown')}]\n"
                    f"Question: {sd.get('question', '')}\n"
                    f"Answer: {sd.get('answer', '')}"
                )
            records.append(rec)
        except json.JSONDecodeError:
            continue

    return records


# ─────────────────────────────────────────────
#  Standalone CLI usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a document into bank JSONL format.")
    parser.add_argument("--file",    required=True,  help="Path to input file (PDF/DOCX/TXT/JSONL)")
    parser.add_argument("--account", required=True,  help="Account/product name to tag records with")
    parser.add_argument("--output",  default=None,   help="Output .jsonl path (default: prints to stdout)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}")
        exit(1)

    file_bytes = path.read_bytes()
    records = parse_file_to_records(file_bytes, path.name, args.account)

    if not records:
        print("No records extracted. Check that your file has readable text.")
        exit(1)

    print(f"Extracted {len(records)} records from '{path.name}'", flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved to: {out_path}")
    else:
        for rec in records:
            print(json.dumps(rec, ensure_ascii=False))