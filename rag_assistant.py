import os
import torch
import warnings
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import guardrails

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
DB_DIR     = "./vector_store_new"
LORA_PATH  = "/content/drive/MyDrive/colab_deploy/lora_bank_model_qwen"
BASE_MODEL = "Qwen/Qwen1.5-4B-Chat"

# ── Keyword → sheet code mapping ──────────────────────────────────────────
KEYWORD_TO_SHEET = {
    "waqaar":"NWA","nwa":"NWA",
    "little champs":"LCA","lca":"LCA",
    "asaan account":"NAA","nust asaan account":"NAA","naa":"NAA",
    "pakwatan":"PWRA","pwra":"PWRA",
    "roshan digital":"RDA","rda":"RDA",
    "value plus current":"VPCA","vpca":"VPCA",
    "value plus business":"VP-BA","vp-ba":"VP-BA",
    "value premium":"VPBA","vpba":"VPBA",
    "special deposit":"NSDA","nsda":"NSDA",
    "profit and loss":"PLS","pls":"PLS",
    "current deposit":"CDA","cda":"CDA",
    "maximiser":"NMA","nma":"NMA",
    "asaan digital account":"NADA","nada":"NADA",
    "asaan digital remittance":"NADRA","nadra account":"NADRA",
    "freelancer":"NFDA","nfda":"NFDA",
    "sahar account":"NSA","nsa":"NSA",
    "exporters":"ESFCA","esfca":"ESFCA",
    "auto finance":"NUST4Car","car finance":"NUST4Car","nust4car":"NUST4Car",
    "personal finance":"PF",
    "mastercard":"NMC","nmc":"NMC",
    "mortgage":"NMF","nmf":"NMF",
    "sahar finance":"NSF","nsf":"NSF",
    "imarat":"NIF","nif":"NIF",
    "ujala":"NUF","nuf":"NUF",
    "flour mill":"NFMF","nfmf":"NFMF",
    "fauri":"NFBF","nfbf":"NFBF",
    "prime minister":"PMYB &ALS","pmyb":"PMYB &ALS",
    "rice finance":"NRF","nrf":"NRF",
    "hunarmand":"NHF","nhf":"NHF",
    "nust life":"Nust Life",
    "efu life":"EFU Life",
    "jubilee":"Jubilee Life ",
    "home remittance":"HOME REMITTANCE","remittance":"HOME REMITTANCE",
}

PRODUCT_GLOSSARY = {
    "LCA":"Little Champs Account","NAA":"NUST Asaan Account","NWA":"NUST Waqaar Account",
    "PWRA":"PakWatan Remittance Account","RDA":"Roshan Digital Account","VPCA":"Value Plus Current Account",
    "VP-BA":"Value Plus Business Account","VPBA":"NUST Value Premium Business Account",
    "NSDA":"NUST Special Deposit Account","PLS":"Profit and Loss Sharing Account",
    "CDA":"Current Deposit Account","NMA":"NUST Maximiser Account",
    "NADA":"NUST Asaan Digital Account","NADRA":"NUST Asaan Digital Remittance Account",
    "NUST4Car":"NUST4Car Auto Finance","ESFCA":"Exporters Special Foreign Currency Account",
    "NFDA":"NUST Freelancer Digital Account","NSA":"NUST Sahar Accounts",
    "PF":"NUST Personal Finance","NMC":"NUST Bank Mastercard","NMF":"NUST Mortgage Finance",
    "NSF":"NUST Sahar Finance","NIF":"NUST Imarat Finance","NUF":"NUST Ujala Finance",
    "NFMF":"NUST Flour Mill Finance","NFBF":"NUST Fauri Business Finance",
    "PMYB&ALS":"Prime Minister Youth Business and Agriculture Loan Scheme",
    "NRF":"NUST Rice Finance","NHF":"NUST Hunarmand Finance",
    "HOME REMITTANCE":"Home Remittance Services",
}


class BankRAGAssistant:
    def __init__(self):
        print("Initializing Bank RAG Assistant...")

        # ── 1. Tokenizer ──────────────────────────────────────────────
        print(f"Loading tokenizer from: {LORA_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            LORA_PATH,
            trust_remote_code=True
        )

        # ── 2. Base model (4-bit) ─────────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading base model: {BASE_MODEL}")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # ── 3. LoRA adapter ───────────────────────────────────────────
        print(f"Applying LoRA adapter from: {LORA_PATH}")
        # Patch peft's repo ID validator to accept local absolute paths
        import huggingface_hub.utils._validators as _hf_validators
        _orig = _hf_validators.validate_repo_id
        _hf_validators.validate_repo_id = lambda *a, **kw: None
        try:
            self.model = PeftModel.from_pretrained(
                self.model,
                LORA_PATH,
                is_trainable=False,
                ignore_mismatched_sizes=True,
            )
        finally:
            _hf_validators.validate_repo_id = _orig

        self.model.eval()

        # ── 4. Vector DB ──────────────────────────────────────────────
        self.db_client  = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.db_client.get_collection(name="bank_knowledge_base")
        print(f"Vector DB loaded ({self.collection.count()} records)")

        # ── 5. Retrieval models ───────────────────────────────────────
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

        print("System Ready!")

    # ──────────────────────────────────────────────────────────────────
    #  Account Detection
    # ──────────────────────────────────────────────────────────────────

    def detect_account(self, query: str):
        """
        Returns the sheet code for the account mentioned in the query.
        Uses longest keyword match first, then falls back to raw sheet codes.
        """
        qlow = query.lower()
        matched_kw, matched_sheet = None, None
        for kw, sheet in KEYWORD_TO_SHEET.items():
            if kw in qlow:
                if matched_kw is None or len(kw) > len(matched_kw):
                    matched_kw, matched_sheet = kw, sheet
        if matched_sheet:
            print(f"[detect] '{matched_kw}' → {matched_sheet}")
            return matched_sheet

        # Fallback: raw sheet codes from DB metadata
        if not hasattr(self, '_sheet_codes'):
            self._sheet_codes = set()
            for m in self.collection.get(include=['metadatas'])['metadatas']:
                if isinstance(m, dict):
                    s = m.get('sheet')
                    if s: self._sheet_codes.add(s.lower())

        for code in self._sheet_codes:
            if code in qlow:
                print(f"[detect] sheet code '{code}'")
                return code.upper()

        print("[detect] No account matched — broad search")
        return None

    # ──────────────────────────────────────────────────────────────────
    #  Retrieval
    # ──────────────────────────────────────────────────────────────────

    def retrieve_context(self, query: str) -> str:
        """
        1. Filter by account sheet if detected (prevents cross-account contamination)
        2. Fetch top 15 chunks
        3. Rerank with CrossEncoder
        4. Return top 5 most relevant
        """
        account = self.detect_account(query)
        where   = {"sheet": account} if account else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=7,
                where=where,
            )
        except Exception:
            print("[retrieve] Filtered query failed, falling back to broad search")
            results = self.collection.query(query_texts=[query], n_results=15)

        docs = results['documents'][0]
        if not docs:
            return ""

        print(f"\n--- [DEBUG: {len(docs)} chunks | filter={where}] ---")
        for i, d in enumerate(docs):
            print(f"{i+1}. {d[:150]}...")
        print("---\n")

        # Rerank with CrossEncoder
        pairs  = [[query, doc] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return "\n\n---\n\n".join(ranked[:5])

    # ──────────────────────────────────────────────────────────────────
    #  Generation
    # ──────────────────────────────────────────────────────────────────

    def build_system_prompt(self, context: str) -> str:
        glossary_str = "\n".join(f'  "{k}": "{v}"' for k,v in PRODUCT_GLOSSARY.items())
        return (
            "You are the 'NUST Bank Elite Advisor'. You are warm, professional, "
            "and always format your responses beautifully using Markdown.\n\n"
            "Answer the user's questions from the bank records:"

            "### PRODUCT GLOSSARY (for expanding abbreviations):\n"
            "{\n" + glossary_str + "\n}\n\n"

            "### FORMATTING RULES — follow these exactly:\n"
            "- If the answer has multiple items or features → use bullet points (- item)\n"
            "- If the answer compares two variants (e.g. Savings vs Term Deposit) → use two labeled sections with bullets\n"
            "- If the answer contains a rate or financial figure → state it clearly (e.g. Profit Rate: 19.00% per annum)\n"
            "- If the answer is a single short fact → one clean sentence\n"
            "- Use **bold** for section headers and key terms\n"
            "- NEVER dump raw unformatted text\n"
            "- Start directly with the answer — no preamble like 'Based on the records...'\n\n"

            "### FORMATTING EXAMPLES:\n\n"
            "Example 1 — features list:\n"
            "User: What are the features of this account?\n"
            "Answer:\n"
            "Here are the key features:\n"
            "- Feature one with its detail\n"
            "- Feature two with its detail\n"
            "- **Profit Rate:** XX% per annum (paid monthly)\n"
            "- **Minimum Deposit:** Rs. XXX\n\n"

            "Example 2 — two-variant comparison:\n"
            "User: What are the salient features of this account?\n"
            "Answer:\n"
            "This account is available in two variants:\n\n"
            "**Savings Account**\n"
            "- Feature: value\n"
            "- **Profit Rate:** XX% per annum\n\n"
            "**Term Deposit**\n"
            "- Tenure: X year(s)\n"
            "- **Profit Rate:** XX% per annum\n\n"

            "Example 3 — single fact:\n"
            "User: What is the minimum age requirement?\n"
            "Answer:\n"
            "The minimum age to open this account is XX years.\n\n"

            "Example 4 — rate comparison table:\n"
            "User: Compare profit rates across accounts\n"
            "Answer:\n"
            "| Account | Profit Rate | Payout | Currency |\n"
            "|---------|-------------|--------|----------|\n"
            "| Account A | XX% | Monthly | PKR |\n"
            "| Account B | XX% | Quarterly | PKR |\n\n"

            "### BEHAVIOR:\n"
            "1. If the records do NOT contain relevant information, say: "
            "'I don't have that specific information in our records.'\n"
            "2. Do NOT invent or assume any information not present in the records.\n"
            "3. If asked non-banking questions, politely redirect to NUST Bank services.\n"
            "4. Keep tone warm and professional.\n\n"

            "### BANK RECORDS:\n"
            f"{context if context else 'No specific records found.'}"
        )

    def generate_answer_stream(self, query: str, history=None, context: str = ""):
        if history is None:
            history = []

        system_msg = self.build_system_prompt(context)
        messages   = [{"role": "system", "content": system_msg}]

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        if not history or history[-1].get("content") != query:
            messages.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            **inputs,
            "streamer":       streamer,
            "max_new_tokens": 1024,
            "temperature":    0.7,
            "top_p":          0.9,
            "do_sample":      True,
            "use_cache":      True,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    # ──────────────────────────────────────────────────────────────────
    #  Chat entry point
    # ──────────────────────────────────────────────────────────────────

    def chat(self, query=None, history=None):
        """
        Can be called two ways:
          1. chat(query, history) → returns (stream_generator, raw_docs) for UI use
          2. chat() → interactive CLI loop
        """
        if query:
            # Safety check
            if not guardrails.is_query_safe(query):
                return iter(["I specialize in NUST Bank financial services. How can I help you today?"]), []

            # Retrieve context with account filtering + reranking
            context = self.retrieve_context(query)

            # Build raw_docs for UI display
            results  = self.collection.query(query_texts=[query], n_results=5)
            raw_docs = []
            for i in range(len(results['documents'][0])):
                class Doc: pass
                d = Doc()
                d.page_content = results['documents'][0][i]
                d.metadata     = results['metadatas'][0][i]
                raw_docs.append(d)

            return self.generate_answer_stream(query, history, context), raw_docs

        # ── CLI mode ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(" NUST Bank AI Assistant — Active")
        print("=" * 60)
        print("Type 'exit' to quit.\n")

        history = []
        while True:
            query = input("👤 User: ").strip()
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            if not query:
                continue

            print("🔍 Searching knowledge base...", end="\r")
            context = self.retrieve_context(query)

            print("🤖 Thinking...               ", end="\r")
            print("\n🤖 Assistant: ", end="")
            full_response = ""
            for token in self.generate_answer_stream(query, history, context):
                print(token, end="", flush=True)
                full_response += token

            # Update history for multi-turn conversation
            history.append({"role": "user",      "content": query})
            history.append({"role": "assistant",  "content": full_response})

            # Keep history to last 6 turns to avoid context overflow
            if len(history) > 12:
                history = history[-12:]

            print("\n" + "-" * 60)


if __name__ == "__main__":
    assistant = BankRAGAssistant()
    assistant.chat()