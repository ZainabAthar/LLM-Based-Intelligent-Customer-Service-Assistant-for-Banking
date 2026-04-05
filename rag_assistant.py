import os
import torch
import warnings
import chromadb
import logging
import uuid
import chromadb.utils.embedding_functions as ef
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import guardrails

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
#  Configuration
#  BASE_MODEL must match exactly what finetune.py trained on.
# ─────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL     = "Qwen/Qwen2.5-3B-Instruct"           # same as finetune.py
DEFAULT_LORA   = os.path.join(SCRIPT_DIR, "lora_bank_model_qwenx")
DEFAULT_DB     = os.path.join(SCRIPT_DIR, "vector_store_new")
COLLECTION     = "bank_knowledge_base"

PRODUCT_GLOSSARY = {
    "LCA":"Little Champs Account","NAA":"NUST Asaan Account","NWA":"NUST Waqaar Account",
    "PWRA":"PakWatan Remittance Account","RDA":"Roshan Digital Account",
    "VPCA":"Value Plus Current Account","VP-BA":"Value Plus Business Account",
    "VPBA":"NUST Value Premium Business Account","NSDA":"NUST Special Deposit Account",
    "PLS":"Profit and Loss Sharing Account","CDA":"Current Deposit Account",
    "NMA":"NUST Maximiser Account","NADA":"NUST Asaan Digital Account",
    "NADRA":"NUST Asaan Digital Remittance Account","NUST4Car":"NUST4Car Auto Finance",
    "ESFCA":"Exporters Special Foreign Currency Account","NFDA":"NUST Freelancer Digital Account",
    "NSA":"NUST Sahar Accounts","PF":"NUST Personal Finance","NMC":"NUST Bank Mastercard",
    "NMF":"NUST Mortgage Finance","NSF":"NUST Sahar Finance","NIF":"NUST Imarat Finance",
    "NUF":"NUST Ujala Finance","NFMF":"NUST Flour Mill Finance","NFBF":"NUST Fauri Business Finance",
    "NRF":"NUST Rice Finance","NHF":"NUST Hunarmand Finance",
    "HOME REMITTANCE":"Home Remittance Services",
}

KEYWORD_TO_SHEET = {
    "waqaar":"NWA","nwa":"NWA","little champs":"LCA","lca":"LCA",
    "asaan account":"NAA","naa":"NAA","pakwatan":"PWRA","pwra":"PWRA",
    "roshan digital":"RDA","rda":"RDA","value plus current":"VPCA","vpca":"VPCA",
    "value plus business":"VP-BA","value premium":"VPBA","vpba":"VPBA",
    "special deposit":"NSDA","nsda":"NSDA","profit and loss":"PLS","pls":"PLS",
    "current deposit":"CDA","cda":"CDA","maximiser":"NMA","nma":"NMA",
    "asaan digital account":"NADA","nada":"NADA","asaan digital remittance":"NADRA",
    "freelancer":"NFDA","nfda":"NFDA","sahar account":"NSA","nsa":"NSA",
    "exporters":"ESFCA","esfca":"ESFCA","auto finance":"NUST4Car","car finance":"NUST4Car",
    "personal finance":"PF","mastercard":"NMC","nmc":"NMC","mortgage":"NMF","nmf":"NMF",
    "sahar finance":"NSF","imarat":"NIF","ujala":"NUF","flour mill":"NFMF",
    "fauri":"NFBF","rice finance":"NRF","hunarmand":"NHF","nust life":"Nust Life",
    "efu life":"EFU Life","jubilee":"Jubilee Life","remittance":"HOME REMITTANCE",
    # Topic detection for broad banking functions
    "limit":"Funds Transfer / RAAST","transfer":"Funds Transfer / RAAST",
    "raast":"Funds Transfer / RAAST","ibft":"Funds Transfer / RAAST",
    "bill":"App Features / Functionalities","utilit":"App Features / Functionalities",
    "biometric":"App Features / Functionalities","mpin":"App Features / Functionalities",
    "atm":"ATM","debit card":"ATM","card":"ATM",
}


class BankRAGAssistant:
    def __init__(self, lora_path=None, db_dir=None):
        self.lora_path = lora_path or os.getenv("LORA_PATH", DEFAULT_LORA)
        self.db_dir    = db_dir    or os.getenv("DB_DIR",    DEFAULT_DB)

        print("Initializing NUST Bank RAG Assistant...")
        print(f"  Base model : {BASE_MODEL}")
        print(f"  LoRA path  : {self.lora_path}")
        print(f"  DB dir     : {self.db_dir}")

        # ── 1. Tokenizer ──────────────────────────────────────────────
        # Try loading from the saved LoRA dir first (preserves any added tokens),
        # fall back to the HF hub copy if not found locally.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.lora_path, trust_remote_code=True,
                padding_side="left", local_files_only=True
            )
            print("  Tokenizer loaded from LoRA directory.")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL, trust_remote_code=True, padding_side="left"
            )
            print("  Tokenizer loaded from HuggingFace hub.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # ── 2. Base model (4-bit quantised) ───────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        base_model.config.use_cache = True

        # ── 3. LoRA adapter via PeftModel ─────────────────────────────
        # FIX: use PeftModel.from_pretrained — validated, safe, correct.
        print(f"  Applying LoRA adapter...")
        import huggingface_hub.utils._validators as _hf_v
        _orig = _hf_v.validate_repo_id
        _hf_v.validate_repo_id = lambda *a, **kw: None  # allow local absolute paths
        try:
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                is_trainable=False,
            )
        finally:
            _hf_v.validate_repo_id = _orig

        self.model.eval()
        print("  LoRA adapter applied.")

        # ── 4. Vector DB ──────────────────────────────────────────────
        self.db_client = chromadb.PersistentClient(path=self.db_dir)
        
        # Consistent embedding function (MUST match vector_db_setup.py)
        self.emb_fn = ef.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.collection = self.db_client.get_collection(
            name=COLLECTION, 
            embedding_function=self.emb_fn
        )
        print(f"  Vector DB: {self.collection.count()} records loaded.")

        # ── 5. Retrieval models ───────────────────────────────────────
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

        print("\nSystem Ready! Type your question.\n")

    # ──────────────────────────────────────────────────────────────────
    #  Document Ingestion (for user-uploaded files)
    # ──────────────────────────────────────────────────────────────────
    def ingest_records(self, records: list, session_id: str = "default"):
        """
        Adds parsed records to the ChromaDB collection tagged with session_id.
        """
        if not records:
            return 0

        ids      = []
        metas    = []
        docs     = []
        embeddings = []

        for rec in records:
            # unique ID to avoid collisions across sessions if same doc uploaded
            rid = f"{session_id}_{rec['hash_id']}"
            ids.append(rid)
            
            # Ensure metadata has the session_id and sheet type
            meta = rec.get("metadata", {})
            meta["session_id"] = session_id
            meta["sheet"]      = "Uploaded"
            metas.append(meta)
            
            docs.append(rec["content"])

        # Let Chroma handle embeddings internally using self.emb_fn
        self.collection.add(
            ids=ids,
            metadatas=metas,
            documents=docs
        )
        print(f"  Ingested {len(records)} segments for session: {session_id}")
        return len(records)

    def clear_session_records(self, session_id: str):
        """Removes all 'Uploaded' records for a specific session."""
        try:
            self.collection.delete(where={"session_id": session_id})
            print(f"  Cleared records for session: {session_id}")
            return True
        except Exception as e:
            print(f"  Error clearing session records: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────
    #  Account Detection (filters retrieval to avoid cross-product noise)
    # ──────────────────────────────────────────────────────────────────
    def _detect_account(self, query: str):
        qlow = query.lower()
        best_kw, best_sheet = None, None
        for kw, sheet in KEYWORD_TO_SHEET.items():
            if kw in qlow:
                if best_kw is None or len(kw) > len(best_kw):
                    best_kw, best_sheet = kw, sheet
        return best_sheet

    # ──────────────────────────────────────────────────────────────────
    #  Retrieval — account-filtered + cross-encoder reranked
    # ──────────────────────────────────────────────────────────────────
    def retrieve_context(self, query: str, session_id: str = "default"):
        account = self._detect_account(query)
        
        # Hybrid Filter: Search detected account OR the uploaded document chunks for this session
        where = None
        if account:
            where = {
                "$or": [
                    {"sheet": account},
                    {"$and": [{"sheet": "Uploaded"}, {"session_id": session_id}]}
                ]
            }
        else:
            # No specific account detected, just search general + uploaded for this session
            where = {"$or": [
                {"sheet": {"$ne": "Uploaded"}}, # Search all official sheets
                {"session_id": session_id}       # AND uploaded for this session
            ]}

        try:
            results = self.collection.query(
                query_texts=[query], n_results=16, where=where
            )
        except Exception:
            # Fallback if complex filter fails
            results = self.collection.query(query_texts=[query], n_results=10)

        docs  = results["documents"][0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return "", []

        # Rerank with CrossEncoder
        scores = self.reranker.predict([[query, d] for d in docs])
        ranked = sorted(zip(scores, docs, metas), reverse=True)
        top_n  = ranked[:10]   # Pass top 10 chunks to ensure values are found

        context = "\n\n---\n\n".join(doc for _, doc, _ in top_n)
        return context, top_n[:5]

    # ──────────────────────────────────────────────────────────────────
    #  System Prompt
    # ──────────────────────────────────────────────────────────────────
    def _build_system_prompt(self, context: str) -> str:
        return (
            "You are the NUST Bank Advisor. Professional, helpful, and concise.\n\n"

            "### MANDATORY INSTRUCTIONS:\n"
            "1. LANGUAGE: ALWAYS respond in English (British/Pakistani English). NEVER use Turkish, Urdu, or any other language.\n"
            "2. SCOPE: Answer ONLY about NUST Bank (products, fees, rates, docs).\n"
            "3. OFF-TOPIC: If the user asks about something else, say: 'I'm NUST Bank's AI Assistant and can only help with banking-related queries.'\n"
            "4. NO HALLUCINATION: Only use the records provided below.\n"
            "5. PERCENTAGES (CRITICAL): You MUST multiply decimals by 100 to show the correct rate. "
            "If the record shows '0.19', your response MUST say '19.00%'. If it shows '0.08', you MUST say '8.00%'. "
            "NEVER display raw decimals like '0.19%' or '0.08%'. Always use two decimal places (e.g., 17.50%).\n"
            "6. CURRENCY: Always PKR (Rs.).\n\n"

            "### BANK RECORDS:\n"
            "Examine exactly to find the correct figures (interest rates, limits, dates).\n"
            f"{context if context else 'No specific records found. Answer from general knowledge only.'}\n\n"

            "NOTE: Some records are 'OFFICIAL' while others are 'UPLOADED' by the user. "
            "Stick to the specific account or document requested."
        )

    # ──────────────────────────────────────────────────────────────────
    #  Generation (streaming)
    # ──────────────────────────────────────────────────────────────────
    def generate_answer_stream(self, query: str, history=None, context: str = ""):
        if history is None:
            history = []

        messages = [{"role": "system", "content": self._build_system_prompt(context)}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3072
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer":          streamer,
            "max_new_tokens":    768,
            # FIX: use sampling with temperature — greedy (temp=0) causes repetition loops
            "do_sample":         True,
            "temperature":       0.4,     # low enough for factual accuracy, high enough to avoid loops
            "top_p":             0.9,
            "repetition_penalty": 1.15,   # prevents the model repeating the same phrase
            "use_cache":         True,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

    # ──────────────────────────────────────────────────────────────────
    #  Chat entry point
    # ──────────────────────────────────────────────────────────────────
    def chat(self, query=None, history=None):
        """
        UI mode:  chat(query, history) → returns (stream_generator, top_docs)
        CLI mode: chat()               → interactive terminal loop
        """
        if query is not None:
            # session_id logic if passed in query (for UI mode)
            session_id = "default"
            if isinstance(query, dict):
                session_id = query.get("session_id", "default")
                query_text = query.get("prompt", "")
            else:
                query_text = query

            if not guardrails.is_query_safe(query_text):
                return iter(["I specialize in NUST Bank financial services. How can I assist you today?"]), []

            context, top_docs = self.retrieve_context(query_text, session_id=session_id)
            return self.generate_answer_stream(query_text, history or [], context), top_docs

        # ── CLI / Terminal mode ───────────────────────────────────────
        print("=" * 60)
        print("  NUST Bank AI Assistant — Terminal Mode")
        print("  Type 'exit' to quit.")
        print("=" * 60 + "\n")

        history = []
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", "bye"):
                print("Goodbye!")
                break

            if not guardrails.is_query_safe(query):
                print("Assistant: I specialize in NUST Bank financial services. How can I assist you?\n")
                continue

            print("Searching knowledge base...", end="\r")
            context, _ = self.retrieve_context(query)

            print("Assistant: ", end="", flush=True)
            full_response = ""
            for token in self.generate_answer_stream(query, history, context):
                print(token, end="", flush=True)
                full_response += token
            print("\n")

            # FIX: update history so multi-turn conversation works
            history.append({"role": "user",      "content": query})
            history.append({"role": "assistant",  "content": full_response})

            # Keep last 6 turns (12 messages) to avoid context overflow
            if len(history) > 12:
                history = history[-12:]

            print("-" * 60)


if __name__ == "__main__":
    assistant = BankRAGAssistant()
    assistant.chat()