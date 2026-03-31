import os
import json
import warnings
import chromadb
from chromadb.utils import embedding_functions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY    = "/content/llm_project/LLM-Based-Intelligent-Customer-Service-Assistant-for-Banking/vector_store_new"
COLLECTION_NAME      = "bank_knowledge_base"

BASE_MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
# Replace with your Hugging Face Hub repository ID (e.g., "your-username/lora-bank-model")
LORA_HUB_ID = "navaal/qwen_lora_bank"
LORA_SUBFOLDER = "lora_bank_model/content/lora_bank_model/checkpoint-54"

import guardrails


class BankRAGAssistant:
    def __init__(self):
        print("Initializing Bank RAG Assistant...")

        # ── 1. Vector DB ──────────────────────────────────────────────
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

        # ── 2. Tokenizer ──────────────────────────────────────────────
        print(f"Loading tokenizer from base model: {BASE_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True
        )

        # ── 3. Model (base + LoRA adapter from Hub) ────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Loading base model: {BASE_MODEL_NAME}...")

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",          # handles device placement automatically
            )

            print(f"Applying LoRA adapter from Hugging Face Hub: {LORA_HUB_ID}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                LORA_HUB_ID,
                is_trainable=False,
                subfolder=LORA_SUBFOLDER,
                ignore_mismatched_sizes=True,   # handles PAD token embedding mismatch
                assign=True,                    # fixes no‑op warnings
            )
            self.model.eval()

        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

        # ── 4. Pipeline ───────────────────────────────────────────────
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        print("System Ready!")

    # ─────────────────────────────────────────────────────────────────────────
    #  Retrieval
    # ─────────────────────────────────────────────────────────────────────────
    def retrieve_context(self, query, n_results=5):
        """Fetch relevant bank data for a query."""
        account = self.detect_account_name(query)
        where   = {"sheet": account} if account else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        docs = results['documents'][0]

        print(f"\n--- [DEBUG: Retrieved {len(docs)} Chunks] ---")
        for i, d in enumerate(docs):
            print(f"{i+1}. {d[:200]}...")
        if where:
            print(f"(filtered by sheet={account})")
        print("------------------------------------------\n")

        return self.rerank_context(query, docs)

    def rerank_context(self, query: str, docs: list) -> str:
        """Re-order chunks so the most relevant one comes first."""
        query_lower = query.lower()
        keywords    = [w for w in query_lower.split() if len(w) > 2]

        def score(doc):
            doc_lower = doc.lower()
            return sum(1 for kw in keywords if kw in doc_lower)

        ranked = sorted(docs, key=score, reverse=True)
        return "\n\n".join(ranked)

    def detect_account_name(self, query: str):
        """Return a sheet name if the query mentions a known account name."""
        if not hasattr(self, '_account_names'):
            self._account_names = set()
            metadatas = self.collection.get(include=['metadatas'])['metadatas']
            for m in metadatas:
                if isinstance(m, dict):
                    name = m.get('sheet')
                    if name:
                        self._account_names.add(name.lower())
                elif isinstance(m, list):
                    for item in m:
                        if isinstance(item, dict):
                            name = item.get('sheet')
                            if name:
                                self._account_names.add(name.lower())
                else:
                    print(f"[detect_account_name] unexpected metadata element: {m!r}")
            print(f"[detect_account_name] loaded account names: {self._account_names}")

        qlow = query.lower()
        for name in self._account_names:
            if name in qlow:
                return name
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Generation
    # ─────────────────────────────────────────────────────────────────────────
    def generate_answer(self, query, context):
        """Generate a grounded response using the retrieved context."""
        prompt = f"""<|im_start|>system
You are a friendly NUST Bank support assistant.

You are given several bank records below. Your task is to:
1. Read all records carefully.
2. Find the record whose "Question" field most closely matches the user's query.
3. Answer using ONLY the "Answer" field of that record.
Guidelines:
- Use the "Answer" field as your source of truth.
- Rephrase the answer into a complete, natural sentence or short paragraph.
- If the answer contains bullet points or lists, present them in a clear, readable way (e.g., "Here are the details: …").
- Do NOT add any information not present in the selected record.
- If no record matches, say: "I don't have that specific information."

BANK RECORDS:
{context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
        outputs    = self.pipe(prompt, return_full_text=False)
        raw_answer = outputs[0]['generated_text'].strip()

        safe_answer = guardrails.validate_response(query, context, raw_answer)
        return safe_answer

    # ─────────────────────────────────────────────────────────────────────────
    #  Chat loop
    # ─────────────────────────────────────────────────────────────────────────
    def chat(self):
        """Interactive CLI loop."""
        print("\n" + "=" * 50)
        print(" NUST Bank AI Support - Active")
        print("=" * 50)
        print("Type 'exit' to quit.\n")

        while True:
            query = input("👤 User: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not query:
                continue

            print("🔍 Consulting knowledge base...", end="\r")
            context = self.retrieve_context(query)

            print("🤖 Thinking...               ", end="\r")
            answer = self.generate_answer(query, context)

            print(f"\n🤖 Assistant: {answer}")
            print("-" * 50)


if __name__ == "__main__":
    try:
        assistant = BankRAGAssistant()
        assistant.chat()
    except Exception as e:
        print(f"\n❌ Error: {e}")