import os
import torch
import warnings
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import guardrails
from threading import Thread
from transformers import TextIteratorStreamer

warnings.filterwarnings("ignore")

# --- CORE SETTINGS ---
DB_DIR = "./vector_store_new"
LORA_PATH = "/content/drive/MyDrive/colab_deploy/lora_bank_model_qwen"
BASE_MODEL = "Qwen/Qwen1.5-4B-Chat"

class BankRAGAssistant:
    def __init__(self):
        print("Initializing Bank RAG Assistant (Qwen-4B Elite Advisor)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        from peft import PeftModel
        print(f"Found LoRA Knowledge at {LORA_PATH}. Enhancing Intelligence...")
        # Newer huggingface_hub rejects local absolute paths as repo IDs.
        # Patch validate_repo_id to allow them during adapter loading.
        import huggingface_hub.utils._validators as _hf_validators
        _orig_validate = _hf_validators.validate_repo_id
        _hf_validators.validate_repo_id = lambda *a, **kw: None
        try:
            self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
        finally:
            _hf_validators.validate_repo_id = _orig_validate
        
        self.db_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.db_client.get_collection(name="bank_knowledge_base")
        print(f"Connected to Vector DB ({self.collection.count()} records).")
        
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        
        self.model.eval()
        print(" System Ready! Elite Advisor Active.")

    def retrieve_context(self, query):
        results = self.collection.query(query_texts=[query], n_results=15)
        pairs = [[query, doc] for doc in results['documents'][0]]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(scores, results['documents'][0]), reverse=True)
        reranked = [doc for score, doc in scored_docs]
        return "\n\n".join(reranked[:10])

    def generate_answer_stream(self, query, history=None, context=""):
        if history is None:
            history = []

        system_msg = (
            "You are the 'NUST Bank Elite Advisor'. You are warm, professional, "
            "and always format your responses beautifully using Markdown.\n\n"
            "### PRODUCT GLOSSARY:\n"
            '{\n'
            '  "LCA": "Little Champs Account", "NAA": "NUST Asaan Account", "NWA": "NUST Waqaar Account",\n'
            '  "PWRA": "PakWatan Remittance Account", "RDA": "Roshan Digital Account", "VPCA": "Value Plus Current Account",\n'
            '  "VP-BA": "Value Plus Business Account", "VPBA": "NUST Value Premium Business Account",\n'
            '  "NSDA": "NUST Special Deposit Account", "PLS": "Profit and Loss Sharing Account",\n'
            '  "CDA": "Current Deposit Account", "NMA": "NUST Maximiser Account",\n'
            '  "NADA": "NUST Asaan Digital Account", "NADRA": "NUST Asaan Digital Remittance Account",\n'
            '  "NUST4Car": "NUST4Car Auto Finance", "ESFCA": "Exporters Special Foreign Currency Account",\n'
            '  "NFDA": "NUST Freelancer Digital Account", "NSA": "NUST Sahar Accounts",\n'
            '  "PF": "NUST Personal Finance", "NMC": "NUST Bank Mastercard", "NMF": "NUST Mortgage Finance",\n'
            '  "NSF": "NUST Sahar Finance", "NIF": "NUST Imarat Finance", "NUF": "NUST Ujala Finance",\n'
            '  "NFMF": "NUST Flour Mill Finance", "NFBF": "NUST Fauri Business Finance",\n'
            '  "PMYB&ALS": "Prime Minister Youth Business and Agriculture Loan Scheme",\n'
            '  "NRF": "NUST Rice Finance", "NHF": "NUST Hunarmand Finance",\n'
            '  "HOME REMITTANCE": "Home Remittance Services"\n'
            '}\n\n'
            "### BANK RECORDS:\n"
            f"{context if context else 'General Advisory Mode.'}\n\n"
            "### FORMATTING RULES (FOLLOW EXACTLY):\n\n"
            "**When listing items** (e.g. 'name all accounts'), use bullet points:\n"
            "- **NUST Asaan Account (NAA)** — Easy-access current account\n"
            "- **Little Champs Account (LCA)** — Savings for children\n"
            "- **NUST Waqaar Account (NWA)** — Premium savings\n"
            "(and so on for each item)\n\n"
            "**When comparing products** (e.g. 'compare rates'), use a table:\n"
            "| Account | Profit Rate | Payout | Currency |\n"
            "|---------|------------|--------|----------|\n"
            "| NUST Sahar | 19.00% | Monthly | PKR |\n"
            "| NUST Waqaar | 18.50% | Quarterly | PKR |\n\n"
            "**When explaining one product**, use structured sections:\n"
            "## NUST Sahar Account (NSA)\n"
            "- **Type:** Savings Account\n"
            "- **Currency:** PKR\n"
            "- **Profit Rate:** 19.00% (paid monthly)\n"
            "- **Key Benefit:** One of the highest yield savings accounts\n"
            "- **Ideal For:** Customers seeking maximum returns\n\n"
            "### BEHAVIOR:\n"
            "1. If the user says their name, greet them warmly and remember it.\n"
            "2. NEVER dump plain text without formatting. Always use bullets, tables, or structured sections.\n"
            "3. If unsure what the user wants, ask a clarifying question.\n"
            "4. If asked non-banking questions, politely redirect to NUST services.\n"
        )

        messages = [{"role": "system", "content": system_msg}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        if not history or history[-1].get("content") != query:
            messages.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "use_cache": True,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def chat(self, query=None, history=None):
        if query:
            if not guardrails.is_query_safe(query):
                return iter(["As your NUST Wealth Advisor, I specialize in financial services. How can I help with your banking needs?"]), []

            results = self.collection.query(query_texts=[query], n_results=12)
            raw_docs = []
            for i in range(len(results['documents'][0])):
                class Doc: pass
                d = Doc()
                d.page_content = results['documents'][0][i]
                d.metadata = results['metadatas'][0][i]
                raw_docs.append(d)

            context = self.retrieve_context(query)
            return self.generate_answer_stream(query, history, context), raw_docs

        print("\n" + "=" * 60)
        print(" NUST Bank AI Terminal (Elite Mode)")
        print("=" * 60)
        while True:
            u_query = input("\n User: ")
            if u_query.lower() in ["exit", "quit", "bye"]:
                break
            gen, _ = self.chat(u_query)
            print("Assistant: ", end="")
            for token in gen:
                print(token, end="", flush=True)
            print("\n" + "-" * 60)

if __name__ == "__main__":
    assistant = BankRAGAssistant()
    assistant.chat()