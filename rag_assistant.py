import os
import json
import time
import chromadb
from chromadb.utils import embedding_functions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" # Lightweight & Fast
PERSIST_DIRECTORY = "vector_store"
COLLECTION_NAME = "bank_knowledge_base"

# Note: You need a Hugging Face login for Llama 3.2. 
# If not logged in, you can use "Qwen/Qwen2.5-1.5B-Instruct" which is open-access.
# Ultra-lightweight model (Fastest - Good for basic queries)
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct" 

# Moderate quality, balanced for medium/slow laptops (0.5B parameters - RECOMMENDED)
# LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# High quality, but slower (1.5B parameters)
# LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 

import guardrails

class BankRAGAssistant:
    def __init__(self):
        print("🚀 Initializing Bank RAG Assistant (High Quality Mode)...")
        
        # 1. Setup Vector DB
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        
        # 2. Setup LLM
        print(f"📥 Loading LLM: {LLM_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        # Very conservative loading strategy for Windows compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False # Avoid meta-tensor issues on some setups
        )
        self.model.to(device)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )
        print("✅ System Ready!")

    def retrieve_context(self, query, n_results=5): # Increase results for more context
        """Fetch relevant bank data for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results['documents'][0]
        
        # DEBUG: Print retrieved context to terminal
        print(f"\n--- [DEBUG: Retrieved {len(docs)} Chunks] ---")
        for i, d in enumerate(docs):
            print(f"{i+1}. {d[:200]}...") 
        print("------------------------------------------\n")
        
        context = "\n\n".join(docs)
        return context

    def generate_answer(self, query, context):
        """Generate a response using the retrieved context."""
        prompt = f"""<|im_start|>system
You are a factual NUST Bank Support AI. Use the provided BANK RECORDS to answer.

INSTRUCTIONS:
1. Answer ONLY using the BANK RECORDS listed below.
2. If the records show rates (like 14.25%), report them EXACTLY.
3. Do NOT make up numbers or monthly payment calculations if not in the records.
4. If you aren't sure, say you don't have that specific data.

BANK RECORDS:
{context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
        outputs = self.pipe(prompt, return_full_text=False)
        raw_answer = outputs[0]['generated_text'].strip()
        
        # Apply Guardrails
        safe_answer = guardrails.validate_response(query, context, raw_answer)
        return safe_answer

    def chat(self):
        """Interactive CLI loop."""
        print("\n" + "="*50)
        print("🏦 NUST Bank AI Support - Active")
        print("="*50)
        print("Type 'exit' to quit.")
        
        while True:
            query = input("\n👤 User: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query:
                continue

            print("🔍 Consulting knowledge base...", end="\r")
            context = self.retrieve_context(query)
            
            print("🤖 Thinking...               ", end="\r")
            answer = self.generate_answer(query, context)
            
            print("\n🤖 Assistant:", answer)
            print("-" * 50)

if __name__ == "__main__":
    try:
        assistant = BankRAGAssistant()
        assistant.chat()
    except Exception as e:
        print(f"\n❌ Error: {e}")
