import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_assistant import BankRAGAssistant

app = FastAPI(title="NUST Bank AI Bridge")

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    prompt: str
    history: list = [] # Required for the new Stateless Sync

# Global assistant instance
assistant = None

@app.on_event("startup")
async def startup_event():
    global assistant
    print(" Powering up the Banking Brain (Qwen-4B)...")
    assistant = BankRAGAssistant()

@app.get("/")
async def root():
    return {"status": "online", "model": "Qwen-1.5-4B-Chat"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not assistant:
        raise HTTPException(status_code=503, detail="Brain still warming up!")
    
    try:
        # Get generator and docs (Passing the history payload)
        generator, docs = assistant.chat(request.prompt, request.history)
        
        # Prepare sources
        sources = [f"Source {i+1}: {doc.metadata.get('account_name', 'Bank Record')}" for i, doc in enumerate(docs)]
        
        def stream_response():
            # 1. Stream tokens
            for token in generator:
                yield token
            
            # 2. Append sources at the end
            yield f"\n\n[[SOURCES]]\n{json.dumps(sources)}"

        return StreamingResponse(stream_response(), media_type="text/plain")
        
    except Exception as e:
        print(f"API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="ngrok auth token")
    args = parser.parse_args()

    # ngrok tunnel setup
    from pyngrok import ngrok
    ngrok.set_auth_token(args.token)
    
    # Close existing tunnels
    tunnels = ngrok.get_tunnels()
    for t in tunnels: ngrok.disconnect(t.public_url)
    
    # Open new tunnel
    public_url = ngrok.connect(8000).public_url
    print("\n" + "="*50)
    print("YOUR ENDPOINT IS ALIVE!")
    print(f"Tunnel URL: {public_url}")
    print("Copy/Paste this into the Streamlit Sidebar.")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
