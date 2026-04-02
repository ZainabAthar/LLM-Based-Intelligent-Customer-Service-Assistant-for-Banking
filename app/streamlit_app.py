import streamlit as st
import requests
import json

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "endpoint" not in st.session_state:
    st.session_state.endpoint = ""
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "stop_gen" not in st.session_state:
    st.session_state.stop_gen = False

# --- CONFIG ---
st.set_page_config(page_title="NUST Bank Advisor", layout="centered")

# --- LUXURY DARK PURPLE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Core: Abyss Navy ── */
    .stApp {
        background: linear-gradient(180deg, #08071a 0%, #0d0b2e 40%, #0a0920 100%);
        color: #d4d4e8;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ── Kill sidebar & chrome ── */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Ambient glow behind content ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: -120px;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        height: 400px;
        background: radial-gradient(ellipse, rgba(124, 58, 237, 0.12) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.6rem 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* User messages - subtle purple tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(124, 58, 237, 0.06) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        border: 1px solid rgba(124, 58, 237, 0.1) !important;
    }
    
    /* Assistant messages */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        padding: 1rem 0.5rem !important;
    }

    /* ── Chat input ── */
    .stChatFloatingInputContainer {
        background: linear-gradient(180deg, transparent 0%, #08071a 30%) !important;
        border-top: none !important;
        padding: 30px 0 20px 0 !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: rgba(124, 58, 237, 0.08) !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 14px !important;
        color: #e2e0f0 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 14px 18px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.15) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #5a5880 !important;
    }

    /* ── Send button ── */
    [data-testid="stChatInput"] button {
        background: rgba(124, 58, 237, 0.3) !important;
        border: none !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: rgba(124, 58, 237, 0.5) !important;
    }

    /* ── Tables ── */
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; border-radius: 8px; overflow: hidden; }
    th { background: rgba(124, 58, 237, 0.12) !important; color: #a78bfa !important;
         font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.08em;
         padding: 12px 16px !important; border-bottom: 1px solid rgba(124,58,237,0.15) !important; }
    td { padding: 10px 16px !important; border-bottom: 1px solid rgba(124,58,237,0.06) !important; color: #c4c0e0 !important; font-size: 0.88rem; }
    tr:hover td { background: rgba(124, 58, 237, 0.04) !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader { color: #7c7a9a !important; font-size: 0.8rem !important; }
    
    /* ── Markdown inside chat ── */
    .stChatMessage h1 { color: #c4b5fd !important; font-size: 1.3rem !important; border-bottom: 1px solid rgba(124,58,237,0.15); padding-bottom: 8px; }
    .stChatMessage h2 { color: #c4b5fd !important; font-size: 1.1rem !important; }
    .stChatMessage h3 { color: #a78bfa !important; font-size: 1rem !important; }
    .stChatMessage strong { color: #c4b5fd !important; }
    .stChatMessage code { background: rgba(124,58,237,0.12) !important; color: #c4b5fd !important; border-radius: 4px; padding: 2px 6px; }
    .stChatMessage a { color: #a78bfa !important; }
    .stChatMessage ul, .stChatMessage ol { padding-left: 1.5rem; }
    .stChatMessage li { margin-bottom: 4px; color: #c4c0e0 !important; }

    /* ── Stop button ── */
    .stButton > button {
        background: rgba(124, 58, 237, 0.15) !important;
        border: 1px solid rgba(124, 58, 237, 0.25) !important;
        color: #a78bfa !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: rgba(124, 58, 237, 0.25) !important;
        border-color: rgba(124, 58, 237, 0.4) !important;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.1) !important;
    }
    
    /* ── Text input on setup screen ── */
    [data-testid="stTextInput"] input {
        background-color: rgba(124, 58, 237, 0.08) !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e0f0 !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.12) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.35); }
</style>
""", unsafe_allow_html=True)

# --- BRAND ---
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1.5rem 0; border-bottom: 1px solid rgba(124,58,237,0.1); margin-bottom: 2rem;">
    <div style="width: 48px; height: 48px; margin: 0 auto 12px auto; border-radius: 14px; 
                background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(167,139,250,0.15)); 
                border: 1px solid rgba(124,58,237,0.2);
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 0 30px rgba(124,58,237,0.15);">
        <span style="font-size: 22px;"></span>
    </div>
    <h3 style="margin:0; color:#c4b5fd; font-weight:700; font-size:1.1rem; letter-spacing:0.03em;">NUST BANK</h3>
    <p style="margin:4px 0 0 0; color:#5a5880; font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;">Elite Financial Advisor</p>
</div>
""", unsafe_allow_html=True)

# --- LINK SETUP ---
if not st.session_state.endpoint:
    st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:2rem; border-radius:16px; 
                background:rgba(124,58,237,0.04); border:1px solid rgba(124,58,237,0.1);">
        <p style="color:#a78bfa; font-size:0.9rem; font-weight:500; margin-bottom:4px;">Initialize Connection</p>
        <p style="color:#5a5880; font-size:0.78rem;">Paste your tunnel endpoint to activate the advisor</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    url = st.text_input("endpoint", placeholder="https://xxx.ngrok-free.app", label_visibility="collapsed")
    if url:
        st.session_state.endpoint = url
        st.rerun()
    st.stop()

# --- RENDER CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- STOP BUTTON ---
if st.session_state.is_generating:
    if st.button("Stop generating"):
        st.session_state.stop_gen = True
        st.session_state.is_generating = False
        st.rerun()

# --- INPUT & STREAMING ---
if prompt := st.chat_input("Ask anything about NUST Bank…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.session_state.is_generating = True
        st.session_state.stop_gen = False

        def stream():
            try:
                with requests.post(
                    f"{st.session_state.endpoint.rstrip('/')}/chat",
                    json={"prompt": prompt, "history": st.session_state.messages[:-1]},
                    stream=True,
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                ) as r:
                    if r.status_code == 200:
                        full = ""
                        src_buf = ""
                        in_src = False
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if st.session_state.stop_gen:
                                break
                            if chunk:
                                if "[[SOURCES]]" in chunk:
                                    in_src = True
                                    parts = chunk.split("[[SOURCES]]")
                                    yield parts[0]; full += parts[0]
                                    if len(parts) > 1: src_buf += parts[1]
                                elif in_src:
                                    src_buf += chunk
                                else:
                                    yield chunk; full += chunk

                        st.session_state.messages.append({"role": "assistant", "content": full})
                        st.session_state.is_generating = False

                        if src_buf:
                            try:
                                refs = json.loads(src_buf.strip())
                                if refs:
                                    with st.expander("References"):
                                        for ref in refs:
                                            st.caption(ref)
                            except:
                                pass
                    else:
                        st.error(f"Error {r.status_code}")
                        st.session_state.is_generating = False
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.session_state.is_generating = False

        st.write_stream(stream())
        st.rerun()
