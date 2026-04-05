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
st.set_page_config(page_title="NUST Bank Advisor", layout="centered", initial_sidebar_state="collapsed")

# --- COLORS (Fixed Light Theme) ---
bg_color = "#ffffff"
text_color = "#000000"
primary_color = "#1d4ed8"  # Corporate Blue
secondary_color = "#f3f4f6"
bubble_user = "#eff6ff"
bubble_bot = "#f9fafb"
border_item = "#e5e7eb"

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    footer {{
        display: none !important;
    }}
    #MainMenu {{
        display: none !important;
    }}
    
    [data-testid="stSidebar"] {{
        display: none !important;
    }}

    /* Center layout */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 800px !important;
    }}

    h1, h2, h3, p {{
        color: {text_color} !important;
    }}

    /* Message Bubbles */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background-color: {bubble_user} !important;
        border: 1px solid {border_item} !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
        background-color: {bubble_bot} !important;
        border: 1px solid {border_item} !important;
    }}

    /* Chat Input */
    [data-testid="stChatInput"] {{
        background-color: {secondary_color} !important;
        border: 1px solid {border_item} !important;
        border-radius: 8px !important;
    }}
    
    [data-testid="stChatInput"] textarea {{
        color: {text_color} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {primary_color} !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
    }}

    /* Logo centering */
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- BRAND (Centered) ---
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("logo.png", width=120)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div style="text-align: center; margin-top: -10px; margin-bottom: 2rem;">
        <h1 style="color: {text_color}; font-size: 2.2rem; margin-bottom: 0;">NUST Bank Advisor</h1>
        <p style="color: {text_color}; opacity: 0.7; font-size: 0.9rem;">Intelligence • Trust • Security</p>
    </div>
""", unsafe_allow_html=True)

# --- LINK SETUP ---
if not st.session_state.endpoint:
    st.markdown(f"""
    <div style="text-align:center; padding:2rem; border-radius:12px; 
                background: {secondary_color}; border: 1px solid {border_item}; margin-bottom: 1rem;">
        <h3 style="margin-bottom: 0.5rem;">Access Required</h3>
        <p style="font-size: 0.9rem; opacity: 0.8;">Secure connection must be established.</p>
    </div>
    """, unsafe_allow_html=True)
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
