"""
app.py
------
Telecom Egypt Customer Service Chatbot — clean chat UI, no file uploads.
"""

import streamlit as st

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Egypt | Customer Service",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Constrain chat width for a messenger feel */
    .block-container {
        max-width: 780px;
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* Avatar label tweak */
    .stChatMessage { padding: 0.4rem 0; }

    /* Typing cursor blink */
    @keyframes blink { 50% { opacity: 0; } }
    .cursor { animation: blink 1s step-start infinite; }

    /* Online dot in header */
    .online-dot {
        display: inline-block;
        width: 9px; height: 9px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 5px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Backend init (cached — runs once, scrapes if DB is empty) ──────────────────
@st.cache_resource(show_spinner=False)
def _init_backend():
    import vectorstore, scraper
    if not vectorstore.is_populated():
        pages = scraper.scrape_te_website(verbose=False)
        vectorstore.add_documents(pages, verbose=False)
    return True


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "lang" not in st.session_state:
    st.session_state.lang = "en"   # "en" or "ar"


# ── Header bar ─────────────────────────────────────────────────────────────────
col_logo, col_info, col_lang, col_clear = st.columns([0.12, 0.58, 0.16, 0.14])

with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Telecom_Egypt_logo.svg/320px-Telecom_Egypt_logo.svg.png",
        width=56,
    )

with col_info:
    is_ar = st.session_state.lang == "ar"
    agent_name = "وكيل خدمة العملاء" if is_ar else "Customer Service Agent"
    st.markdown(
        f"**{agent_name}**<br>"
        f'<span style="font-size:0.78rem;color:#666;">'
        f'<span class="online-dot"></span>'
        f'{"متصل الآن" if is_ar else "Online now"}</span>',
        unsafe_allow_html=True,
    )

with col_lang:
    new_lang = st.selectbox(
        label="lang",
        options=["English", "عربي"],
        index=1 if is_ar else 0,
        label_visibility="collapsed",
    )
    st.session_state.lang = "ar" if new_lang == "عربي" else "en"
    is_ar = st.session_state.lang == "ar"

with col_clear:
    clear_label = "مسح" if is_ar else "Clear"
    if st.button(clear_label, use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()


# ── Initialise backend (show spinner only on first run) ────────────────────────
with st.spinner("جاري التحميل..." if is_ar else "Loading knowledge base..."):
    _init_backend()


# ── Welcome message (shown when chat is empty) ─────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📡"):
        if is_ar:
            st.markdown(
                "أهلاً بك! 👋 أنا وكيل خدمة العملاء لدى **تيليكوم مصر**.\n\n"
                "كيف يمكنني مساعدتك اليوم؟ يمكنك السؤال عن:\n"
                "- باقات الإنترنت والألياف\n"
                "- الأسعار والعروض\n"
                "- خدمات الأعمال\n"
                "- الدعم الفني"
            )
        else:
            st.markdown(
                "Hello! 👋 I'm a **Telecom Egypt** customer service agent.\n\n"
                "How can I help you today? You can ask me about:\n"
                "- Internet & fiber packages\n"
                "- Prices and offers\n"
                "- Business services\n"
                "- Technical support"
            )


# ── Render message history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "📡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ── Chat input ─────────────────────────────────────────────────────────────────
placeholder = "اكتب رسالتك هنا..." if is_ar else "Type your message here..."

if user_input := st.chat_input(placeholder):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Stream agent response
    with st.chat_message("assistant", avatar="📡"):
        import rag

        placeholder_el = st.empty()
        full_response  = ""

        try:
            # Pass the full message history so Gemini can follow up on previous turns
            for token, _ in rag.ask_stream(user_input, history=st.session_state.messages, top_k=5):
                if token is not None:
                    full_response += token
                    placeholder_el.markdown(full_response + '<span class="cursor">▌</span>', unsafe_allow_html=True)

            placeholder_el.markdown(full_response)

        except Exception as e:
            full_response = (
                ("حدث خطأ، يرجى المحاولة مرة أخرى." if is_ar else "Something went wrong, please try again.")
                + f"\n\n`{e}`"
            )
            placeholder_el.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
