import os
import base64
from io import BytesIO
import streamlit as st
from openai import OpenAI
import pandas as pd

# ── API Key: Streamlit Cloud secrets o variable de entorno ────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
client = OpenAI(api_key=OPENAI_API_KEY)

model_chat       = "gpt-4o"
model_transcribe = "whisper-1"
model_tts        = "gpt-4o-mini-tts"
CHECKIN_RATE     = 230
PICK_RATE        = 310

DEFAULT_PROMPT = (
    "Eres MFPA-AI, asistente especializado en Materials Flow Performance Analysis "
    "para el centro de distribucion RC03 de MercadoLibre.\n"
    "Conocimiento clave:\n"
    "- Dataset: 760906 inbounds, ~25631 sellers (Feb-Mar 2025)\n"
    "- Columnas: Inbound_ID, seller, Agenda, Recibido, Melis, "
    "Checkin_Procesado, Pick_Procesado, IB_Recepcion\n"
    "- Check-in: 230 u/h/persona | Picking: 310 u/h/persona\n"
    "- KMeans k=2: Cluster0 (25258 sellers estandar) | Cluster1 (373 sellers high-volume)\n"
    "- Cluster1 peak HC checkin: 76.8 personas | Cluster0: 6.4 personas\n"
    "- Hora pico historica: 10-12h\n"
    "Responde siempre en espanol, de forma clara y ejecutiva."
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MFPA RC03 · AI Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f5 !important;
    font-family: Inter, sans-serif !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #e0e0e0 !important;
}
.block-container { padding: 0 2rem 2rem !important; }

.navbar {
    background: #FFE600;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: -4rem -2rem 1.8rem -2rem;
    border-bottom: 2px solid #d4c000;
}
.navbar-logo {
    width: 36px; height: 36px; border-radius: 50%;
    background: #1a1a1a;
    display: flex; align-items: center; justify-content: center;
    color: #FFE600; font-weight: 700; font-size: 0.82rem;
}
.navbar-brand { font-weight: 700; font-size: 1.05rem; color: #1a1a1a; }
.navbar-sep { width: 1px; height: 22px; background: #c8b800; }
.navbar-tag {
    background: rgba(0,0,0,0.09); padding: 3px 11px;
    border-radius: 20px; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: #1a1a1a;
}
.navbar-user { margin-left: auto; font-size: 0.86rem; font-weight: 600; color: #1a1a1a; }
.navbar-user b { color: #1059d5; }

.hero {
    text-align: center;
    background: #fff;
    border-radius: 18px;
    border: 1px solid #e4e4e4;
    padding: 2.2rem 1rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
.orb {
    width: 96px; height: 96px; border-radius: 50%;
    background: radial-gradient(circle at 38% 32%, #f5a623, #b85a08);
    margin: 0 auto 1.2rem;
    box-shadow:
        0 0 0 20px rgba(245,166,35,0.11),
        0 0 0 40px rgba(245,166,35,0.05),
        0 8px 28px rgba(184,90,8,0.38);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem;
    color: #fff; letter-spacing: 1px;
}
.hero-sub {
    font-size: 0.66rem; letter-spacing: 4px;
    text-transform: uppercase; color: #aaa; margin-bottom: 0.5rem;
}
.hero-hello { font-size: 1.45rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.15rem; }
.hero-hello b { color: #1059d5; }
.hero-title { font-size: 1.08rem; font-weight: 600; color: #1a1a1a; }
.hero-title b { color: #1059d5; }

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 1.3rem; }
.kpi {
    background: #fff; border: 1px solid #e4e4e4;
    border-radius: 12px; padding: 1rem; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.kpi-n { font-size: 1.55rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-l { font-size: 0.67rem; color: #999; text-transform: uppercase; letter-spacing: 1px; }

.stButton > button {
    background: #FFE600 !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.81rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #e6cf00 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.14) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stChatInput"] > div {
    border: 2px solid #e0e0e0 !important;
    border-radius: 12px !important;
    background: #fff !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #1059d5 !important;
}
.info-box {
    background: #eef4ff;
    border-left: 3px solid #1059d5;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.79rem; color: #444; margin: 0.4rem 0;
}
.stSidebar label, .stSidebar p, .stSidebar small { color: #555 !important; font-size: 0.82rem !important; }
.stTextArea textarea {
    background: #fafafa !important;
    border: 1px solid #ddd !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
.empty-chat { text-align:center; padding:2.5rem 1rem; color:#ccc; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="navbar">'
    '<div class="navbar-logo">ML</div>'
    '<span class="navbar-brand">MercadoLibre</span>'
    '<div class="navbar-sep"></div>'
    '<span class="navbar-tag">MFPA</span>'
    '<span class="navbar-tag">RC03</span>'
    '<span class="navbar-tag">AI Agent</span>'
    '<div class="navbar-user">&#128075; Hola, <b>Guillermo</b></div>'
    '</div>',
    unsafe_allow_html=True
)

# ── Session state ──────────────────────────────────────────────────────────
defaults = {
    "messages": [], "files": [],
    "system_prompt": DEFAULT_PROMPT,
    "df_loaded": None, "df_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📦 MFPA · RC03")
    st.caption("Materials Flow Performance Analysis")
    st.divider()

    st.markdown("**📂 Archivo de datos**")
    uploaded = st.file_uploader(
        "Sube tu Excel o CSV",
        type=["xlsx", "csv", "txt", "png", "jpg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded:
        new_names = {f.name for f in uploaded}
        existing  = {c["name"] for c in st.session_state.files}
        for f in uploaded:
            if f.name not in existing:
                raw   = f.read()
                entry = {"name": f.name, "type": f.type}
                if f.type.startswith("image/"):
                    entry.update(kind="image",
                                 b64=base64.b64encode(raw).decode(),
                                 media_type=f.type)
                elif f.name.endswith((".xlsx", ".xls")):
                    try:
                        sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
                        parts  = []
                        for sn, sdf in sheets.items():
                            if "Recibido" in sdf.columns:
                                sdf["hc_checkin"] = (sdf["Recibido"] / CHECKIN_RATE).round(3)
                                sdf["hc_pick"]    = (sdf["Recibido"] / PICK_RATE).round(3)
                            parts.append(
                                f"Hoja {sn} ({sdf.shape[0]}x{sdf.shape[1]}):\n"
                                f"Columnas: {list(sdf.columns)}\n"
                                f"Muestra:\n{sdf.head(5).to_string()}\n"
                                f"Stats:\n{sdf.describe().to_string()}"
                            )
                            if st.session_state.df_loaded is None:
                                st.session_state.df_loaded = sdf
                                st.session_state.df_name   = f.name
                        entry.update(kind="text", content="\n\n".join(parts))
                    except Exception as e:
                        entry.update(kind="text", content=f"Error Excel: {e}")
                elif f.name.endswith(".csv"):
                    try:
                        sdf = pd.read_csv(BytesIO(raw))
                        if "Recibido" in sdf.columns:
                            sdf["hc_checkin"] = (sdf["Recibido"] / CHECKIN_RATE).round(3)
                            sdf["hc_pick"]    = (sdf["Recibido"] / PICK_RATE).round(3)
                        if st.session_state.df_loaded is None:
                            st.session_state.df_loaded = sdf
                            st.session_state.df_name   = f.name
                        entry.update(kind="text", content=(
                            f"CSV {f.name} ({sdf.shape[0]}x{sdf.shape[1]}):\n"
                            f"{sdf.head(5).to_string()}\n{sdf.describe().to_string()}"
                        ))
                    except Exception as e:
                        entry.update(kind="text", content=f"Error CSV: {e}")
                else:
                    entry.update(kind="text",
                                 content=raw.decode("utf-8", errors="replace"))
                st.session_state.files.append(entry)
        st.session_state.files = [
            c for c in st.session_state.files if c["name"] in new_names
        ]

    if st.session_state.files:
        for c in st.session_state.files:
            st.success(f'{"🖼" if c.get("kind")=="image" else "📄"} {c["name"]}')
        if st.button("🗑 Limpiar archivos", use_container_width=True):
            st.session_state.files     = []
            st.session_state.df_loaded = None
            st.rerun()

    st.divider()
    with st.expander("⚙️ Instrucciones del agente"):
        sys_in = st.text_area("", value=st.session_state.system_prompt,
                              height=160, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Guardar", use_container_width=True):
                st.session_state.system_prompt = sys_in
                st.success("Guardado ✓")
        with c2:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.system_prompt = DEFAULT_PROMPT
                st.rerun()

    st.divider()
    st.markdown("**🎤 Entrada de voz**")
    audio_val  = st.audio_input("", label_visibility="collapsed")
    ca, cb     = st.columns(2)
    with ca: send_audio = st.button("🎤 Enviar", use_container_width=True)
    with cb: tts_on     = st.toggle("🔊 TTS", value=False)

    st.divider()
    model_sel = st.selectbox("🤖 Modelo", ["gpt-4o", "gpt-4o-mini"])
    if st.button("🧹 Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════
st.markdown(
    '<div class="hero">'
    '<div class="orb">MFPA</div>'
    '<div class="hero-sub">Materials Flow Performance Analysis</div>'
    '<div class="hero-hello">&#128075; Hola, <b>Guillermo</b></div>'
    '<div class="hero-title">MFPA <b>Justificativas &middot; RC03</b></div>'
    '</div>',
    unsafe_allow_html=True
)

df_c   = st.session_state.df_loaded
n_rows = f"{df_c.shape[0]:,}" if df_c is not None else "760,906"
st.markdown(
    f'<div class="kpi-grid">'
    f'<div class="kpi"><div class="kpi-n" style="color:#1059d5">{n_rows}</div>'
    f'<div class="kpi-l">Inbounds</div></div>'
    f'<div class="kpi"><div class="kpi-n" style="color:#1059d5">25,631</div>'
    f'<div class="kpi-l">Sellers</div></div>'
    f'<div class="kpi"><div class="kpi-n" style="color:#00a650">25,258</div>'
    f'<div class="kpi-l">Cluster 0 Estandar</div></div>'
    f'<div class="kpi"><div class="kpi-n" style="color:#f5a623">373</div>'
    f'<div class="kpi-l">Cluster 1 High Vol</div></div>'
    '</div>',
    unsafe_allow_html=True
)

col_chat, col_data = st.columns([3, 1], gap="medium")

# ── Right panel ─────────────────────────────────────────────────────────
with col_data:
    st.markdown("**📊 Datos cargados**")
    if df_c is not None:
        st.caption(f"{st.session_state.df_name} · {df_c.shape[0]:,}×{df_c.shape[1]}")
        st.dataframe(df_c.head(8), use_container_width=True, height=200)
        if "Recibido" in df_c.columns:
            st.metric("HC Check-in", f"{df_c['Recibido'].sum()/CHECKIN_RATE:,.1f} h")
            st.metric("HC Picking",  f"{df_c['Recibido'].sum()/PICK_RATE:,.1f} h")
    else:
        st.caption("Carga un archivo desde el panel lateral.")
    st.markdown(
        '<div class="info-box">&#9989; Check-in: <b>230 u/h</b>'
        '<br>&#9989; Picking: <b>310 u/h</b></div>',
        unsafe_allow_html=True
    )

# ── Chat panel ───────────────────────────────────────────────────────────
with col_chat:

    def build_payload(user_prompt):
        ctx = ""
        for c in st.session_state.files:
            if c.get("kind") == "text":
                ctx += f"\n\n### {c['name']}\n{c['content'][:10000]}"
        system  = {"role": "system",
                   "content": st.session_state.system_prompt + ctx}
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages]
        imgs    = [c for c in st.session_state.files if c.get("kind") == "image"]
        if imgs:
            parts = [{"type": "text", "text": user_prompt}]
            for img in imgs:
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{img['media_type']};base64,{img['b64']}"}})
            uturn = {"role": "user", "content": parts}
        else:
            uturn = {"role": "user", "content": user_prompt}
        return [system] + history + [uturn]

    def send_message(prompt, display=None):
        if not prompt.strip():
            return
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.write(display or prompt)
        payload = build_payload(prompt)
        with chat_box:
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=model_sel, messages=payload,
                    stream=True, max_tokens=2048
                )
                response    = st.write_stream(stream)
                audio_bytes = None
                if tts_on:
                    with st.spinner("Generando audio..."):
                        try:
                            sp = client.audio.speech.create(
                                model=model_tts, voice="alloy",
                                input=response[:4000]
                            )
                            audio_bytes = sp.read()
                        except Exception:
                            pass
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        st.session_state.messages.append({
            "role": "assistant", "content": response, "audio": audio_bytes
        })

    chat_box = st.container()
    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                '<div class="empty-chat">'
                '<div style="font-size:2.4rem;margin-bottom:.6rem">&#128172;</div>'
                '<div>Escribe una pregunta sobre RC03 o carga un archivo para comenzar.</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("audio"):
                        st.audio(msg["audio"], format="audio/mp3")

    if not st.session_state.messages:
        st.markdown("**Sugerencias rápidas:**")
        c1, c2, c3 = st.columns(3)
        quick = [
            ("📦 Headcount pico",
             "Calcula cuantas personas necesito en hora pico para el Cluster 1 en check-in y picking."),
            ("🔍 Analizar archivo",
             "Analiza el archivo cargado e identifica puntos clave, clusters y metricas principales."),
            ("📈 Comparar clusters",
             "Explica diferencias operativas entre Cluster 0 y Cluster 1 y recomienda acciones de staffing."),
        ]
        for col, (lbl, prm) in zip([c1, c2, c3], quick):
            with col:
                if st.button(lbl, use_container_width=True):
                    send_message(prm)
                    st.rerun()

    if text_in := st.chat_input("Escribe tu pregunta sobre MFPA · RC03..."):
        send_message(text_in)

    if send_audio:
        if audio_val is not None:
            af      = BytesIO(audio_val.getvalue())
            af.name = getattr(audio_val, "name", "voz.mp3") or "voz.mp3"
            with st.spinner("Transcribiendo..."):
                t = client.audio.transcriptions.create(
                    model=model_transcribe, file=af
                )
            if t.text.strip():
                send_message(t.text.strip(), display=f"🎤 {t.text.strip()}")
        else:
            st.warning("Graba un mensaje primero.")
