import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import json
import os
import tempfile
import pickle
import shutil
from datetime import datetime
from pathlib import Path

# App config
st.set_page_config(page_title="Influencer Brain", page_icon="🧠", layout="wide")
BRAINS_DIR = Path("brains")
BRAINS_DIR.mkdir(exist_ok=True)

st.title("🧠 Influencer Brain")
st.markdown("Create persistent AI brains from any YouTube channel — with their actual voice clone.")

# ── Persistence helpers ──────────────────────────────────────────────
def list_saved_brains():
    brains = []
    for folder in BRAINS_DIR.iterdir():
        if folder.is_dir() and (folder / "meta.json").exists():
            try:
                with open(folder / "meta.json") as f:
                    meta = json.load(f)
                brains.append({
                    "name": meta.get("channel_name", folder.name),
                    "folder": folder.name,
                    "chunks": meta.get("chunk_count", 0),
                    "videos": meta.get("video_count", 0),
                    "has_voice": (folder / "voice_sample.wav").exists(),
                    "created": meta.get("created", ""),
                })
            except Exception:
                continue
    return sorted(brains, key=lambda x: x["created"], reverse=True)

def save_brain(channel_name, chunks, vectorizer, tfidf_matrix, messages, voice_sample_path=None):
    safe_name = re.sub(r"[^\w\-]", "_", channel_name)[:40]
    folder = BRAINS_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(exist_ok=True)
    
    meta = {
        "channel_name": channel_name,
        "chunk_count": len(chunks),
        "video_count": len(set(c["video_id"] for c in chunks)),
        "created": datetime.now().isoformat(),
        "has_voice": voice_sample_path is not None,
    }
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    with open(folder / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(folder / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(folder / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(folder / "messages.pkl", "wb") as f:
        pickle.dump(messages, f)
    
    if voice_sample_path and os.path.exists(voice_sample_path):
        shutil.copy(voice_sample_path, folder / "voice_sample.wav")
    
    return folder.name

def load_brain(folder_name):
    folder = BRAINS_DIR / folder_name
    with open(folder / "meta.json") as f:
        meta = json.load(f)
    with open(folder / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(folder / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(folder / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    try:
        with open(folder / "messages.pkl", "rb") as f:
            messages = pickle.load(f)
    except Exception:
        messages = []
    voice_path = folder / "voice_sample.wav" if (folder / "voice_sample.wav").exists() else None
    return meta["channel_name"], chunks, vectorizer, tfidf_matrix, messages, voice_path

def delete_brain(folder_name):
    folder = BRAINS_DIR / folder_name
    if folder.exists():
        shutil.rmtree(folder)

# ── Voice cloning ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading voice model (first time takes a few minutes)...")
def load_tts():
    try:
        from TTS.api import TTS
        # XTTS-v2: clones from a short reference wav
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return tts
    except Exception as e:
        st.error(f"Could not load TTS model: {e}")
        return None

def extract_voice_sample(video_id, out_path, duration=15):
    """Download a short clean audio clip for voice cloning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "sample.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": ["-t", str(duration)],  # first N seconds
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            for f in os.listdir(tmpdir):
                if f.endswith(".wav"):
                    shutil.copy(os.path.join(tmpdir, f), out_path)
                    return True
        except Exception:
            pass
    return False

def speak_with_clone(text, voice_sample_path):
    tts = load_tts()
    if tts is None or not voice_sample_path or not os.path.exists(voice_sample_path):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        tts.tts_to_file(
            text=text[:500],  # limit length for speed
            speaker_wav=str(voice_sample_path),
            language="en",
            file_path=out_path,
        )
        return out_path
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")
        return None

# ── Session state ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "brain_ready" not in st.session_state:
    st.session_state.brain_ready = False
if "current_brain_folder" not in st.session_state:
    st.session_state.current_brain_folder = None
if "voice_sample" not in st.session_state:
    st.session_state.voice_sample = None

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Brains")
    saved = list_saved_brains()
    
    if saved:
        brain_options = {f"{b['name']} ({b['chunks']} chunks){' 🎙️' if b['has_voice'] else ''}": b["folder"] for b in saved}
        selected = st.selectbox("Load existing brain", ["— Select —"] + list(brain_options.keys()))
        if selected != "— Select —":
            if st.button("📂 Load Brain"):
                folder = brain_options[selected]
                name, chunks, vec, mat, msgs, voice = load_brain(folder)
                st.session_state.channel_name = name
                st.session_state.chunks = chunks
                st.session_state.vectorizer = vec
                st.session_state.tfidf_matrix = mat
                st.session_state.messages = msgs or [{"role": "assistant", "content": f"Loaded brain for **{name}**. Ask me anything!"}]
                st.session_state.brain_ready = True
                st.session_state.current_brain_folder = folder
                st.session_state.voice_sample = str(voice) if voice else None
                st.rerun()
        
        if st.session_state.current_brain_folder:
            if st.button("🗑️ Delete current brain"):
                delete_brain(st.session_state.current_brain_folder)
                for key in ["brain_ready", "chunks", "vectorizer", "tfidf_matrix", "channel_name", "messages", "current_brain_folder", "voice_sample"]:
                    st.session_state.pop(key, None)
                st.rerun()
    else:
        st.caption("No saved brains yet.")
    
    st.divider()
    st.header("Create New Brain")
    channel_input = st.text_input("YouTube username", placeholder="mkbhd or @veritasium")
    max_videos = st.slider("Max videos", 1, 15, 5)
    force_transcribe = st.checkbox("Force Whisper transcription", value=False)
    clone_voice = st.checkbox("Clone their voice (recommended)", value=True)
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    build_btn = st.button("🚀 Build & Save Brain", type="primary")
    
    if st.session_state.get("brain_ready"):
        st.success(f"**{st.session_state.get('channel_name')}**")
        st.caption(f"{len(st.session_state.get('chunks', []))} chunks")
        if st.session_state.voice_sample:
            st.caption("🎙️ Voice clone ready")

# ── Core functions ───────────────────────────────────────────────────
def normalize_channel_url(input_str):
    input_str = input_str.strip().lstrip("@")
    if not input_str:
        return ""
    if "youtube.com" in input_str or "youtu.be" in input_str:
        if "/videos" not in input_str:
            return input_str.rstrip("/") + "/videos"
        return input_str
    return f"https://www.youtube.com/@{input_str}/videos"

def get_channel_videos(channel_url, max_videos):
    ydl_opts = {"quiet": True, "extract_flat": True, "playlistend": max_videos, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        if not info or "entries" not in info:
            return [], "Unknown"
        videos = [{"id": e["id"], "title": e.get("title", "Untitled"), "url": f"https://www.youtube.com/watch?v={e['id']}"} 
                  for e in info["entries"] if e and e.get("id")]
        return videos[:max_videos], info.get("channel", info.get("uploader", "Unknown"))

def get_existing_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join(s.text for s in transcript)
        return text if len(text) > 50 else None
    except Exception:
        return None

def download_and_transcribe(video_id, title):
    try:
        import whisper
    except ImportError:
        return None, "Whisper not installed"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            audio = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith((".m4a", ".webm", ".mp3"))), None)
            if not audio:
                return None, "No audio file"
            model = whisper.load_model("base")
            result = model.transcribe(audio, fp16=False)
            return result["text"], None
        except Exception as e:
            return None, str(e)

def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size-overlap) if len(" ".join(words[i:i+chunk_size])) > 50]

def build_brain(channel_input, max_videos, force_transcribe=False, clone_voice=True):
    progress = st.progress(0)
    status = st.empty()
    url = normalize_channel_url(channel_input)
    if not url:
        st.error("Enter a username.")
        return
    status.text("Fetching videos...")
    videos, channel_name = get_channel_videos(url, max_videos)
    if not videos:
        st.error("Could not fetch videos.")
        return
    
    voice_sample = None
    if clone_voice and videos:
        status.text("Extracting voice sample...")
        tmp_voice = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        if extract_voice_sample(videos[0]["id"], tmp_voice):
            voice_sample = tmp_voice
            status.text("Voice sample captured.")
    
    all_chunks = []
    for i, video in enumerate(videos):
        status.text(f"Processing {i+1}/{len(videos)}: {video['title'][:45]}...")
        progress.progress((i + 0.2) / len(videos))
        transcript = None if force_transcribe else get_existing_transcript(video["id"])
        if not transcript:
            status.text(f"Transcribing {i+1}/{len(videos)}...")
            transcript, err = download_and_transcribe(video["id"], video["title"])
            if err:
                continue
        if transcript:
            for chunk in chunk_text(transcript):
                all_chunks.append({"text": chunk, "title": video["title"], "url": video["url"], "video_id": video["id"]})
        progress.progress((i + 1) / len(videos))
    
    if not all_chunks:
        st.error("No content processed.")
        return
    
    status.text("Indexing & saving...")
    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    messages = [{"role": "assistant", "content": f"Brain for **{channel_name}** ready! {len(all_chunks)} chunks from {len(videos)} videos."}]
    folder = save_brain(channel_name, all_chunks, vectorizer, tfidf_matrix, messages, voice_sample)
    
    st.session_state.chunks = all_chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf_matrix
    st.session_state.channel_name = channel_name
    st.session_state.messages = messages
    st.session_state.brain_ready = True
    st.session_state.current_brain_folder = folder
    st.session_state.voice_sample = str(BRAINS_DIR / folder / "voice_sample.wav") if voice_sample else None
    progress.progress(1.0)
    status.success(f"Saved! {len(all_chunks)} chunks." + (" Voice clone ready." if voice_sample else ""))

def search_relevant(query, top_k=5):
    if not st.session_state.brain_ready:
        return []
    qv = st.session_state.vectorizer.transform([query])
    sims = cosine_similarity(qv, st.session_state.tfidf_matrix).flatten()
    idxs = sims.argsort()[-top_k:][::-1]
    return [st.session_state.chunks[i] for i in idxs if sims[i] > 0.05]

def generate_with_openai(query, context_chunks, channel_name, api_key):
    context = "\n\n".join(f"From '{c['title']}':\n{c['text']}" for c in context_chunks)
    system = f"You are {channel_name}. Answer ONLY from the provided video knowledge. Stay in character, cite videos, be conversational."
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}], "temperature": 0.7, "max_tokens": 700},
            timeout=40)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI error: {e}"

# ── Main UI ──────────────────────────────────────────────────────────
if build_btn and channel_input:
    build_brain(channel_input, max_videos, force_transcribe, clone_voice)
    st.rerun()

if st.session_state.brain_ready:
    st.subheader(f"Chat with {st.session_state.channel_name}")
    speak_replies = st.checkbox("🎙️ Speak replies in their voice", value=True, disabled=not st.session_state.voice_sample)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"])
    
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant = search_relevant(prompt)
                if openai_key and relevant:
                    reply = generate_with_openai(prompt, relevant, st.session_state.channel_name, openai_key)
                elif relevant:
                    excerpts = "\n\n".join(f"**{c['title']}** ([link]({c['url']}))\n> {c['text'][:350]}..." for c in relevant[:3])
                    reply = f"Relevant knowledge:\n\n{excerpts}\n\n*(Add OpenAI key for natural replies)*"
                else:
                    reply = "No relevant info found in this brain."
                
                st.markdown(reply)
                audio_path = None
                if speak_replies and st.session_state.voice_sample and reply:
                    with st.spinner("Generating voice..."):
                        audio_path = speak_with_clone(reply, st.session_state.voice_sample)
                        if audio_path:
                            st.audio(audio_path)
                
                msg = {"role": "assistant", "content": reply}
                if audio_path:
                    msg["audio"] = audio_path
                st.session_state.messages.append(msg)
                
                # Persist chat (audio paths are temporary; re-generate on load if needed)
                if st.session_state.current_brain_folder:
                    with open(BRAINS_DIR / st.session_state.current_brain_folder / "messages.pkl", "wb") as f:
                        pickle.dump(st.session_state.messages, f)
else:
    st.info("Create a new brain or load an existing one from the sidebar.")
    st.markdown("""
    ### How to use
    1. Enter a YouTube username
    2. Check **Clone their voice**
    3. Click **Build & Save Brain**
    4. Chat — enable "Speak replies in their voice" to hear them talk back
    5. Everything (knowledge + voice sample) is saved in the `brains/` folder
    
    **First voice use** downloads the XTTS model (~1.8 GB). After that it’s cached.
    """)
