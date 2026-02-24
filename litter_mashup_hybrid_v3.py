# litter_mashup_hybrid_v3.py
# Hybrid: YouTube search + multi-links (batch) + Demucs separation + MP3 320k + metadata + archive folders
# Brian-friendly: simple UI, predictable structure, robust errors

import os
import re
import time
import shutil
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import streamlit as st
import numpy as np

# Heavy deps: handle gracefully
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

try:
    from mutagen.id3 import ID3, TIT2, TPE1, TBPM, TKEY, TXXX, COMM
    MUTAGEN_AVAILABLE = True
except Exception:
    MUTAGEN_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except Exception:
    YT_DLP_AVAILABLE = False


# -------------------------
# App config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_PATH = os.path.join(BASE_DIR, "litter_mashup_hybrid.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
DEFAULT_ACAPELLAS_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "acapellas")
DEFAULT_INSTRUMENTALS_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "instrumentals")
DEFAULT_FULL_SONGS_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "full_songs")

DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads_tmp")
SEPARATED_DIR = os.path.join(BASE_DIR, "separated_tmp")
CONVERTED_DIR = os.path.join(BASE_DIR, "converted_tmp")


# -------------------------
# Helpers
# -------------------------
def safe_mkdirs(*paths: str) -> None:
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception as e:
            st.error(f"Kunne ikke oprette mappe: {p}\nFejl: {e}")

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def which(cmd: str) -> Optional[str]:
    # cross-platform-ish: uses shutil.which
    import shutil as _sh
    return _sh.which(cmd)

def detect_js_runtime() -> Optional[str]:
    """Return first available JS runtime for yt_dlp (node/deno/bun)."""
    for cmd in ("node", "deno", "bun"):
        path = which(cmd)
        if path:
            return f"{cmd}:{path}"
    return None

def run_cmd(cmd: List[str], label: str) -> Tuple[bool, str]:
    """Run command and return (ok, message). Captures stdout/stderr for debugging."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            msg = f"{label} fejlede.\n\nSTDERR:\n{p.stderr[:2000]}\n\nSTDOUT:\n{p.stdout[:2000]}"
            logging.error(msg)
            return False, msg
        return True, p.stdout[:2000]
    except FileNotFoundError as e:
        msg = f"{label} kunne ikke køres: {e}"
        logging.error(msg)
        return False, msg
    except Exception as e:
        msg = f"{label} ukendt fejl: {e}"
        logging.error(msg)
        return False, msg

def open_bytes_with_retries(path: str, retries: int = 10, delay: float = 0.25) -> Optional[bytes]:
    for _ in range(retries):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
        except Exception:
            time.sleep(delay)
    return None

def has_conflict_markers(file_path: str) -> bool:
    """Detect unresolved git merge markers in a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read()
        return "<<<<<<<" in txt or "=======" in txt or ">>>>>>>" in txt
    except Exception:
        return False


# -------------------------
# Music analysis (best-effort)
# -------------------------
CAMELOT_WHEEL = {
    "C": ("8B", "8A"), "C#": ("3B", "3A"), "D": ("10B", "10A"), "D#": ("5B", "5A"),
    "E": ("12B", "12A"), "F": ("7B", "7A"), "F#": ("2B", "2A"), "G": ("9B", "9A"),
    "G#": ("4B", "4A"), "A": ("11B", "11A"), "A#": ("6B", "6A"), "B": ("1B", "1A")
}

def estimate_bpm(audio_path: str, max_seconds: int = 120) -> Optional[int]:
    if not LIBROSA_AVAILABLE:
        return None
    try:
        # read only first part for speed
        y, sr = librosa.load(audio_path, sr=None, duration=max_seconds)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()
        return int(round(float(tempo)))
    except Exception as e:
        logging.warning(f"BPM estimation failed: {e}")
        return None

def estimate_key(audio_path: str, max_seconds: int = 120) -> Optional[Dict[str, str]]:
    if not LIBROSA_AVAILABLE:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=max_seconds)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = int(chroma_mean.argmax())
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = keys[key_index]

        # heuristic: tonnetz sign for major/minor-ish
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        mode_score = float(tonnetz.mean(axis=1)[0])
        scale = "major" if mode_score > 0 else "minor"
        camelot = CAMELOT_WHEEL.get(key, ("", ""))[0 if scale == "major" else 1]
        return {"key": key, "scale": scale, "camelot": camelot}
    except Exception as e:
        logging.warning(f"Key estimation failed: {e}")
        return None


def add_metadata_to_mp3(
    file_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    camelot: Optional[str] = None,
    grouping: Optional[str] = None,
    comment: Optional[str] = None
) -> None:
    if not MUTAGEN_AVAILABLE:
        return
    try:
        try:
            audio = ID3(file_path)
        except Exception:
            audio = ID3()

        if title:
            audio["TIT2"] = TIT2(encoding=3, text=[title])
        if artist:
            audio["TPE1"] = TPE1(encoding=3, text=[artist])
        if bpm:
            audio["TBPM"] = TBPM(encoding=3, text=[str(bpm)])
        if key:
            audio["TKEY"] = TKEY(encoding=3, text=[key])
        if camelot:
            audio.add(TXXX(encoding=3, desc="Camelot", text=[camelot]))
        if grouping:
            audio.add(TXXX(encoding=3, desc="Grouping", text=[grouping]))
        if comment:
            audio.add(COMM(encoding=3, lang="eng", desc="", text=[comment]))

        audio.save(file_path)
    except Exception as e:
        logging.error(f"Metadata error for {os.path.basename(file_path)}: {e}")


# -------------------------
# YouTube search + download
# -------------------------
@dataclass
class YTResult:
    title: str
    url: str
    duration: str

def build_yt_dlp_base_opts(cookies_file: Optional[str] = None) -> Dict:
    """Common yt-dlp options to reduce noisy warnings and improve reliability."""
    opts: Dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }

    js_runtime = detect_js_runtime()
    if js_runtime:
        opts["js_runtimes"] = [js_runtime]
    else:
        opts["extractor_args"] = {"youtube": {"player_skip": ["js"]}}

    if cookies_file and os.path.isfile(cookies_file):
        opts["cookiefile"] = cookies_file

    return opts

def yt_search(query: str, limit: int = 5) -> List[YTResult]:
    results: List[YTResult] = []
    if not YT_DLP_AVAILABLE:
        return results

    ydl_opts = build_yt_dlp_base_opts()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        entries = info.get("entries", []) or []
        for e in entries:
            title = e.get("title", "")
            url = e.get("webpage_url", "")
            dur = int(e.get("duration") or 0)
            mm = dur // 60
            ss = dur % 60
            results.append(YTResult(title=title, url=url, duration=f"{mm}:{ss:02d}"))
    except Exception as e:
        logging.error(f"YouTube search failed: {e}")
    return results

def download_youtube_to_mp3_320k(youtube_url: str, cookies_file: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Returns (uploader, title, video_id, mp3_path, error_message). Uses fallback clients for common YouTube 403 cases."""
    if not YT_DLP_AVAILABLE:
        return None, None, None, None, "yt_dlp er ikke installeret."

    safe_mkdirs(DOWNLOAD_DIR)

    # yt_dlp uses ffmpeg for postprocessing
    common_opts = build_yt_dlp_base_opts(cookies_file=cookies_file)
    common_opts.update({
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s [%(id)s].%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "320"
        }],
    })

    # Fallback strategy: prefer web-like clients to avoid PO token warnings from android/ios.
    attempts = [
        {"extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"extractor_args": {"youtube": {"player_client": ["mweb"]}}},
        {"extractor_args": {"youtube": {"player_client": ["tv"]}}},
        {},
    ]

    last_error = None
    for extra in attempts:
        ydl_opts = dict(common_opts)
        base_extractor_args = common_opts.get("extractor_args", {})
        extra_extractor_args = extra.get("extractor_args", {})
        if base_extractor_args or extra_extractor_args:
            merged = {**base_extractor_args}
            for key, value in extra_extractor_args.items():
                merged[key] = value
            ydl_opts["extractor_args"] = merged
        for key, value in extra.items():
            if key != "extractor_args":
                ydl_opts[key] = value
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                raw = ydl.prepare_filename(info)
                mp3_path = os.path.splitext(raw)[0] + ".mp3"
                uploader = info.get("uploader") or info.get("channel") or "Unknown"
                title = info.get("title") or "Unknown"
                vid = info.get("id") or ""
                if not os.path.exists(mp3_path):
                    return uploader, title, vid, None, "MP3-fil blev ikke oprettet efter download."
                return uploader, title, vid, mp3_path, None
        except Exception as e:
            last_error = e
            logging.warning(f"Download attempt failed ({extra or 'default'}): {e}")

    final_msg = f"{type(last_error).__name__}: {last_error}" if last_error else "Ukendt downloadfejl."
    lower_msg = final_msg.lower()
    if "sign in to confirm you’re not a bot" in lower_msg or "sign in to confirm you're not a bot" in lower_msg:
        final_msg += " | Løsning: angiv YouTube cookies-fil i sidebaren (Netscape format)."
    elif "http error 403" in lower_msg:
        final_msg += " | Løsning: prøv cookies-fil eller et andet link/video."
    logging.error(f"Download failed after fallbacks: {final_msg}")
    return None, None, None, None, final_msg


# -------------------------
# Demucs separation
# -------------------------
MODEL_DESCRIPTIONS = {
    "htdemucs": "Standard",
    "htdemucs_ft": "Bedst til vokal",
    "mdx_extra": "Høj kvalitet"
}

def demucs_separate_two_stems_vocals(model: str, input_audio_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (vocal_wav, instr_wav, demucs_output_folder)
    Demucs CLI output default: SEPARATED_DIR/model/<track_name>/(vocals.wav,no_vocals.wav)
    track_name is basename without extension
    """
    safe_mkdirs(SEPARATED_DIR)

    track_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    cmd = ["demucs", "--two-stems", "vocals", "-n", model, "-o", SEPARATED_DIR, input_audio_path]
    ok, msg = run_cmd(cmd, "Demucs")
    if not ok:
        return None, None, msg

    out_folder = os.path.join(SEPARATED_DIR, model, track_name)
    vocal_wav = os.path.join(out_folder, "vocals.wav")
    instr_wav = os.path.join(out_folder, "no_vocals.wav")

    if not (os.path.exists(vocal_wav) or os.path.exists(instr_wav)):
        return None, None, f"Demucs output ikke fundet i: {out_folder}"

    return vocal_wav if os.path.exists(vocal_wav) else None, instr_wav if os.path.exists(instr_wav) else None, out_folder


def ffmpeg_wav_to_mp3_320k(src_wav: str, dst_mp3: str) -> Tuple[bool, str]:
    safe_mkdirs(os.path.dirname(dst_mp3))
    cmd = ["ffmpeg", "-y", "-i", src_wav, "-b:a", "320k", dst_mp3]
    return run_cmd(cmd, "FFmpeg konvertering")


# -------------------------
# Streamlit UI
# -------------------------
def ensure_prereqs_ui():
    st.sidebar.subheader("Status")

    if not YT_DLP_AVAILABLE:
        st.sidebar.error("yt_dlp: mangler")
    else:
        st.sidebar.success("yt_dlp: OK")

    if not LIBROSA_AVAILABLE:
        st.sidebar.warning("librosa: mangler (BPM/toneart slås fra)")
    else:
        st.sidebar.success("librosa: OK")

    if not MUTAGEN_AVAILABLE:
        st.sidebar.warning("mutagen: mangler (metadata slås fra)")
    else:
        st.sidebar.success("mutagen: OK")

    ffmpeg_ok = which("ffmpeg") is not None
    demucs_ok = which("demucs") is not None

    if ffmpeg_ok:
        st.sidebar.success("ffmpeg: OK")
    else:
        st.sidebar.error("ffmpeg: mangler (download/konvertering virker ikke)")

    if demucs_ok:
        st.sidebar.success("demucs: OK")
    else:
        st.sidebar.error("demucs: mangler (separation virker ikke)")

    st.sidebar.caption("Logfil: litter_mashup_hybrid.log")


def main():
    st.set_page_config(page_title="Litter Mashup Hybrid v3", page_icon="🎵", layout="wide")
    st.title("🎵 Litter Mashup Hybrid v3")
    st.write("YouTube-søgning + batch links + Demucs separation + MP3 320k + metadata + arkivering.")

    ensure_prereqs_ui()

    if has_conflict_markers(__file__):
        st.error("Filen indeholder uløste merge-konflikter (<<<<<<< ======= >>>>>>>). Ret filen før brug.")
        return

    # Output dirs configurable
    st.sidebar.subheader("Output-mapper")
    acapellas_dir = st.sidebar.text_input("🎤 Acapellas mappe", DEFAULT_ACAPELLAS_DIR)
    instrumentals_dir = st.sidebar.text_input("🎹 Instrumentals mappe", DEFAULT_INSTRUMENTALS_DIR)
    full_songs_dir = st.sidebar.text_input("💿 Full songs mappe", DEFAULT_FULL_SONGS_DIR)

    safe_mkdirs(DOWNLOAD_DIR, SEPARATED_DIR, CONVERTED_DIR, acapellas_dir, instrumentals_dir, full_songs_dir)

    st.sidebar.subheader("Model")
    model = st.sidebar.selectbox("Demucs model", list(MODEL_DESCRIPTIONS.keys()), index=0)
    st.sidebar.info(f"Valgt: {MODEL_DESCRIPTIONS[model]}")

    st.sidebar.subheader("Output")
    want_vocals = st.sidebar.checkbox("Acapella (vokal)", value=True)
    want_instr = st.sidebar.checkbox("Instrumental", value=True)
    st.sidebar.divider()
    offer_downloads = st.sidebar.checkbox("Vis download-knapper (udover at gemme i mapper)", value=True)

    st.sidebar.subheader("YouTube adgang (valgfri)")
    cookies_file = st.sidebar.text_input("Cookies-fil (Netscape .txt)", value="")
    st.sidebar.caption("Bruges hvis YouTube kræver 'Sign in to confirm you’re not a bot'.")

    # Session state: link area
    if "link_area" not in st.session_state:
        st.session_state.link_area = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    # Tabs: YouTube search + Upload
    tab_search, tab_upload = st.tabs(["YouTube (søg + batch links)", "Upload lydfil"])

    with tab_search:
        st.subheader("1) Søg på YouTube")
        if not YT_DLP_AVAILABLE:
            st.error("yt_dlp mangler, så YouTube-søgning virker ikke i denne installation.")
        else:
            q = st.text_input("Søg", placeholder="fx 'Lars Hug kysser himlen farvel'")
            colA, colB = st.columns([1, 1])
            with colA:
                limit = st.number_input("Antal resultater", min_value=1, max_value=15, value=5, step=1)
            with colB:
                do_search = st.button("Søg", type="primary")

            if do_search and q.strip():
                with st.spinner("Søger..."):
                    st.session_state.search_results = yt_search(q.strip(), int(limit))

            if st.session_state.search_results:
                st.markdown("**Top resultater**")
                labels = []
                url_map = {}
                title_map = {}
                for r in st.session_state.search_results:
                    label = f"{r.title} ({r.duration})"
                    labels.append(label)
                    url_map[label] = r.url
                    title_map[label] = r.title

                choice = st.selectbox("Vælg video", labels)
                if st.button("Tilføj link til batch-liste"):
                    st.session_state.link_area += url_map[choice] + "\n"
                    st.success(f"Tilføjet: {title_map[choice]}")

        st.subheader("2) Batch-liste")
        st.session_state.link_area = st.text_area("YouTube-links (én pr. linje)", value=st.session_state.link_area, height=160)

        st.subheader("3) Kør batch")
        start_batch = st.button("Processér alle links", type="primary")

        if start_batch:
            links = [ln.strip() for ln in st.session_state.link_area.splitlines() if ln.strip()]
            if not links:
                st.warning("Indsæt mindst ét link.")
            else:
                process_links_batch(
                    links=links,
                    model=model,
                    acapellas_dir=acapellas_dir,
                    instrumentals_dir=instrumentals_dir,
                    full_songs_dir=full_songs_dir,
                    want_vocals=want_vocals,
                    want_instr=want_instr,
                    offer_downloads=offer_downloads,
                    cookies_file=cookies_file.strip() or None
                )

    with tab_upload:
        st.subheader("Upload lydfil (mp3, wav, flac, m4a)")
        up = st.file_uploader("Vælg fil", type=["mp3", "wav", "flac", "m4a"])
        if up is not None:
            safe_mkdirs(DOWNLOAD_DIR)
            dst = os.path.join(DOWNLOAD_DIR, sanitize_filename(up.name))
            with open(dst, "wb") as f:
                f.write(up.getbuffer())

            st.success(f"Upload gemt midlertidigt: {os.path.basename(dst)}")
            artist = st.text_input("Artist (valgfrit)", value="Unknown")
            title = st.text_input("Titel (valgfrit)", value=os.path.splitext(up.name)[0])

            if st.button("Start separation + arkivering", type="primary"):
                process_one_file(
                    local_audio_path=dst,
                    artist=artist,
                    title=title,
                    model=model,
                    acapellas_dir=acapellas_dir,
                    instrumentals_dir=instrumentals_dir,
                    full_songs_dir=full_songs_dir,
                    want_vocals=want_vocals,
                    want_instr=want_instr,
                    offer_downloads=offer_downloads
                )


def process_links_batch(
    links: List[str],
    model: str,
    acapellas_dir: str,
    instrumentals_dir: str,
    full_songs_dir: str,
    want_vocals: bool,
    want_instr: bool,
    offer_downloads: bool,
    cookies_file: Optional[str] = None
) -> None:
    st.divider()
    st.subheader("Batch-kørsel")

    # Basic prereq checks
    if which("ffmpeg") is None:
        st.error("ffmpeg mangler i PATH. Stopper.")
        return
    if which("demucs") is None:
        st.error("demucs mangler i PATH. Stopper.")
        return
    if not YT_DLP_AVAILABLE:
        st.error("yt_dlp mangler. Stopper.")
        return

    if cookies_file and not os.path.isfile(cookies_file):
        st.warning(f"Cookies-fil ikke fundet: {cookies_file} (fortsætter uden cookies)")

    for i, url in enumerate(links, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}/{len(links)}")
            st.write(url)
            with st.spinner("Downloader fra YouTube..."):
                artist, title, vid, mp3_path, err = download_youtube_to_mp3_320k(url, cookies_file)

            if not mp3_path:
                detail = err or "Download fejlede (mp3 ikke fundet)."
                st.error(f"Download fejlede: {detail}")
                continue

            st.success(f"Klar: {title}")
            process_one_file(
                local_audio_path=mp3_path,
                artist=artist or "Unknown",
                title=title or os.path.splitext(os.path.basename(mp3_path))[0],
                model=model,
                acapellas_dir=acapellas_dir,
                instrumentals_dir=instrumentals_dir,
                full_songs_dir=full_songs_dir,
                want_vocals=want_vocals,
                want_instr=want_instr,
                offer_downloads=offer_downloads
            )


def process_one_file(
    local_audio_path: str,
    artist: str,
    title: str,
    model: str,
    acapellas_dir: str,
    instrumentals_dir: str,
    full_songs_dir: str,
    want_vocals: bool,
    want_instr: bool,
    offer_downloads: bool
) -> None:
    safe_mkdirs(acapellas_dir, instrumentals_dir, full_songs_dir, CONVERTED_DIR)

    # Analyze (best-effort)
    with st.spinner("Analyserer BPM og toneart (best-effort)..."):
        bpm = estimate_bpm(local_audio_path)  # may be None
        k = estimate_key(local_audio_path)    # may be None

    # Build nice base filename
    base_name_raw = sanitize_filename(title) or sanitize_filename(os.path.splitext(os.path.basename(local_audio_path))[0])
    suffix = ""
    if bpm:
        suffix += f"_{bpm}BPM"
    if k and k.get("camelot"):
        suffix += f"_{k['camelot']}"

    clean_base = f"{base_name_raw}{suffix}"

    # Show meta
    meta_parts = []
    meta_parts.append(f"**Titel:** {title}")
    meta_parts.append(f"**Artist:** {artist}")
    meta_parts.append(f"**BPM:** {bpm if bpm else 'N/A'}")
    if k:
        meta_parts.append(f"**Toneart:** {k.get('key','')} {k.get('scale','')}".strip())
        if k.get("camelot"):
            meta_parts.append(f"**Camelot:** {k['camelot']}")
    st.markdown(" • ".join(meta_parts))

    # 1) Archive full song as MP3 (copy as-is if mp3, otherwise convert)
    full_song_dest = os.path.join(full_songs_dir, f"{clean_base}.mp3")

    # If input isn't mp3, convert to mp3 first
    input_ext = os.path.splitext(local_audio_path)[1].lower()
    if input_ext != ".mp3":
        temp_full_mp3 = os.path.join(CONVERTED_DIR, f"{clean_base} - Full.mp3")
        ok, msg = run_cmd(["ffmpeg", "-y", "-i", local_audio_path, "-b:a", "320k", temp_full_mp3], "FFmpeg full-song")
        if not ok:
            st.error(msg)
            return
        shutil.copy2(temp_full_mp3, full_song_dest)
    else:
        shutil.copy2(local_audio_path, full_song_dest)

    add_metadata_to_mp3(
        full_song_dest,
        title=title,
        artist=artist,
        bpm=bpm,
        key=k["key"] if k else None,
        camelot=k["camelot"] if k else None,
        grouping="Full Song"
    )
    st.success(f"Fuld sang gemt: {os.path.basename(full_song_dest)}")

    # 2) Separate with demucs
    if which("demucs") is None:
        st.error("demucs mangler i PATH, kan ikke separere.")
        return

    with st.spinner(f"Separerer med Demucs ({model})..."):
        vocal_wav, instr_wav, msg = demucs_separate_two_stems_vocals(model, full_song_dest)

    if msg and (vocal_wav is None and instr_wav is None):
        st.error(msg)
        return

    # 3) Convert wav stems to mp3 320k + metadata
    final_acapella_path = None
    final_instr_path = None

    if want_vocals and vocal_wav and os.path.exists(vocal_wav):
        acapella_temp = os.path.join(CONVERTED_DIR, f"{clean_base} - Acapella.mp3")
        ok, err = ffmpeg_wav_to_mp3_320k(vocal_wav, acapella_temp)
        if not ok:
            st.error(err)
        else:
            final_acapella_path = os.path.join(acapellas_dir, os.path.basename(acapella_temp))
            shutil.copy2(acapella_temp, final_acapella_path)
            add_metadata_to_mp3(
                final_acapella_path,
                title=f"{title} (Acapella)",
                artist=artist,
                bpm=bpm,
                key=k["key"] if k else None,
                camelot=k["camelot"] if k else None,
                grouping="Acapella"
            )

    if want_instr and instr_wav and os.path.exists(instr_wav):
        instr_temp = os.path.join(CONVERTED_DIR, f"{clean_base} - Instrumental.mp3")
        ok, err = ffmpeg_wav_to_mp3_320k(instr_wav, instr_temp)
        if not ok:
            st.error(err)
        else:
            final_instr_path = os.path.join(instrumentals_dir, os.path.basename(instr_temp))
            shutil.copy2(instr_temp, final_instr_path)
            add_metadata_to_mp3(
                final_instr_path,
                title=f"{title} (Instrumental)",
                artist=artist,
                bpm=bpm,
                key=k["key"] if k else None,
                camelot=k["camelot"] if k else None,
                grouping="Instrumental"
            )

    # 4) Preview + downloads
    st.markdown("#### Resultat")
    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Acapella**")
        if final_acapella_path and os.path.exists(final_acapella_path):
            st.caption(os.path.basename(final_acapella_path))
            b = open_bytes_with_retries(final_acapella_path)
            if b:
                st.audio(b, format="audio/mpeg")
                if offer_downloads:
                    st.download_button(
                        "Download acapella (mp3)",
                        data=b,
                        file_name=os.path.basename(final_acapella_path),
                        mime="audio/mpeg"
                    )
        else:
            st.info("Ingen acapella genereret (enten valgt fra eller fejl).")

    with cols[1]:
        st.markdown("**Instrumental**")
        if final_instr_path and os.path.exists(final_instr_path):
            st.caption(os.path.basename(final_instr_path))
            b = open_bytes_with_retries(final_instr_path)
            if b:
                st.audio(b, format="audio/mpeg")
                if offer_downloads:
                    st.download_button(
                        "Download instrumental (mp3)",
                        data=b,
                        file_name=os.path.basename(final_instr_path),
                        mime="audio/mpeg"
                    )
        else:
            st.info("Ingen instrumental genereret (enten valgt fra eller fejl).")

    st.success("Færdig ✅")


if __name__ == "__main__":
    main()
