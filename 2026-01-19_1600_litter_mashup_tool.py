# litter_mashup_tool_2026.py
import os
import shutil
import subprocess
import time
import re
import json
import logging
from typing import Dict, Union, Literal, Optional, Tuple

import streamlit as st
import librosa
import numpy as np
from mutagen.id3 import (
    ID3, TALB, TBPM, TCOM, TCON, TDRC, TKEY, TPE1, TPE2,
    TPOS, TRCK, TIT2, TXXX, COMM
)

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except Exception:
    YT_DLP_AVAILABLE = False

# --- Dynamisk Sti-håndtering (Base) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Logging setup ---
logging.basicConfig(filename=os.path.join(BASE_DIR, 'litter_mashup.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION & PATHS ---
# Her definerer vi dine specifikke stier som anmodet.
# Vi bruger r"..." for at sikre at Windows backslashes læses korrekt.

# 1. Acapellas
ACAPELLAS_DIR = r"C:\Users\brian\iCloudDrive\01_Organiseret\03_Projekter\Samples\01_Acapellas"

# 2. Instrumentals
INSTRUMENTALS_DIR = r"C:\Users\brian\iCloudDrive\01_Organiseret\03_Projekter\Samples\02_Instrumentals"

# 3. Fulde sange (Originaler)
FULL_SONGS_DIR = r"C:\Users\brian\iCloudDrive\01_Organiseret\03_Projekter\Samples\04_Full_songs"

# Midlertidige arbejdsmapper (disse kan blive liggende i script-mappen for oprydning senere)
DOWNLOAD_DIR  = os.path.join(BASE_DIR, "downloads_tmp")
SEPARATED_DIR = os.path.join(BASE_DIR, "separated_tmp")
CONVERTED_DIR = os.path.join(BASE_DIR, "converted_tmp")

# --- Hjælpefunktioner ---
def safe_mkdirs(*paths: str) -> None:
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except OSError as e:
            st.error(f"Kunne ikke oprette mappe: {p}. Fejl: {e}")

def sanitize_filename(name: str) -> str:
    # Fjerner ulovlige tegn og rydder op i mellemrum
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def open_bytes_with_retries(path: str, retries: int = 10, delay: float = 0.3) -> Optional[bytes]:
    for _ in range(retries):
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return f.read()
        except Exception:
            time.sleep(delay)
    return None

# --- Musik Analyse & Metadata ---
CAMELOT_WHEEL = {
    'C': ('8B', '8A'), 'C#': ('3B', '3A'), 'D': ('10B', '10A'), 'D#': ('5B', '5A'),
    'E': ('12B', '12A'), 'F': ('7B', '7A'), 'F#': ('2B', '2A'), 'G': ('9B', '9A'),
    'G#': ('4B', '4A'), 'A': ('11B', '11A'), 'A#': ('6B', '6A'), 'B': ('1B', '1A')
}

def estimate_bpm(audio_path: str) -> Optional[int]:
    try:
        y, sr = librosa.load(audio_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()
        return round(float(tempo))
    except Exception as e:
        logging.warning(f"BPM estimation failed: {e}")
        return None

def estimate_key(audio_path: str) -> Optional[Dict[str, str]]:
    try:
        y, sr = librosa.load(audio_path)
        # Bruger en lidt kortere varighed for hurtigere analyse hvis filen er meget lang, 
        # men her læser vi hele filen for præcision.
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = chroma_mean.argmax()
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        mode = tonnetz.mean(axis=1)[0]
        scale = 'major' if mode > 0 else 'minor'
        camelot = CAMELOT_WHEEL[key][0 if scale == 'major' else 1]
        return {"key": key, "scale": scale, "camelot": camelot}
    except Exception as e:
        logging.warning(f"Key estimation failed: {e}")
        return None

def add_metadata_to_mp3(file_path: str, title: Optional[str] = None, artist: Optional[str] = None, 
                        bpm: Optional[int] = None, key: Optional[str] = None, camelot: Optional[str] = None,
                        grouping: Optional[str] = None, comment: Optional[str] = None) -> None:
    try:
        try:
            audio = ID3(file_path)
        except Exception:
            audio = ID3()
        if title: audio["TIT2"] = TIT2(encoding=3, text=[title])
        if artist: audio["TPE1"] = TPE1(encoding=3, text=[artist])
        if bpm: audio["TBPM"] = TBPM(encoding=3, text=[str(bpm)])
        if key: audio["TKEY"] = TKEY(encoding=3, text=[key])
        if camelot: audio.add(TXXX(encoding=3, desc='Camelot', text=[camelot]))
        if grouping: audio.add(TXXX(encoding=3, desc='Grouping', text=[grouping]))
        if comment: audio.add(COMM(encoding=3, lang='eng', desc='', text=[comment]))
        audio.save(file_path)
    except Exception as e:
        logging.error(f"Metadata error for {os.path.basename(file_path)}: {e}")

# --- Kerne-funktionalitet ---
def download_audio_to_temp(youtube_url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not YT_DLP_AVAILABLE:
        st.error("yt_dlp er ikke installeret.")
        return None, None, None, None

    safe_mkdirs(DOWNLOAD_DIR)
    
    # Højeste kvalitet (320k)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s [%(id)s].%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320'
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # Filnavnet ændres af postprocessor til .mp3
            temp_path = os.path.splitext(ydl.prepare_filename(info))[0] + '.mp3'
            return info.get('uploader'), info.get('title'), info.get('id'), temp_path
    except Exception as e:
        st.error(f"Download fejl: {e}")
        return None, None, None, None

def process_audio_separation(model: str) -> None:
    with st.spinner(f"Analyserer og adskiller spor med {model}..."):
        try:
            # 1. Start med Analyse (BPM & Key)
            # Vi gør det før separationen, så vi kan bruge info til navngivning af alle filer
            st.text("Analyserer BPM og Toneart...")
            bpm = estimate_bpm(st.session_state["filepath"])
            k = estimate_key(st.session_state["filepath"])
            
            base_name = os.path.splitext(st.session_state["filename"])[0]
            
            # Lav et pænt filnavn-tag
            tag_suffix = ""
            if bpm:
                tag_suffix += f"_{bpm}BPM"
            if k:
                tag_suffix += f"_{k['camelot']}"
            
            clean_base_name = f"{base_name}{tag_suffix}"

            # 2. Gem den originale fulde sang i "04_Full_songs"
            safe_mkdirs(FULL_SONGS_DIR)
            full_song_dest = os.path.join(FULL_SONGS_DIR, f"{clean_base_name}.mp3")
            
            # Vi kopierer den downloadede/uploadede fil til destinationen
            shutil.copy2(st.session_state["filepath"], full_song_dest)
            
            # Opdater metadata på den fulde sang
            add_metadata_to_mp3(full_song_dest, 
                                title=base_name,
                                artist=st.session_state.get("artist"), 
                                bpm=bpm, 
                                camelot=k['camelot'] if k else None,
                                grouping="Full Song")

            st.success(f"Fuld sang gemt i: {FULL_SONGS_DIR}")

            # 3. Kør Demucs (Separation)
            st.text("Separerer vokal og instrumental...")
            cmd = ["demucs", "--two-stems", "vocals", "-n", model, "-o", SEPARATED_DIR, st.session_state["filepath"]]
            subprocess.run(cmd, check=True)

            # Demucs output sti
            demucs_output_name = os.path.splitext(os.path.basename(st.session_state["filepath"]))[0]
            output_folder = os.path.join(SEPARATED_DIR, model, demucs_output_name)
            vocal_wav = os.path.join(output_folder, "vocals.wav")
            instr_wav = os.path.join(output_folder, "no_vocals.wav")

            # Stier til de konverterede filer (midlertidigt)
            safe_mkdirs(CONVERTED_DIR)
            acapella_mp3_temp = os.path.join(CONVERTED_DIR, f"{clean_base_name} - Acapella.mp3")
            instr_mp3_temp = os.path.join(CONVERTED_DIR, f"{clean_base_name} - Instrumental.mp3")

            # 4. Konverter til MP3 (Højeste kvalitet: 320k)
            for src, dst, grp in [(vocal_wav, acapella_mp3_temp, "Acapella"), (instr_wav, instr_mp3_temp, "Instrumental")]:
                if os.path.exists(src):
                    # -b:a 320k sikrer 320kbps
                    subprocess.run(["ffmpeg", "-y", "-i", src, "-b:a", "320k", dst], check=True)
                    
                    add_metadata_to_mp3(dst, 
                                        title=f"{base_name} ({grp})",
                                        artist=st.session_state.get("artist"), 
                                        bpm=bpm, 
                                        camelot=k['camelot'] if k else None,
                                        grouping=grp)

            # 5. Flyt til slutdestinationer
            safe_mkdirs(ACAPELLAS_DIR, INSTRUMENTALS_DIR)
            
            final_acapella_path = os.path.join(ACAPELLAS_DIR, os.path.basename(acapella_mp3_temp))
            final_instr_path = os.path.join(INSTRUMENTALS_DIR, os.path.basename(instr_mp3_temp))

            shutil.copy2(acapella_mp3_temp, final_acapella_path)
            shutil.copy2(instr_mp3_temp, final_instr_path)

            st.success("Separation Færdig!")
            st.write(f"📂 **Acapella gemt i:** {ACAPELLAS_DIR}")
            st.write(f"📂 **Instrumental gemt i:** {INSTRUMENTALS_DIR}")

            # Vis lydafspillere
            st.subheader("Lyt til resultatet")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Acapella**")
                st.audio(open_bytes_with_retries(final_acapella_path), format="audio/mpeg")
            with col2:
                st.markdown("**Instrumental**")
                st.audio(open_bytes_with_retries(final_instr_path), format="audio/mpeg")

        except Exception as e:
            st.error(f"Fejl under processering: {e}")
            logging.error(f"Process error: {e}")

# --- Streamlit UI ---
MODEL_DESCRIPTIONS = {"htdemucs": "Standard", "htdemucs_ft": "Bedst til vokal", "mdx_extra": "Høj kvalitet"}

def main():
    st.set_page_config(page_title="Litter Mashup Tool", page_icon="🎵", layout="wide")
    st.title("🎵 The Litter Mashup Tool 2026")
    st.markdown(f"""
    **Output mapper:**
    - 🎤 Acapellas: `{ACAPELLAS_DIR}`
    - 🎹 Instrumentals: `{INSTRUMENTALS_DIR}`
    - 💿 Fulde Sange: `{FULL_SONGS_DIR}`
    """)
    
    # Opret mapper med det samme hvis de ikke findes
    safe_mkdirs(DOWNLOAD_DIR, SEPARATED_DIR, CONVERTED_DIR, ACAPELLAS_DIR, INSTRUMENTALS_DIR, FULL_SONGS_DIR)

    tab1, tab2 = st.tabs(["YouTube", "Upload"])
    model = st.sidebar.selectbox("Model", list(MODEL_DESCRIPTIONS.keys()))
    st.sidebar.info(f"Valgt model: {MODEL_DESCRIPTIONS[model]}")

    with tab1:
        url = st.text_input("YouTube URL")
        if st.button("Hent fra YouTube"):
            with st.spinner("Downloader..."):
                artist, title, vid, path = download_audio_to_temp(url)
                if path:
                    st.session_state.update({"filepath": path, "filename": os.path.basename(path), "artist": artist})
                    st.success(f"Klar: {title}")

    with tab2:
        up = st.file_uploader("Upload lydfil", type=["mp3", "wav", "flac", "m4a"])
        if up:
            # Gem uploadet fil midlertidigt
            path = os.path.join(DOWNLOAD_DIR, sanitize_filename(up.name))
            with open(path, "wb") as f: f.write(up.getbuffer())
            
            # Hvis det er en upload, sætter vi artist til Unknown, medmindre vi kan udtrække det senere
            st.session_state.update({"filepath": path, "filename": up.name, "artist": "Unknown"})
            st.info(f"Fil uploadet: {up.name}")

    if "filepath" in st.session_state:
        st.divider()
        st.write(f"Valgt fil: **{st.session_state['filename']}**")
        if st.button("Start Separation & Arkivering", type="primary"):
            process_audio_separation(model)

if __name__ == "__main__":
    main()