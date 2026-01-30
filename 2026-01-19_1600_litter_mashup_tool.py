import streamlit as st
import yt_dlp
import librosa
import numpy as np
import os
import tempfile
import uuid

try:
    from spleeter.separator import Separator
    spleeter_installed = True
except ImportError:
    spleeter_installed = False

st.title("Mashup Generator 🎵")
st.write("Lav mashups ved at hente YouTube-lyd, opdage BPM & toneart og separere vokal/instrumental-stammer.")

# YouTube-søgning
search_query = st.text_input("Søg på YouTube efter video")
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if st.button("Søg"):
    if search_query:
        with st.spinner("Søger på YouTube..."):
            ydl_opts = {'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(f"ytsearch5:{search_query}", download=False)
                    results = info.get('entries', [])
                except Exception as e:
                    st.error(f"Søgefejl: {e}")
                    results = []
            st.session_state.search_results = []
            for entry in results:
                title = entry.get('title', '')
                url = entry.get('webpage_url', '')
                dur = entry.get('duration', 0)
                minutes = dur // 60
                seconds = dur % 60
                dur_str = f"{minutes}:{seconds:02d}"
                st.session_state.search_results.append({'title': title, 'url': url, 'duration': dur_str})
if st.session_state.search_results:
    st.markdown("**Top resultater:**")
    options = []
    opt_map = {}
    title_map = {}
    for res in st.session_state.search_results:
        label = f"{res['title']} ({res['duration']})"
        options.append(label)
        opt_map[label] = res['url']
        title_map[label] = res['title']
    choice = st.selectbox("Vælg video for at tilføje link", options)
    if st.button("Tilføj link"):
        if 'link_area' not in st.session_state:
            st.session_state.link_area = ""
        st.session_state.link_area += opt_map[choice] + "\n"
        st.success(f"Link tilføjet: {title_map[choice]}")

# Links input
st.text_area("YouTube-links (én pr. linje)", value=st.session_state.get('link_area', ''), key='link_area')

# Stammedelingsmodel
model_choice = st.selectbox("Vælg stammedelingsmodel", ["Spleeter 2-stems (hurtig, god kvalitet)", "Demucs (bedre vokalkvalitet)"])
if "Demucs" in model_choice:
    st.info("Demucs giver bedre vokalkvalitet (ikke implementeret her). Bruger Spleeter.")
model = "spleeter"

# Output stammer
stem_options = ["Acapella (vokal)", "Instrumental"]
chosen_stems = st.multiselect("Vælg outputstammer", stem_options, default=stem_options)

# Processer links
if st.button("Processer links"):
    links = st.session_state.get('link_area', '').splitlines()
    if not links:
        st.warning("Indtast mindst ét YouTube-link.")
    for link in links:
        link = link.strip()
        if not link:
            continue
        st.write(f"Behandler: {link}")
        with st.spinner("Downloader, analyserer og deler stammer..."):
            # Download audio
            try:
                tmpdir = tempfile.mkdtemp()
                unique_name = str(uuid.uuid4())
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(tmpdir, unique_name + '.%(ext)s'),
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(link, download=True)
                downloaded_file = None
                for fname in os.listdir(tmpdir):
                    if fname.startswith(unique_name):
                        downloaded_file = os.path.join(tmpdir, fname)
                        break
                if not downloaded_file:
                    st.error("Download fejlede.")
                    continue
            except Exception as e:
                st.error(f"Downloadfejl: {e}")
                continue

            # Konverter til WAV
            base, ext = os.path.splitext(downloaded_file)
            if ext.lower() != '.wav':
                wav_path = os.path.join(tmpdir, unique_name + '.wav')
                os.system(f'ffmpeg -y -i "{downloaded_file}" "{wav_path}"')
            else:
                wav_path = downloaded_file

            # BPM og toneart
            try:
                y, sr = librosa.load(wav_path, sr=None)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = int(round(tempo))
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                max_idx = np.argmax(chroma_mean)
                key = chroma_labels[max_idx]
                camelot_map = {
                    'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B',
                    'E': '12B', 'F': '7B', 'F#': '2B', 'G': '9B',
                    'G#': '4B', 'A': '11B', 'A#': '6B', 'B': '1B'
                }
                camelot = camelot_map.get(key, '')
            except Exception as e:
                st.error(f"Analysefejl: {e}")
                bpm = None; key = ""; camelot = ""

            # Stammedeling (Spleeter)
            vocals_file = None
            instr_file = None
            if spleeter_installed:
                separator = Separator('spleeter:2stems')
                separator.separate_to_file(wav_path, tmpdir)
                folder = os.path.join(tmpdir, unique_name)
                vocals_file = os.path.join(folder, 'vocals.wav')
                accompaniment = os.path.join(folder, 'accompaniment.wav')
                if os.path.exists(accompaniment):
                    instr_file = accompaniment
                    os.rename(accompaniment, os.path.join(tmpdir, unique_name + '_instrumental.wav'))
                    instr_file = os.path.join(tmpdir, unique_name + '_instrumental.wav')

            # Vis BPM og toneart
            meta = f"BPM: {bpm if bpm else 'N/A'}, Toneart: {key}"
            if camelot:
                meta += f" (Camelot {camelot})"
            st.write(meta)

            # Filnavne til download
            title = info.get('title', unique_name)
            safe_title = "".join(x if x.isalnum() else "_" for x in title)
            acapella_name = safe_title + "_acapella.wav"
            instr_name = safe_title + "_instrumental.wav"

            # Download-knapper
            if vocals_file and os.path.exists(vocals_file) and "Acapella" in chosen_stems:
                with open(vocals_file, "rb") as f:
                    st.download_button("Download acapella (vokal)", f, file_name=acapella_name)
            if instr_file and os.path.exists(instr_file) and "Instrumental" in chosen_stems:
                with open(instr_file, "rb") as f:
                    st.download_button("Download instrumental", f, file_name=instr_name)
