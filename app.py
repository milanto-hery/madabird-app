import os
import sys
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import onnxruntime as ort
import streamlit as st

# Safe import for setuptools/pkg_resources compatibility with tensorflow_hub
try:
    import setuptools
    import pkg_resources
except ImportError:
    pass

# Ensure root directory is in Python path for module imports
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS THEME
# ==============================================================================
st.set_page_config(
    page_title="MadaBird-AI | Bioacoustic PAM Platform",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Field Ecologist Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
        margin-top: 4px;
    }
    .species-tag-endemic {
        background-color: #065f46;
        color: #34d399;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #374151;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2937;
        border-radius: 8px 8px 0px 0px;
        color: #9ca3af;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #059669 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PATHS & PIPELINE IMPORTS / FALLBACKS
# ==============================================================================
EXPORTS_DIR = ROOT_DIR / "exports"
MODEL_ONNX_PATH = EXPORTS_DIR / "model.onnx"
SPECIES_MAP_PATH = EXPORTS_DIR / "species_mapping.json"
PERCH_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4"

# Attempt importing pipeline modules directly from src (matching Google Colab setup)
try:
    from src.data.audio_preprocessor import PerchTFHubEmbedder, extract_embeddings
    from src.data.spatial_priors import encode_spatial_priors
    HAS_SRC_MODULE = True
except Exception:
    HAS_SRC_MODULE = False

# Standalone robust fallback implementations matching Colab pipeline
if not HAS_SRC_MODULE:
    class PerchTFHubEmbedder:
        def __init__(self, model_url: str):
            import tensorflow as tf
            import tensorflow_hub as hub
            self.tf = tf
            self.model_url = model_url
            self.model = hub.load(model_url)

        def __call__(self, audio_tensor):
            if not isinstance(audio_tensor, self.tf.Tensor):
                tensor = self.tf.convert_to_tensor(audio_tensor, dtype=self.tf.float32)
            else:
                tensor = audio_tensor

            if len(tensor.shape) == 1:
                tensor = self.tf.expand_dims(tensor, axis=0)

            # 1. Primary endpoint for TFHub Perch model: infer_tf
            if hasattr(self.model, 'infer_tf'):
                return self.model.infer_tf(tensor)

            # 2. Check serving_default signature
            if hasattr(self.model, 'signatures') and 'serving_default' in self.model.signatures:
                sig = self.model.signatures['serving_default']
                try:
                    return sig(inputs=tensor)
                except Exception:
                    try:
                        return sig(tensor)
                    except Exception:
                        pass

            # 3. KerasLayer fallback
            import tensorflow_hub as hub
            keras_layer = hub.KerasLayer(self.model_url)
            return keras_layer(tensor)

    def extract_embeddings(audio_paths: list, embedder, output_path: Path, sample_rate: int = 48000, window_seconds: float = 3.0, hop_seconds: float = 1.5):
        output_path.mkdir(exist_ok=True, parents=True)
        for audio_p in audio_paths:
            audio_p = Path(audio_p)
            y, sr = librosa.load(str(audio_p), sr=sample_rate, mono=True)
            win_samples = int(window_seconds * sample_rate)
            hop_samples = int(hop_seconds * sample_rate)

            num_samples = len(y)
            segments = []
            if num_samples < win_samples:
                padded = np.pad(y, (0, win_samples - num_samples), mode='constant')
                segments.append(padded)
            else:
                start = 0
                while start + win_samples <= num_samples:
                    segments.append(y[start:start + win_samples])
                    start += hop_samples

            embeddings = []
            for seg in segments:
                inp = np.expand_dims(seg.astype(np.float32), axis=0)
                outputs = embedder(inp)

                if isinstance(outputs, dict):
                    if 'output_1' in outputs:
                        val = outputs['output_1']
                    elif 'embedding' in outputs:
                        val = outputs['embedding']
                    elif 'output_0' in outputs:
                        val = outputs['output_0']
                    else:
                        val = list(outputs.values())[0]
                else:
                    val = outputs

                if hasattr(val, 'numpy'):
                    emb_np = val.numpy()
                else:
                    emb_np = np.array(val)

                emb_flat = np.squeeze(emb_np)
                if emb_flat.ndim > 1:
                    emb_flat = emb_flat.mean(axis=0)

                embeddings.append(emb_flat)

            embeddings_arr = np.array(embeddings, dtype=np.float32)
            out_file = output_path / f"{audio_p.stem}.npy"
            np.save(out_file, embeddings_arr)

    def encode_spatial_priors(latitude=None, longitude=None, day_of_year=None) -> np.ndarray:
        lat_val = -19.0 if latitude is None else float(latitude)
        lon_val = 47.0 if longitude is None else float(longitude)
        day_val = 183 if day_of_year is None else int(day_of_year)

        lat_norm = lat_val / 90.0
        lon_norm = lon_val / 180.0
        rad = 2.0 * np.pi * day_val / 365.25
        sin_day = math.sin(rad)
        cos_day = math.cos(rad)

        return np.array([lat_norm, lon_norm, sin_day, cos_day], dtype=np.float32)


def load_robust_species_lookup(map_path: Path) -> dict:
    """Robust JSON species lookup builder."""
    if not map_path.exists():
        return {}
    with open(map_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    lookup = {}
    if isinstance(raw_data, list):
        for idx, item in enumerate(raw_data):
            lookup[idx] = item if isinstance(item, dict) else {"scientific_name": str(item)}
    elif isinstance(raw_data, dict):
        items = raw_data.get("species", raw_data)
        if isinstance(items, list):
            for idx, item in enumerate(items):
                lookup[idx] = item if isinstance(item, dict) else {"scientific_name": str(item)}
        elif isinstance(items, dict):
            for k, v in items.items():
                if isinstance(v, int):  # Reverse mapping: {"Foudia madagascariensis": 190}
                    lookup[v] = {"scientific_name": k}
                elif isinstance(v, dict):
                    idx = int(v.get("index", v.get("class_id", k if str(k).isdigit() else 0)))
                    lookup[idx] = v
                elif str(k).isdigit():  # Normal mapping: {"190": {"scientific_name": "..."}}
                    lookup[int(k)] = v if isinstance(v, dict) else {"scientific_name": str(v)}
    return lookup


@st.cache_resource
def load_onnx_session(model_path: Path):
    """Load ONNX runtime session."""
    if not model_path.exists():
        return None
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


@st.cache_resource
def get_perch_embedder():
    """Load Perch TFHub embedder with caching."""
    return PerchTFHubEmbedder(PERCH_MODEL_URL)


def compute_ecoacoustic_indices(y: np.ndarray, sr: int) -> dict:
    """Compute key ecological & bioacoustic soundscape indices."""
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    diff = np.abs(np.diff(stft, axis=1))
    sum_intensity = np.sum(stft[:, :-1], axis=1) + 1e-10
    aci = float(np.sum(np.sum(diff, axis=1) / sum_intensity))

    bio_mask = (freqs >= 2000) & (freqs <= 8000)
    bi_energy = np.mean(stft[bio_mask, :]) if np.any(bio_mask) else 0.0
    bi_index = float(np.log1p(bi_energy * 1000))

    num_bands = 10
    band_edges = np.linspace(0, min(10000, sr // 2), num_bands + 1)
    band_energies = [np.sum(stft[(freqs >= band_edges[i]) & (freqs < band_edges[i+1]), :]) for i in range(num_bands)]
    probs = np.array(band_energies) / (np.sum(band_energies) + 1e-10)

    adi = float(-np.sum(probs * np.log(probs + 1e-10)))
    aei = float(np.sum((probs - 1/num_bands)**2))

    anthro_energy = np.sum(stft[(freqs >= 1000) & (freqs < 2000), :]) + 1e-10
    bio_energy = np.sum(stft[bio_mask, :]) + 1e-10
    ndsi = float((bio_energy - anthro_energy) / (bio_energy + anthro_energy))

    rms = librosa.feature.rms(y=y)
    snr_est = float(20 * np.log10((np.max(rms) + 1e-6) / (np.min(rms) + 1e-6)))

    return {
        "ACI": round(aci, 2), "BI": round(bi_index, 2),
        "ADI": round(adi, 2), "AEI": round(aei, 3),
        "NDSI": round(ndsi, 2), "SNR_dB": round(snr_est, 1)
    }

# ==============================================================================
# 3. HEADER & SIDEBAR NAVIGATION
# ==============================================================================
st.markdown("""
<div style='text-align: center; padding: 10px 0px 25px 0px;'>
    <h1 style='color: #10b981; font-size: 2.5rem; margin-bottom: 0px;'>🦜 MadaBird-AI Platform</h1>
    <p style='color: #9ca3af; font-size: 1.1rem; margin-top: 5px;'>
        Bioacoustic Passive Acoustic Monitoring (PAM) & AI Species Identification for Madagascar's Avifauna
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Field Parameters")

REGIONAL_ZONES = {
    "Madagascar General / Unspecified Zone": (None, None, "Island-wide"),
    "Antananarivo & High Plateau Zone": (-18.91, 47.52, "Highland Moist Forest & Savannah"),
    "Toamasina & East Coast Rainforest Zone": (-18.15, 49.40, "Lowland Evergreen Rainforest"),
    "Fianarantsoa & South-East Zone": (-21.45, 47.08, "Mid-Elevation Rainforest"),
    "Mahajanga & North-West Zone": (-15.71, 46.31, "Dry Deciduous Forest"),
    "Toliara & South-West Arid Zone": (-23.35, 43.66, "Sub-arid Spiny Thicket"),
    "Antsiranana & North Zone": (-12.27, 49.29, "Moist & Dry Mixed Forest")
}

selected_zone = st.sidebar.selectbox("📍 Select Regional Zone", list(REGIONAL_ZONES.keys()))
default_lat, default_lon, env_type = REGIONAL_ZONES[selected_zone]

st.sidebar.subheader("🌍 Biogeographical Priors (Optional)")
enable_priors = st.sidebar.checkbox("Enable GPS & Date Priors in Model", value=False)

if enable_priors:
    lat_val = st.sidebar.number_input("Latitude", value=default_lat if default_lat else -18.9100, format="%.4f")
    lon_val = st.sidebar.number_input("Longitude", value=default_lon if default_lon else 47.5200, format="%.4f")
    day_val = st.sidebar.slider("Day of Year (Seasonal Prior)", min_value=1, max_value=365, value=183)
else:
    lat_val, lon_val, day_val = None, None, None

st.sidebar.subheader("🎯 Detection Threshold")
confidence_threshold = st.sidebar.slider("Minimum Confidence Cutoff (%)", min_value=5, max_value=90, value=50, step=5) / 100.0

st.sidebar.markdown("---")
st.sidebar.caption("🔒 **MadaBird-AI v2.0** | Standalone Bioacoustic AI Engine")

# ==============================================================================
# 4. APP TABULAR INTERFACE
# ==============================================================================
tab_id, tab_indices, tab_map, tab_report = st.tabs([
    "🎙️ Audio ID & Predictions", 
    "🔊 Soundscape & Ecoacoustic Indices", 
    "🗺️ GIS & Spatial Map", 
    "📊 PAM Biodiversity Survey Report"
])

if "y_audio" not in st.session_state: st.session_state.y_audio = None
if "sr_audio" not in st.session_state: st.session_state.sr_audio = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None
if "inference_results" not in st.session_state: st.session_state.inference_results = None

# --- TAB 1: AUDIO INFERENCE & SPECIES ID ---
with tab_id:
    st.subheader("1. Upload Acoustic Recording")
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        uploaded_file = st.file_uploader(
            "Select soundscape audio file from field recorder (MP3, WAV, M4A, OGG, FLAC)", 
            type=["mp3", "wav", "m4a", "ogg", "flac"]
        )

    with col_up2:
        st.info("💡 **Field Note:** The model automatically segments audio into 3s sliding windows (48 kHz) and extracts Perch embeddings for ONNX classification.")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_audio_path = Path(tmp.name)

        # Dynamic Audio Loading & Value Extraction
        y, sr = librosa.load(str(tmp_audio_path), sr=48000, mono=True)
        st.session_state.y_audio = y
        st.session_state.sr_audio = sr
        st.session_state.uploaded_filename = uploaded_file.name
        duration_sec = len(y) / sr

        # Audio Player placed ABOVE Visualizations / Predictions
        st.markdown("#### 🎧 Audio Player")
        st.audio(uploaded_file.getvalue(), format="audio/mp3")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{duration_sec:.1f} s")
        c2.metric("Sample Rate", f"{sr:,} Hz")
        c3.metric("Evaluated Windows", f"{max(1, int(np.ceil(duration_sec / 1.5))):,}")
        c4.metric("Regional Zone", env_type)

        st.markdown("---")
        if st.button("🚀 Analyze Soundscape & Predict Species", type="primary", use_container_width=True):
            with st.spinner("⚡ Extracting Perch Embeddings & Executing ONNX Classifier..."):
                species_lookup = load_robust_species_lookup(SPECIES_MAP_PATH)

                embedder = get_perch_embedder()
                temp_emb_dir = Path(tempfile.gettempdir()) / "test_embeddings"
                temp_emb_dir.mkdir(exist_ok=True, parents=True)

                extract_embeddings(
                    audio_paths=[tmp_audio_path],
                    embedder=embedder,
                    output_path=temp_emb_dir,
                    sample_rate=48_000,
                    window_seconds=3.0,
                    hop_seconds=1.5
                )

                emb_file = temp_emb_dir / f"{tmp_audio_path.stem}.npy"
                audio_embeddings = np.load(emb_file)
                num_windows = len(audio_embeddings)

                spatial_vector = encode_spatial_priors(latitude=lat_val, longitude=lon_val, day_of_year=day_val)
                spatial_batch = np.tile(spatial_vector, (num_windows, 1)).astype(np.float32)

                session = load_onnx_session(MODEL_ONNX_PATH)
                if session is not None:
                    input_names = [inp.name for inp in session.get_inputs()]
                    inputs = {
                        input_names[0]: audio_embeddings.astype(np.float32),
                        input_names[1]: spatial_batch
                    }
                    logits = session.run(None, inputs)[0]
                    probs = 1.0 / (1.0 + np.exp(-logits))
                else:
                    st.error("❌ model.onnx file not found in exports/ directory!")
                    st.stop()

                avg_probs = np.mean(probs, axis=0)
                max_probs = np.max(probs, axis=0)

                all_records = []
                for class_idx, avg_p in enumerate(avg_probs):
                    sp_info = species_lookup.get(class_idx, {})
                    sci_name = sp_info.get("scientific_name", f"Species_{class_idx}")
                    com_name = sp_info.get("common_name", sp_info.get("english_name", "Malagasy Avifauna"))
                    malagasy_name = sp_info.get("malagasy_name", "Vorona")
                    iucn = sp_info.get("iucn_status", "LC")
                    is_endemic = sp_info.get("endemic", True)

                    all_records.append({
                        "class_id": class_idx,
                        "scientific_name": sci_name,
                        "common_name": com_name,
                        "malagasy_name": malagasy_name,
                        "mean_confidence": float(avg_p),
                        "peak_confidence": float(max_probs[class_idx]),
                        "iucn": iucn,
                        "endemic": "Endemic" if is_endemic else "Native"
                    })

                df_all = pd.DataFrame(all_records).sort_values(by="mean_confidence", ascending=False).reset_index(drop=True)
                df_filtered = df_all[df_all["mean_confidence"] >= confidence_threshold].reset_index(drop=True)

                df_display = df_filtered if not df_filtered.empty else df_all.head(10)

                st.session_state.inference_results = {
                    "df": df_display,
                    "df_all": df_all,
                    "probs_matrix": probs,
                    "num_windows": num_windows,
                    "duration": duration_sec,
                    "species_lookup": species_lookup
                }
                st.success(f"✅ Prediction complete! Evaluated {num_windows} audio windows successfully.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", "-- s")
        c2.metric("Sample Rate", "-- Hz")
        c3.metric("Evaluated Windows", "--")
        c4.metric("Regional Zone", env_type)
        st.info("👆 Mampidira rakitra feo (.mp3, .wav) mba hamakafakana ny vokatry ny model.")

    if st.session_state.inference_results is not None:
        res = st.session_state.inference_results
        df_res = res["df"]

        st.markdown("---")
        st.subheader("2. Identification Results & Top Candidate Species")

        col_res1, col_res2 = st.columns([1.5, 1])

        with col_res1:
            fig_bar = px.bar(
                df_res.head(10),
                x="mean_confidence",
                y="common_name",
                orientation="h",
                color="mean_confidence",
                color_continuous_scale="Viridis",
                hover_data=["scientific_name", "malagasy_name", "iucn", "endemic"],
                labels={"mean_confidence": "Average Confidence", "common_name": "Species"},
                title="<b>Top Detected Species (Average Confidence)</b>"
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e6ed"),
                height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_res2:
            st.markdown("#### 🥇 Primary Species Detected")
            top_sp = df_res.iloc[0]
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{top_sp['scientific_name']}</div>
                <div class="metric-value">{top_sp['common_name']}</div>
                <p style="color: #34d399; margin-top: 5px; font-weight: 600;">
                    Malagasy: <i>{top_sp['malagasy_name']}</i>
                </p>
                <div style="margin-top: 10px;">
                    <span class="species-tag-endemic">{top_sp['endemic']}</span>
                    <span style="background: #374151; color: #f3f4f6; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                        IUCN: {top_sp['iucn']}
                    </span>
                </div>
                <div style="margin-top: 15px;">
                    <p style="margin:0; font-size: 0.9rem;">Mean Confidence: <b>{top_sp['mean_confidence']*100:.1f}%</b></p>
                    <p style="margin:0; font-size: 0.9rem;">Peak Confidence: <b>{top_sp['peak_confidence']*100:.1f}%</b></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("3. Species Activity Timeline (Window Detections)")
        probs_matrix = res["probs_matrix"]
        top_class_ids = df_res.head(8)["class_id"].tolist()
        top_names = df_res.head(8)["common_name"].tolist()

        timeline_data = probs_matrix[:, top_class_ids].T
        time_stamps = [f"{i*1.5:.1f}s" for i in range(res["num_windows"])]

        fig_heat = px.imshow(
            timeline_data,
            x=time_stamps,
            y=top_names,
            color_continuous_scale="Magma",
            labels=dict(x="Time Window (sec)", y="Species", color="Confidence"),
            aspect="auto",
            title="<b>Temporal Vocalization Heatmap</b>"
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e6ed"),
            height=320
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 2: SOUNDSCAPE & ECOACOUSTIC INDICES ---
with tab_indices:
    st.subheader("🔊 Soundscape Analysis & Acoustic Health Metrics")
    
    if st.session_state.y_audio is None:
        st.info("👈 Upload an audio file in **Tab 1** to analyze soundscape indices.")
    else:
        y_audio = st.session_state.y_audio
        sr_audio = st.session_state.sr_audio

        st.markdown("#### 🎧 Audio Player")
        st.audio(st.session_state.y_audio, sample_rate=sr_audio)

        indices = compute_ecoacoustic_indices(y_audio, sr_audio)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("ACI (Complexity)", indices["ACI"])
        m2.metric("BI (Bioacoustic)", indices["BI"])
        m3.metric("ADI (Diversity)", indices["ADI"])
        m4.metric("AEI (Evenness)", indices["AEI"])
        m5.metric("NDSI (Biophony)", indices["NDSI"])
        m6.metric("SNR (Signal/Noise)", f"{indices['SNR_dB']} dB")

        st.markdown("---")
        col_spec1, col_spec2 = st.columns(2)

        with col_spec1:
            st.markdown("#### 📈 Waveform Amplitude Envelope")
            fig_wave, ax_wave = plt.subplots(figsize=(8, 3.5), facecolor="#0e1117")
            ax_wave.set_facecolor("#111827")
            librosa.display.waveshow(y_audio, sr=sr_audio, ax=ax_wave, color="#10b981", alpha=0.8)
            ax_wave.set_title("Audio Waveform", color="#e0e6ed", fontsize=11, fontweight="bold")
            ax_wave.tick_params(colors="#9ca3af")
            for spine in ax_wave.spines.values(): spine.set_color("#374151")
            st.pyplot(fig_wave)

        with col_spec2:
            st.markdown("#### 🎼 Dual-Band Log-Mel Spectrogram")
            S_mel = librosa.feature.melspectrogram(y=y_audio, sr=sr_audio, n_mels=128, fmin=150, fmax=12000)
            S_db = librosa.power_to_db(S_mel, ref=np.max)

            fig_spec, ax_spec = plt.subplots(figsize=(8, 3.5), facecolor="#0e1117")
            ax_spec.set_facecolor("#111827")
            img = librosa.display.specshow(S_db, sr=sr_audio, x_axis='time', y_axis='mel', fmin=150, fmax=12000, ax=ax_spec, cmap='magma')
            fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
            ax_spec.set_title("Log-Mel Spectrogram (150Hz - 12kHz)", color="#e0e6ed", fontsize=11, fontweight="bold")
            ax_spec.tick_params(colors="#9ca3af")
            for spine in ax_spec.spines.values(): spine.set_color("#374151")
            st.pyplot(fig_spec)

# --- TAB 3: GIS & SPATIAL MAP ---
with tab_map:
    st.subheader("🗺️ Biogeographical Regional Map")

    map_lat = lat_val if lat_val is not None else -18.9100
    map_lon = lon_val if lon_val is not None else 47.5200

    col_map1, col_map2 = st.columns(2)

    with col_map1:
        map_df = pd.DataFrame({"lat": [map_lat], "lon": [map_lon], "name": [selected_zone]})
        st.map(map_df, latitude="lat", longitude="lon", zoom=5)

    with col_map2:
        st.markdown("#### 📍 Zone Characteristics")
        st.write(f"**Selected Regional Zone:** {selected_zone}")
        st.write(f"**Latitude:** `{map_lat:.4f}°`")
        st.write(f"**Longitude:** `{map_lon:.4f}°`")
        st.write(f"**Biome/Habitat:** {env_type}")

# --- TAB 4: PAM BIODIVERSITY SURVEY REPORT ---
with tab_report:
    st.subheader("📋 Passive Acoustic Monitoring (PAM) Survey Summary")

    if st.session_state.inference_results is None:
        st.info("👈 Run species prediction in **Tab 1** first to generate field monitoring data.")
    else:
        res = st.session_state.inference_results
        df_report = res.get("df_all", res.get("df"))

        col_rep1, col_rep2 = st.columns([1.5, 1])

        with col_rep1:
            st.markdown("#### 🦅 Species Detection Summary Table")
            st.dataframe(
                df_report[["scientific_name", "common_name", "malagasy_name", "endemic", "iucn", "mean_confidence", "peak_confidence"]].head(25),
                use_container_width=True,
                hide_index=True
            )

        with col_rep2:
            st.markdown("#### 📊 Biodiversity Statistics")
            total_detected = len(df_report[df_report["mean_confidence"] >= 0.1])
            endemics_count = len(df_report[(df_report["mean_confidence"] >= 0.1) & (df_report["endemic"] == "Endemic")])

            st.metric("Detected Taxa (>10% Conf)", total_detected)
            st.metric("Malagasy Endemic Taxa", f"{endemics_count}")

            csv_data = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Full Survey CSV Report",
                data=csv_data,
                file_name="madabird_pam_report.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown("---")
st.markdown("<div style='text-align:center; color:#6b7280; font-size:0.85rem;'>MadaBird-AI Platform · Developed for Madagascar Bioacoustics & Avian Conservation · Built with Streamlit & PyTorch</div>", unsafe_allow_html=True)
