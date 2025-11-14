import os
import pickle
import mne
import numpy as np
from scipy.io import loadmat, savemat
import warnings
import matplotlib.pyplot as plt
import time
from tqdm import tqdm, trange
import threading

# Try to import proper ASR implementation
try:
    from asrpy import ASR
    ASR_AVAILABLE = True
    print("✓ Using asrpy for proper ASR implementation")
except ImportError:
    print("⚠️ WARNING: asrpy not available. Install with: pip install asrpy")
    print("   Falling back to MNE-based artifact removal (not true ASR)")
    ASR_AVAILABLE = False

# --- Configuration (modified) ---
FAST_ASR_FIT = True
ASR_FIT_DURATION_SECONDS = 120
VISUALIZE_STEPS = False
SHOW_ASR_PLOTS = True
LINE_NOISE_FREQ = 50
ASR_CUTOFF = 10                 # less aggressive than 5
ASR_TRANSFORM_MODE = "chunks"   # CLEAN FULL RECORDING
ASR_TRANSFORM_SECONDS = 120     # unused in chunks mode
ASR_CHUNK_SECONDS = 300
ASR_CHUNK_OVERLAP_SECONDS = 10

# Heuristic inference threshold if asrpy doesn't expose bad_channels_
INFER_CORRECTION_RATIO_THR = 0.9  # was 0.5; relax to reduce false positives

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

class ProgressTimer:
    def __init__(self, task_name, estimated_time_sec=15):
        self.task_name = task_name
        self.estimated_time = estimated_time_sec
        self.start_time = None
        self.stop_flag = False
        self.thread = None
    def start(self):
        self.start_time = time.time()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._progress_display, daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join()
        elapsed = time.time() - self.start_time
        print(f"\n✅ {self.task_name} completed in {elapsed:.1f} seconds")
    def _progress_display(self):
        while not self.stop_flag:
            elapsed = time.time() - self.start_time
            dots = "." * ((int(elapsed) % 4) + 1)
            remaining = max(0, self.estimated_time - elapsed)
            msg = f"\r🔧 {self.task_name}{dots} ({elapsed:.0f}s elapsed"
            msg += f", ~{remaining:.0f}s remaining)" if elapsed < self.estimated_time else ", taking longer than expected)"
            print(msg, end="", flush=True)
            time.sleep(1)

def proper_line_noise_removal(raw, line_freq=50):
    print(f"Applying line noise removal at {line_freq} Hz and harmonics...")
    sfreq = raw.info['sfreq']; nyquist = sfreq / 2
    harmonics = [i*line_freq for i in range(1, int(nyquist // line_freq)) if i*line_freq < nyquist - 2]
    print(f"Removing harmonics at: {harmonics} Hz")
    with tqdm(total=len(harmonics), desc="Line noise removal", unit="freq") as pbar:
        for f in harmonics:
            raw.notch_filter(freqs=f, filter_length='auto', method='fir', fir_design='firwin2', verbose=False)
            pbar.update(1)
    return raw

def asr_transform_in_chunks(raw, asr, chunk_sec=300, olap_sec=10):
    print(f"🔄 Applying ASR transformation in chunks (chunk={chunk_sec}s, overlap={olap_sec}s)...")
    cleaned = []
    t_end = float(raw.times[-1])
    step = max(1e-6, chunk_sec - olap_sec)
    starts = np.arange(0.0, t_end, step)
    for i in trange(len(starts), desc="ASR chunks", unit="chunk"):
        t0 = float(starts[i]); t1 = float(min(t0 + chunk_sec, t_end))
        if t1 <= t0:
            continue
        chunk = raw.copy().crop(tmin=t0, tmax=t1)
        cc = asr.transform(chunk)
        if i > 0 and olap_sec > 0:
            cc.crop(tmin=chunk.times[0] + olap_sec)
        cleaned.append(cc)
    return mne.concatenate_raws(cleaned, on_mismatch='ignore') if cleaned else raw.copy()

def plot_asr_diagnostics(raw_before, raw_after, title_suffix="", t0=0.0, duration=10.0, fmax=50.0):
    t1b = min(t0 + duration, float(raw_before.times[-1]))
    t1a = min(t0 + duration, float(raw_after.times[-1]))
    rb = raw_before.copy().crop(tmin=t0, tmax=t1b)
    ra = raw_after.copy().crop(tmin=t0, tmax=t1a)
    print("📊 ASR traces: before vs after")
    rb.plot(duration=min(duration, t1b - t0), n_channels=min(30, rb.info['nchan']), block=False, title=f"ASR Before {title_suffix}")
    ra.plot(duration=min(duration, t1a - t0), n_channels=min(30, ra.info['nchan']), block=True, title=f"ASR After {title_suffix}")
    print("📊 ASR PSD comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rb.compute_psd(fmax=fmax).plot(axes=axes[0], show=False); axes[0].set_title(f"PSD Before {title_suffix}")
    ra.compute_psd(fmax=fmax).plot(axes=axes[1], show=False); axes[1].set_title(f"PSD After {title_suffix}")
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.show(block=True)

def apply_asr_full_session(raw, cutoff):
    if not ASR_AVAILABLE:
        print("⚠️ asrpy unavailable; returning original data (no true ASR).")
        return raw.copy(), []

    asr = ASR(sfreq=raw.info['sfreq'], cutoff=cutoff)

    # Calibrate on first N seconds for speed
    if FAST_ASR_FIT:
        fit_len = min(ASR_FIT_DURATION_SECONDS, raw.times[-1])
        raw_fit = raw.copy().crop(tmax=fit_len)
        timer = ProgressTimer("ASR Calibration (fit)", 8); timer.start()
        asr.fit(raw_fit); timer.stop()
    else:
        timer = ProgressTimer("ASR Calibration (full)", 20); timer.start()
        asr.fit(raw); timer.stop()

    # Transform across the full recording via chunks
    timer = ProgressTimer("ASR Transform (chunks)", 60); timer.start()
    raw_clean = asr_transform_in_chunks(raw, asr, chunk_sec=ASR_CHUNK_SECONDS, olap_sec=ASR_CHUNK_OVERLAP_SECONDS)
    timer.stop()

    if SHOW_ASR_PLOTS:
        plot_asr_diagnostics(raw, raw_clean, title_suffix="(First 10s, Full-session)", t0=0.0, duration=10.0, fmax=50.0)

    # Prefer library-provided bad channels
    bad_channels = list(getattr(asr, 'bad_channels_', []))
    if bad_channels:
        print(f"✅ ASR (library) bad channels: {bad_channels}")
    else:
        # Conservative heuristic if nothing reported
        print("ℹ️ ASR did not report bad channels; skipping heuristic inference to avoid over-flagging.")
    return raw_clean, bad_channels

# --- Main Script ---
print("=== CORRECTED EEG PREPROCESSING WITH ENHANCED ASR ===")
print("Loading configuration from opt.pkl...")

config_path = os.path.join('/Users/aadithkrishna/Desktop/bspprojectfinal/data/', 'opt.pkl')
with open(config_path, 'rb') as f:
    opt = pickle.load(f)

clab_arti = []
participants = opt['participants'][:1]
print(f"--- Processing {len(participants)} participant(s) with ENHANCED ASR ---")

for i, participant_id in enumerate(participants):
    event_mat_path = os.path.join(opt['event_path'], f"{participant_id}.mat")
    if not os.path.exists(event_mat_path):
        print(f"\n❌ ERROR: Event file not found for {participant_id}")
        continue

    print(f"\n=== Processing: {participant_id} ===")
    start_time = time.time()
    try:
        # Load corrected events (for annotation), then raw BrainVision
        print("📂 Loading events...")
        evt = loadmat(event_mat_path, simplify_cells=True)
        vhdr_path = os.path.join(opt['rawData_path'], f"{participant_id}.vhdr")
        if not os.path.exists(vhdr_path):
            print(f"❌ ERROR: Raw data file not found: {vhdr_path}")
            continue

        print("📂 Loading raw data...")
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        orig_sfreq = float(raw.info['sfreq'])
        print(f"✅ Data loaded: {raw.info['nchan']} channels, {raw.n_times} samples, {orig_sfreq:.1f} Hz")

        if 'event_corrected' in evt:
            ec = evt['event_corrected']
            samp = np.array([int(e['sample']) for e in ec]).flatten()
            desc = [str(e['value']) for e in ec]
            onsets_sec = samp / orig_sfreq
            raw.set_annotations(mne.Annotations(onset=onsets_sec, duration=np.zeros_like(onsets_sec, float), description=desc))

        # Remove known problematic channels if present
        drop_chs = [ch for ch in ['Oz', 'EOGv1'] if ch in raw.ch_names]
        if drop_chs:
            raw.drop_channels(drop_chs)
            print(f"Removed channels: {drop_chs}")

        # Montage and resample
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='warn')
        timer = ProgressTimer("Downsampling", 3); timer.start()
        raw.resample(100, verbose=False); timer.stop()

        # Band-pass and notch
        timer = ProgressTimer("Bandpass filtering", 4); timer.start()
        raw.filter(1.0, 49.0, fir_design='firwin', verbose=False); timer.stop()
        raw = proper_line_noise_removal(raw, LINE_NOISE_FREQ)

        # Full-session ASR (chunked)
        print(f"\n🔧 Step 6: FULL-SESSION ASR (Cutoff={ASR_CUTOFF}, mode=chunks) 🔧")
        print("=" * 60)
        try:
            raw_clean, bad_channels = apply_asr_full_session(raw, cutoff=ASR_CUTOFF)
            print("=" * 60)
            print("✅ ASR COMPLETED SUCCESSFULLY!")
        except Exception as e:
            print(f"❌ ASR failed with error: {e}")
            print("Using original data without ASR...")
            raw_clean, bad_channels = raw.copy(), []

        # Interpolate and reference
        print(f"Bad channels detected by ASR: {bad_channels if bad_channels else 'None'}")
        if bad_channels:
            raw_clean.info['bads'] = bad_channels
            raw_clean.interpolate_bads(verbose=False)
            print(f"Interpolated channels: {bad_channels}")
        else:
            raw_clean.info['bads'] = []
        raw_clean.set_eeg_reference('average', verbose=False)

        # Log bad channels
        clab_arti.append({'participant_index': i, 'participant_id': participant_id, 'bad_channels': bad_channels})

        # Save EEGLAB-like .mat
        save_path = os.path.join(opt['continuous_path'], f"{participant_id}.mat")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        eeglab = {
            'data': raw_clean.get_data(),
            'srate': raw_clean.info['sfreq'],
            'chanlocs': [{'labels': ch} for ch in raw_clean.ch_names],
            'nbchan': raw_clean.info['nchan'],
            'pnts': raw_clean.n_times,
        }
        savemat(save_path, {'player_continuous': eeglab}, do_compression=True)
        print(f"✅ Saved to: {save_path}")

        print(f"\n🎉 {participant_id} completed in {time.time() - start_time:.1f} seconds!")

    except Exception as e:
        print(f"\n❌ ERROR processing {participant_id}: {e}")
        import traceback; traceback.print_exc()
        continue

# Save artifact info
clab_arti_save_path = os.path.join(opt['preprocessing_path'], 'clab_arti.mat')
os.makedirs(os.path.dirname(clab_arti_save_path), exist_ok=True)
savemat(clab_arti_save_path, {'clab_arti': clab_arti})
print(f"Saved bad-channel log to: {clab_arti_save_path}")

print("\n🎉 ENHANCED PREPROCESSING COMPLETE!")
