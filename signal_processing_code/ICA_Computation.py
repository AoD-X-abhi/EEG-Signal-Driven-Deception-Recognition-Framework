import os
import pickle
import mne
import numpy as np
from scipy.io import loadmat, savemat
from mne.preprocessing import ICA
import warnings
import matplotlib.pyplot as plt
from collections import Counter

# Optional ICLabel (recommended)
try:
    from mne_icalabel import label_components
    ICALABEL_AVAILABLE = True
    print("Using mne-icalabel for component classification (ICLabel)")
except ImportError:
    ICALABEL_AVAILABLE = False
    print("mne-icalabel not installed. Install with: pip install mne-icalabel")

# --- Configuration ---
FAST_ICA_FIT = True
ICA_FIT_DURATION_SECONDS = 120
VISUALIZE_STEPS = False
SHOW_ICA_DONE_PLOTS = True
MAX_ICA_COMPONENTS = 25

# Epoching configuration
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
EPOCH_BASELINE = (None, 0)
REJECT_BY_ANNOTATION = True

# ICLabel thresholds (keep as in your current script)
THRESHOLDS = {
    "eye": 0.50,
    "muscle": 0.70,
    "heart": 0.70,
    "line_noise": 0.70,
    "channel_noise": 0.70,
}

# ICLabel classes and mapping
ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]

def map_label_to_key(label_str: str) -> str:
    lab = (label_str or "").strip().lower()
    if "eye" in lab or "blink" in lab:
        return "eye"
    if "muscle" in lab:
        return "muscle"
    if "heart" in lab or "cardiac" in lab:
        return "heart"
    if "line" in lab and "noise" in lab:
        return "line_noise"
    if "channel" in lab and "noise" in lab:
        return "channel_noise"
    return "other"

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

def load_opt():
    base = '/Users/aadithkrishna/Desktop/bspprojectfinal/data/'
    cfg = os.path.join(base, 'opt.pkl')
    with open(cfg, 'rb') as f:
        opt = pickle.load(f)
    return opt

def _to_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.ravel().tolist()
    return [obj]

def load_clab_arti_map(opt):
    clab_path = os.path.join(opt['preprocessing_path'], 'clab_arti.mat')
    bad_map = {}
    if not os.path.exists(clab_path):
        print("clab_arti.mat not found. Proceeding without bad-channel info.")
        return bad_map

    data = loadmat(clab_path, simplify_cells=True)
    clab = data.get('clab_arti', None)

    if isinstance(clab, dict):
        for k, v in clab.items():
            if isinstance(v, dict) and 'bad_channels' in v:
                bad_map[str(k)] = _to_list(v['bad_channels'])
            else:
                bad_map[str(k)] = _to_list(v)
        return bad_map

    clab_list = _to_list(clab)
    is_list_of_dicts = all(isinstance(x, dict) for x in clab_list if x is not None)

    if is_list_of_dicts:
        for item in clab_list:
            if not isinstance(item, dict):
                continue
            pid = item.get('participant_id', None)
            bads = item.get('bad_channels', [])
            if pid is not None:
                bad_map[str(pid)] = _to_list(bads)
        return bad_map

    bad_map["__INDEXED__"] = []
    for item in clab_list:
        if isinstance(item, dict) and 'bad_channels' in item:
            bad_map["__INDEXED__"].append(_to_list(item['bad_channels']))
        else:
            bad_map["__INDEXED__"].append(_to_list(item))
    return bad_map

def get_bad_channels_for(pid, participant_index, bad_map):
    if pid in bad_map:
        return _to_list(bad_map[pid])
    if "__INDEXED__" in bad_map:
        indexed = bad_map["__INDEXED__"]
        if isinstance(indexed, list) and 0 <= participant_index < len(indexed):
            return _to_list(indexed[participant_index])
    return []

def load_preprocessed_mat(opt, participant_id):
    cont_path = os.path.join(opt['continuous_path'], f"{participant_id}.mat")
    if not os.path.exists(cont_path):
        raise FileNotFoundError(f"Preprocessed .mat not found: {cont_path}")

    mat = loadmat(cont_path, simplify_cells=True)
    if 'player_continuous' not in mat:
        raise KeyError("Expected key 'player_continuous' not found in MAT file.")

    eeg = mat['player_continuous']
    data = eeg['data']
    srate = float(eeg['srate'])

    if 'chanlocs' in eeg and eeg['chanlocs']:
        ch_names = [c['labels'] for c in eeg['chanlocs']]
    else:
        ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
    return data, srate, ch_names

def make_raw(data, srate, ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')
    return raw

def pick_n_components(raw, bads, cap=MAX_ICA_COMPONENTS):
    n_total = len(raw.ch_names)
    n_bad = len(bads) if bads else 0
    n_good = max(1, n_total - n_bad)
    n_comp = max(1, min(n_good - 1, cap))
    return n_comp

def _parse_iclabel_output(result):
    labels = None
    y_pred = None
    y_pred_proba = None
    if isinstance(result, dict):
        labels = result.get("labels", None)
        y_pred = result.get("y_pred", None)
        y_pred_proba = result.get("y_pred_proba", None)
    elif isinstance(result, (list, tuple)):
        if len(result) >= 3:
            labels, y_pred, y_pred_proba = result[0], result[1], result[2]
        elif len(result) == 2:
            labels, y_pred = result[0], result[1]
        elif len(result) == 1:
            labels = result[0]
    return labels, y_pred, y_pred_proba

def classify_components_iclabel(raw, ica, print_diagnostics=True):
    result = label_components(raw, ica, method='iclabel')
    labels, y_pred, y_prob = _parse_iclabel_output(result)
    if isinstance(y_prob, list):
        y_prob = np.array(y_prob)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    exclude = []

    if print_diagnostics and labels is not None:
        labs = np.array(labels).astype(str)
        if isinstance(y_prob, np.ndarray) and y_prob.ndim == 1 and len(y_prob) == len(labs):
            for i, (lab, p) in enumerate(zip(labs, y_prob)):
                print(f"IC {i:02d}: {lab:14s} p={float(p):0.3f}")
        else:
            for i, lab in enumerate(labs):
                print(f"IC {i:02d}: {lab:14s}")

    if isinstance(y_prob, np.ndarray) and y_prob.ndim == 2 and y_prob.shape[1] == len(ICLABEL_CLASSES):
        for idx in range(y_prob.shape[0]):
            cls_idx = int(np.argmax(y_prob[idx, :]))
            predicted_label = ICLABEL_CLASSES[cls_idx]
            key = map_label_to_key(predicted_label)
            thr = THRESHOLDS.get(key, 1.1)
            if y_prob[idx, cls_idx] >= thr and key in THRESHOLDS:
                exclude.append(idx)
        return sorted(set(exclude)), dict(labels=labels, y_pred=y_pred, y_pred_proba=y_prob)

    if labels is not None:
        labs = np.array(labels).astype(str)
        probs = None
        if isinstance(y_prob, np.ndarray) and y_prob.ndim == 1 and len(y_prob) == len(labs):
            probs = y_prob
        for idx, lab in enumerate(labs):
            key = map_label_to_key(lab)
            if key in THRESHOLDS:
                if probs is not None:
                    if float(probs[idx]) >= THRESHOLDS[key]:
                        exclude.append(idx)
                else:
                    exclude.append(idx)
        return sorted(set(exclude)), dict(labels=labels, y_pred=y_pred, y_pred_proba=y_prob)

    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 1:
        for idx, cls_idx in enumerate(y_pred):
            try:
                predicted_label = ICLABEL_CLASSES[int(cls_idx)]
            except Exception:
                predicted_label = "other"
            key = map_label_to_key(predicted_label)
            if key in THRESHOLDS:
                exclude.append(idx)
        return sorted(set(exclude)), dict(labels=labels, y_pred=y_pred, y_pred_proba=y_prob)

    return [], dict(labels=labels, y_pred=y_pred, y_pred_proba=y_prob)

def classify_components_fallback(raw, ica):
    exclude = []
    comp_maps = ica.get_components()
    frontal_idx = [i for i, ch in enumerate(raw.ch_names) if any(tag in ch.upper() for tag in ['FP', 'AF', 'F'])]
    if not frontal_idx:
        return exclude
    for c in range(comp_maps.shape[1]):
        m = comp_maps[:, c]
        frontal_power = np.mean(np.abs(m[frontal_idx]))
        total_power = np.mean(np.abs(m))
        if total_power > 0 and frontal_power > 1.5 * total_power:
            exclude.append(c)
    return sorted(list(set(exclude)))

def save_ica_outputs(opt, participant_id, raw_clean, ica, excluded_components):
    ica_dir = opt['ica_path']
    os.makedirs(ica_dir, exist_ok=True)

    fif_path = os.path.join(ica_dir, f"{participant_id}-ica.fif")
    ica.save(fif_path, overwrite=True)
    print(f"ICA solution saved to: {fif_path}")

    mat_path = os.path.join(ica_dir, f"{participant_id}_ica_clean.mat")
    eeglab_struct = {
        'data': raw_clean.get_data(),
        'srate': raw_clean.info['sfreq'],
        'chanlocs': [{'labels': ch} for ch in raw_clean.ch_names],
        'nbchan': raw_clean.info['nchan'],
        'pnts': raw_clean.n_times,
        'reject': {'icacomps': excluded_components}
    }
    try:
        eeglab_struct['icaweights'] = ica.unmixing_matrix_
        eeglab_struct['icawinv'] = ica.mixing_matrix_
        eeglab_struct['icasphere'] = np.eye(ica.unmixing_matrix_.shape[0])
    except Exception:
        pass

    savemat(mat_path, {'player_continuous': eeglab_struct}, do_compression=True)
    print(f"ICA-cleaned data saved to: {mat_path}")

def show_ica_done_plots(raw, raw_clean, ica, excluded):
    print("ICA: Before vs After (time)")
    raw.plot(duration=10, n_channels=min(20, raw.info['nchan']), title="Before ICA", block=True)
    raw_clean.plot(duration=10, n_channels=min(20, raw_clean.info['nchan']), title="After ICA", block=True)

    print("ICA: PSD Before vs After")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw.compute_psd(fmax=50).plot(axes=axes[0], show=False); axes[0].set_title("PSD Before ICA")
    raw_clean.compute_psd(fmax=50).plot(axes=axes[1], show=False); axes[1].set_title("PSD After ICA")
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.show(block=True)

    print("ICA: All components")
    ica.plot_components(picks=range(min(25, ica.n_components_)), title="ICA Components")
    plt.show(block=True)

    if len(excluded) > 0:
        print(f"ICA: Excluded components {excluded}")
        ica.plot_components(picks=excluded, title="Excluded Components")
        plt.show(block=True)

    print("ICA: Sources (before/after)")
    ica.plot_sources(raw, start=0, stop=min(10, raw.times[-1]), title="ICA Sources (Before)")
    plt.show(block=True)
    ica.plot_sources(raw_clean, start=0, stop=min(10, raw_clean.times[-1]), title="ICA Sources (After)")
    plt.show(block=True)

def tolerant_keys(codes):
    keys = []
    for k in codes:
        k = str(k)
        # bare numeric and padded
        keys += [k, k.strip(), f"{k} "]
        # BrainVision S variants
        keys += [f"S {k}", f"S{k}", f"S  {k}"]
        # Prefix variants seen in some exports
        keys += [f"Stimulus/S {k}", f"Stimulus/S{k}", f"Stimulus/ {k}", f"Stimulus/{k}"]
    seen = set()
    out = []
    for v in keys:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def build_event_id_for_task(task_name: str):
    if task_name.lower() == "decisionmaking":
        code_map = {30: 30, 31: 31, 32: 32, 33: 33}
    elif task_name.lower() == "feedback":
        code_map = {101: 101, 102: 102}
    else:
        code_map = {}

    event_id_map = {}
    for k_num, code in code_map.items():
        for key in tolerant_keys([k_num]):
            event_id_map[key] = code
    return event_id_map

def main():
    print("=== CORRECTED ICA COMPUTATION (MNE + ICLabel) ===")
    opt = load_opt()
    bad_map = load_clab_arti_map(opt)

    participants = opt['participants'][:1]
    print(f"--- Processing ICA for {len(participants)} participant(s) ---")

    for participant_index, pid in enumerate(participants):
        print(f"\n=== ICA for: {pid} ===")
        try:
            # Load preprocessed data (already at 100 Hz)
            data, srate, ch_names = load_preprocessed_mat(opt, pid)
            raw = make_raw(data, srate, ch_names)

            # Rebuild annotations from corrected events using ORIGINAL BV sampling rate
            event_mat_path = os.path.join(opt['event_path'], f"{pid}.mat")
            try:
                player_evt = loadmat(event_mat_path, simplify_cells=True)
                if raw.annotations is None or len(raw.annotations) == 0:
                    if 'event_corrected' in player_evt and len(player_evt['event_corrected']):
                        corr = player_evt['event_corrected']
                        onsets_samples = np.array([int(ev['sample']) for ev in corr]).flatten()
                        descriptions = [str(ev['value']) for ev in corr]

                        # CRITICAL FIX: use original BrainVision sampling rate (≈500 Hz)
                        vhdr_path = os.path.join(opt['rawData_path'], f"{pid}.vhdr")
                        orig_raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
                        orig_sfreq = float(orig_raw.info['sfreq'])
                        onsets_sec = onsets_samples / orig_sfreq

                        anns = mne.Annotations(
                            onset=onsets_sec,
                            duration=np.zeros_like(onsets_sec, dtype=float),
                            description=descriptions
                        )
                        raw.set_annotations(anns)
                        print(f"Restored {len(anns)} annotations from {event_mat_path} (using orig_sfreq={orig_sfreq:.1f} Hz)")
                    else:
                        print(f"No 'event_corrected' in {event_mat_path}; annotations remain empty")
                else:
                    print(f"MAT already had {len(raw.annotations)} annotations")
            except Exception as _e:
                print(f"Could not rebuild annotations from {event_mat_path}: {_e}")

            # Cache annotations so they survive through apply()
            orig_annotations = raw.annotations.copy() if raw.annotations is not None else mne.Annotations([], [], [])

            # Bad channels from clab_arti if available
            bads = get_bad_channels_for(pid, participant_index, bad_map)
            raw.info['bads'] = bads
            print(f"Bad channels: {bads if bads else 'None'}")
            print(f"Data: {raw.info['nchan']} ch, {raw.n_times} samples, {srate} Hz, {raw.times[-1]:.1f} s")

            if VISUALIZE_STEPS:
                raw.plot(duration=10, n_channels=20, block=True, title="Preprocessed (before ICA)")

            # ICA
            n_components = pick_n_components(raw, bads, cap=MAX_ICA_COMPONENTS)
            print(f"Computing ICA with {n_components} components (cap={MAX_ICA_COMPONENTS})")
            ica = ICA(
                n_components=n_components,
                method='infomax',
                fit_params=dict(extended=True),
                max_iter=500,
                random_state=42
            )
            if FAST_ICA_FIT:
                tmax_fit = float(min(ICA_FIT_DURATION_SECONDS, raw.times[-1]))
                print(f"Fitting ICA on first {tmax_fit:.1f} seconds...")
                raw_fit = raw.copy().crop(tmin=0.0, tmax=tmax_fit)
                ica.fit(raw_fit, verbose=False)
            else:
                print("Fitting ICA on full dataset...")
                ica.fit(raw, verbose=False)
            print(f"ICA fit complete. Components: {ica.n_components_}")

            if VISUALIZE_STEPS:
                ica.plot_components(picks=range(min(12, ica.n_components_)), title="ICA Components")
                plt.show(block=True)
                ica.plot_sources(raw, start=0, stop=min(10, raw.times[-1]), title="ICA Sources")
                plt.show(block=True)

            # IC classification
            if ICALABEL_AVAILABLE:
                print("Classifying components with ICLabel...")
                to_exclude, iclabel_info = classify_components_iclabel(raw, ica, print_diagnostics=True)
                print(f"ICLabel exclude (thresholded): {to_exclude}")
            else:
                print("ICLabel unavailable, using fallback heuristic...")
                to_exclude = classify_components_fallback(raw, ica)
                print(f"Heuristic exclude: {to_exclude}")

            # Add EOG/ECG heuristics
            try:
                eog_inds, _ = ica.find_bads_eog(raw, ch_name=None)
                if len(eog_inds):
                    print(f"EOG-based exclude: {eog_inds}")
                    to_exclude = sorted(set(to_exclude).union(eog_inds))
            except Exception:
                pass
            try:
                ecg_inds, _ = ica.find_bads_ecg(raw, method='correlation')
                if len(ecg_inds):
                    print(f"ECG-based exclude: {ecg_inds}")
                    to_exclude = sorted(set(to_exclude).union(ecg_inds))
            except Exception:
                pass

            print(f"Final components to exclude: {to_exclude}")

            ica.exclude = to_exclude
            print(f"Applying ICA (removing {len(to_exclude)} components)...")
            raw_clean = ica.apply(raw.copy())

            # Restore annotations after ICA
            raw_clean.set_annotations(orig_annotations)

            # Save
            save_ica_outputs(opt, pid, raw_clean, ica, to_exclude)

            # Debug annotation labels
            desc_list = [a['description'] for a in raw_clean.annotations] if raw_clean.annotations is not None else []
            print("Total annotations on cleaned:", len(desc_list))
            for d, c in Counter(desc_list).most_common(20):
                print(f"{c:5d} {repr(d)}")

            # Build mapping and filter annotations to task labels
            task_name = opt.get('task_name', 'DecisionMaking')
            event_id_map = build_event_id_for_task(task_name)

            if raw_clean.annotations is not None and len(raw_clean.annotations):
                keys = set(event_id_map.keys())
                kept_onsets, kept_durs, kept_desc = [], [], []
                for ann in raw_clean.annotations:
                    d = (ann["description"] or "").strip()
                    keep = None
                    if d in keys:
                        keep = d
                    else:
                        d_norm = d.strip()
                        if d_norm in keys:
                            keep = d_norm
                        elif d_norm.startswith("Stimulus/"):
                            d_short = d_norm.split("Stimulus/", 1)[1]
                            if d_short in keys:
                                keep = d_short
                    if keep is not None:
                        kept_onsets.append(float(ann["onset"]))
                        kept_durs.append(float(ann["duration"]))
                        kept_desc.append(keep)

                if len(kept_desc) == 0:
                    kept_ann = mne.Annotations([], [], [])
                    print("After mapping filter, no task annotations remained; skipping epoching.")
                else:
                    kept_ann = mne.Annotations(
                        onset=np.array(kept_onsets, float),
                        duration=np.array(kept_durs, float),
                        description=kept_desc
                    )
                    print(f"Kept {len(kept_desc)} mapped task annotations for epoching.")
                raw_clean.set_annotations(kept_ann)

            # Create events/epochs
            events, event_id_used = mne.events_from_annotations(raw_clean, event_id=event_id_map)
            print("Event mapping used:", event_id_used)
            print("n_events:", len(events))

            if len(events) == 0 or len(event_id_used) == 0:
                print("No events found on cleaned data after mapping. Skipping epoch creation.")
            else:
                epochs = mne.Epochs(
                    raw_clean,
                    events,
                    event_id=event_id_used,
                    tmin=EPOCH_TMIN,
                    tmax=EPOCH_TMAX,
                    baseline=EPOCH_BASELINE,
                    preload=True,
                    reject_by_annotation=REJECT_BY_ANNOTATION
                )
                print(epochs)
                epochs_fif = os.path.join(opt['preprocessing_path'], f"{pid}-epo.fif")
                epochs.save(epochs_fif, overwrite=True)
                print(f"Epochs saved to: {epochs_fif}")

            if SHOW_ICA_DONE_PLOTS:
                show_ica_done_plots(raw, raw_clean, ica, to_exclude)

            print(f"{pid} ICA complete. Excluded {len(to_exclude)} components.")

        except Exception as e:
            print(f"ERROR in ICA for {pid}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nICA PROCESSING COMPLETE!")

if __name__ == "__main__":
    main()
