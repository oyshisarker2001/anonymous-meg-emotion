import os
import glob
import numpy as np
import pandas as pd
import mne

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = "ds007640-download"   # change this
OUTPUT_DIR = "processed_meg_regression"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TMIN = 0.0
TMAX = 5.0         # change if needed
PRELOAD = True
NORMALIZE_LABELS = True

# =========================================================
# HELPERS
# =========================================================
def find_single_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if len(files) == 0:
        return None
    if len(files) > 1:
        print(f"[WARN] Multiple files found for pattern {pattern} in {folder}. Using first one.")
    return files[0]


def load_events_from_raw(raw):
    """
    Try annotations first, then stim channels.
    Returns:
        events: ndarray of shape (n_events, 3)
        event_id: dict or None
        method: str
    """
    # 1) annotations
    if raw.annotations is not None and len(raw.annotations) > 0:
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if len(events) > 0:
                return events, event_id, "annotations"
        except Exception as e:
            print(f"[WARN] events_from_annotations failed: {e}")

    # 2) stim channel fallback
    stim_candidates = []
    ch_names = raw.ch_names
    ch_types = raw.get_channel_types()

    for name, ctype in zip(ch_names, ch_types):
        if ctype == "stim" or "STI" in name.upper() or "TRIG" in name.upper():
            stim_candidates.append(name)

    for stim in stim_candidates:
        try:
            events = mne.find_events(raw, stim_channel=stim, verbose=False)
            if len(events) > 0:
                return events, None, f"stim:{stim}"
        except Exception as e:
            print(f"[WARN] mne.find_events failed on {stim}: {e}")

    return None, None, None


def choose_trial_events(events, expected_n_trials):
    """
    If multiple event IDs exist, choose the one whose count is closest
    to expected_n_trials. If only one event type exists, return all.
    """
    if events is None or len(events) == 0:
        return None

    unique_ids, counts = np.unique(events[:, 2], return_counts=True)

    if len(unique_ids) == 1:
        return events

    best_id = None
    best_diff = float("inf")

    for eid, cnt in zip(unique_ids, counts):
        diff = abs(cnt - expected_n_trials)
        if diff < best_diff:
            best_diff = diff
            best_id = eid

    chosen = events[events[:, 2] == best_id]
    print(f"[INFO] Selected event_id={best_id} with {len(chosen)} events (expected ~{expected_n_trials})")
    return chosen


def normalize_targets(y):
    """
    Normalize SAM scores from 1..9 to 0..1
    """
    return (y - 1.0) / 8.0


def process_subject_session(sub_dir, ses_dir, output_dir):
    subject = os.path.basename(sub_dir)
    session = os.path.basename(ses_dir)

    meg_dir = os.path.join(ses_dir, "meg")
    beh_dir = os.path.join(ses_dir, "beh")

    if not os.path.isdir(meg_dir) or not os.path.isdir(beh_dir):
        print(f"[SKIP] Missing meg/ or beh/ for {subject} {session}")
        return None

    meg_file = find_single_file(meg_dir, "*_meg.fif")
    beh_file = find_single_file(beh_dir, "*_beh.tsv")

    if meg_file is None or beh_file is None:
        print(f"[SKIP] Missing MEG or BEH file for {subject} {session}")
        return None

    print(f"\n[PROCESSING] {subject} | {session}")
    print(f"  MEG: {os.path.basename(meg_file)}")
    print(f"  BEH: {os.path.basename(beh_file)}")

    # -------------------------
    # Load labels
    # -------------------------
    labels = pd.read_csv(beh_file, sep="\t")

    required_cols = ["SAM_valence", "SAM_arousal"]
    missing_cols = [c for c in required_cols if c not in labels.columns]
    if missing_cols:
        print(f"[SKIP] Missing required label columns {missing_cols} in {beh_file}")
        return None

    y = labels[required_cols].to_numpy(dtype=np.float32)

    if NORMALIZE_LABELS:
        y = normalize_targets(y)

    expected_n_trials = len(y)

    # -------------------------
    # Load raw MEG
    # -------------------------
    try:
        raw = mne.io.read_raw_fif(meg_file, preload=PRELOAD, verbose=False)
    except Exception as e:
        print(f"[SKIP] Could not load raw MEG for {subject} {session}: {e}")
        return None

    # -------------------------
    # Extract events
    # -------------------------
    events, event_id, method = load_events_from_raw(raw)
    if events is None:
        print(f"[SKIP] No events found for {subject} {session}")
        return None

    print(f"  Event extraction method: {method}")
    print(f"  Total events found: {len(events)}")

    # Select likely trial-start events
    trial_events = choose_trial_events(events, expected_n_trials)
    if trial_events is None or len(trial_events) == 0:
        print(f"[SKIP] No usable trial events for {subject} {session}")
        return None

    # -------------------------
    # Build epochs
    # -------------------------
    try:
        epochs = mne.Epochs(
            raw,
            trial_events,
            event_id=None,
            tmin=TMIN,
            tmax=TMAX,
            baseline=None,
            preload=True,
            verbose=False
        )
        X = epochs.get_data().astype(np.float32)   # (n_trials, n_channels, n_times)
    except Exception as e:
        print(f"[SKIP] Failed to epoch {subject} {session}: {e}")
        return None

    # -------------------------
    # Align X and y
    # -------------------------
    n_x = len(X)
    n_y = len(y)
    n = min(n_x, n_y)

    if n == 0:
        print(f"[SKIP] Empty data after epoching for {subject} {session}")
        return None

    if n_x != n_y:
        print(f"[WARN] Mismatch for {subject} {session}: X={n_x}, y={n_y}. Trimming to {n}.")

    X = X[:n]
    y = y[:n]
    labels_trimmed = labels.iloc[:n].reset_index(drop=True)

    # -------------------------
    # Save
    # -------------------------
    save_prefix = f"{subject}_{session}"
    np.save(os.path.join(output_dir, f"{save_prefix}_X.npy"), X)
    np.save(os.path.join(output_dir, f"{save_prefix}_y.npy"), y)
    labels_trimmed.to_csv(os.path.join(output_dir, f"{save_prefix}_labels.csv"), index=False)

    meta = {
        "subject": subject,
        "session": session,
        "meg_file": meg_file,
        "beh_file": beh_file,
        "event_method": method,
        "n_events_total": int(len(events)),
        "n_trial_events_used": int(len(trial_events)),
        "X_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
        "tmin": TMIN,
        "tmax": TMAX,
        "normalized_labels": NORMALIZE_LABELS,
    }

    pd.DataFrame([meta]).to_csv(
        os.path.join(output_dir, f"{save_prefix}_meta.csv"),
        index=False
    )

    print(f"  Saved: {save_prefix}_X.npy, {save_prefix}_y.npy")
    print(f"  Final X shape: {X.shape}")
    print(f"  Final y shape: {y.shape}")

    return meta


# =========================================================
# MAIN LOOP: ALL SUBJECTS, ALL SESSIONS
# =========================================================
all_meta = []

subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))

for sub_dir in subject_dirs:
    session_dirs = sorted(glob.glob(os.path.join(sub_dir, "ses-*")))

    if len(session_dirs) == 0:
        print(f"[SKIP] No sessions found in {sub_dir}")
        continue

    for ses_dir in session_dirs:
        meta = process_subject_session(sub_dir, ses_dir, OUTPUT_DIR)
        if meta is not None:
            all_meta.append(meta)
            

# Save summary
if len(all_meta) > 0:
    summary_df = pd.DataFrame(all_meta)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_all_subjects_sessions.csv"), index=False)
    print(f"\n[DONE] Processed {len(all_meta)} subject-session pairs.")
    print(summary_df[["subject", "session", "X_shape", "y_shape", "event_method"]].head())
else:
    print("\n[DONE] No subject-session pairs were successfully processed.")