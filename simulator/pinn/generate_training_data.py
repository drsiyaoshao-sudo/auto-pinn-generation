"""
generate_training_data.py
─────────────────────────
Synthetic PINN training dataset generator.
Bill reference: bill_data_config_v1 (RATIFIED 2026-04-03)
Constitutional authority: Article I (Physics First), Amendment 14 (progress logging),
                          Amendment 7 (three-strike rule).
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import date

import numpy as np

# ── Resolve project root and import walker_model ──────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulator"))

from walker_model import WalkerProfile, PROFILES, generate_imu_sequence  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH  = Path(__file__).parent / "data_config.json"
OUTPUT_DIR   = Path(__file__).parent / "training_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

SEED                = CFG["seed"]                  # 42
N_RANDOM            = CFG["n_random_profiles"]     # 500
SAMPLES_PER_PROFILE = CFG["samples_per_profile"]   # 208
TERRAIN_DIST        = CFG["terrain_distribution"]  # flat/slope/stairs probs
PARAM_DIST          = CFG["parameter_distributions"]
ANCHOR_NAMES        = CFG["anchor_profiles"]       # ["flat","bad_wear","stairs","slope"]

# Conditioning vector field order (10 fields)
INPUT_FIELDS = [
    "cadence_spm",
    "step_length_m",
    "vertical_oscillation_cm",
    "slope_deg",
    "stance_frac",
    "si_stance_true_pct",
    "mounting_offset_deg",
    "loose_fit_attenuation",
    "step_variability_ms",
    "terrain_int",
]
OUTPUT_FIELDS = ["ax_ms2", "ay_ms2", "az_ms2", "gx_dps", "gy_dps", "gz_dps"]

TERRAIN_INT = {"flat": 0, "slope": 1, "stairs": 2}
TERRAIN_NAMES = {0: "flat", 1: "slope", 2: "stairs"}


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_field(field_name: str, terrain: str, rng: np.random.Generator) -> float:
    """Sample a single field with resample-until-valid loop (max 20 attempts)."""
    spec = PARAM_DIST[field_name]

    # Terrain-conditional specs
    if spec["type"] == "terrain_conditional":
        spec = spec[terrain]

    lo = spec.get("min", -np.inf)
    hi = spec.get("max",  np.inf)

    for attempt in range(20):
        if spec["type"] == "fixed":
            return float(spec["value"])
        elif spec["type"] == "normal":
            v = float(rng.normal(spec["mu"], spec["sigma"]))
            clip_neg = spec.get("clip_negative_to_zero", False)
            if clip_neg:
                v = max(v, 0.0)
        elif spec["type"] == "uniform":
            v = float(rng.uniform(spec["min"], spec["max"]))
        else:
            raise ValueError(f"Unknown distribution type: {spec['type']}")

        if lo <= v <= hi:
            return v
        # else resample this field only

    # After 20 attempts: clamp and warn
    v = float(np.clip(v, lo, hi))
    warnings.warn(
        f"Field '{field_name}' on terrain='{terrain}' could not be sampled within "
        f"[{lo}, {hi}] in 20 attempts. Clamped to {v:.4f}.",
        stacklevel=2,
    )
    return v


def _sample_profile(idx: int, rng: np.random.Generator) -> WalkerProfile:
    """Sample a random WalkerProfile using terrain-conditional distributions."""
    # Sample terrain first
    terrain_keys = ["flat", "slope", "stairs"]
    probs = [TERRAIN_DIST[k] for k in terrain_keys]
    terrain = str(rng.choice(terrain_keys, p=probs))

    cadence                 = _sample_field("cadence_spm",             terrain, rng)
    step_length             = _sample_field("step_length_m",           terrain, rng)
    vert_osc                = _sample_field("vertical_oscillation_cm", terrain, rng)
    slope_deg               = _sample_field("slope_deg",               terrain, rng)
    stance_frac             = _sample_field("stance_frac",             terrain, rng)
    si_stance               = _sample_field("si_stance_true_pct",      terrain, rng)
    mounting_offset         = _sample_field("mounting_offset_deg",     terrain, rng)
    loose_fit               = _sample_field("loose_fit_attenuation",   terrain, rng)
    step_variability        = _sample_field("step_variability_ms",     terrain, rng)

    return WalkerProfile(
        name=f"random_{idx:04d}",
        terrain=terrain,
        cadence_spm=cadence,
        step_length_m=step_length,
        vertical_oscillation_cm=vert_osc,
        slope_deg=slope_deg,
        stance_frac=stance_frac,
        si_stance_true_pct=si_stance,
        mounting_offset_deg=mounting_offset,
        loose_fit_attenuation=loose_fit,
        step_variability_ms=step_variability,
    )


def profile_to_x_vector(profile: WalkerProfile) -> np.ndarray:
    """Convert a WalkerProfile to a 10-element float32 conditioning vector."""
    terrain_int = TERRAIN_INT[profile.terrain]
    return np.array([
        profile.cadence_spm,
        profile.step_length_m,
        profile.vertical_oscillation_cm,
        profile.slope_deg,
        profile.stance_frac,
        profile.si_stance_true_pct,
        profile.mounting_offset_deg,
        profile.loose_fit_attenuation,
        profile.step_variability_ms,
        terrain_int,
    ], dtype=np.float32)


def profile_to_dict(profile: WalkerProfile) -> dict:
    """Serialise a WalkerProfile to a JSON-safe dict."""
    return {
        "name": profile.name,
        "terrain": profile.terrain,
        "cadence_spm": profile.cadence_spm,
        "step_length_m": profile.step_length_m,
        "vertical_oscillation_cm": profile.vertical_oscillation_cm,
        "slope_deg": profile.slope_deg,
        "stance_frac": profile.stance_frac,
        "si_stance_true_pct": profile.si_stance_true_pct,
        "mounting_offset_deg": profile.mounting_offset_deg,
        "loose_fit_attenuation": profile.loose_fit_attenuation,
        "step_variability_ms": profile.step_variability_ms,
        "walking_speed_ms": profile.walking_speed_ms,
        "hs_impact_ms2": profile.hs_impact_ms2,
        "peak_angvel_dps": profile.peak_angvel_dps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step A — Generate 500 random profiles
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("GaitSense PINN training data generator")
print("Bill: bill_data_config_v1  |  Status: RATIFIED 2026-04-03")
print("=" * 60)
print()
print("Step A — Generating 500 random WalkerProfile objects (seed=42)...")

master_rng = np.random.default_rng(SEED)
random_profiles: list[WalkerProfile] = []
skip_log: list[dict] = []

for i in range(N_RANDOM):
    strikes = 0
    while strikes < 3:
        try:
            prof = _sample_profile(i, master_rng)
            random_profiles.append(prof)
            break
        except Exception as exc:
            strikes += 1
            warnings.warn(f"Profile {i} strike {strikes}: {exc}")
    else:
        skip_log.append({"profile_idx": i, "reason": "three-strike limit reached"})
        warnings.warn(f"Profile {i} skipped after 3 strikes (Amendment 7 escalation).")

    if (i + 1) % 50 == 0:
        pct = (i + 1) / N_RANDOM * 100
        print(f"  {pct:5.0f}% — {i+1} profiles sampled")

print(f"  Done. {len(random_profiles)} random profiles generated, {len(skip_log)} skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Step B — Anchor profiles (indices 500-503)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("Step B — Loading 4 anchor profiles from PROFILES dict...")

anchor_profiles: list[WalkerProfile] = []
for name in ANCHOR_NAMES:
    p = PROFILES[name]
    anchor_profiles.append(p)
    print(f"  anchor: {name}  terrain={p.terrain}  cadence={p.cadence_spm} spm")

# Save anchor profile JSON
anchor_json = {name: profile_to_dict(p) for name, p in zip(ANCHOR_NAMES, anchor_profiles)}
with open(OUTPUT_DIR / "anchor_profiles.json", "w") as f:
    json.dump(anchor_json, f, indent=2)
print("  Saved anchor_profiles.json")

# ─────────────────────────────────────────────────────────────────────────────
# Step C — Generate IMU sequences for all 504 profiles
# ─────────────────────────────────────────────────────────────────────────────

print()
print("Step C — Generating IMU sequences for 504 profiles (n_steps=1 → 208 samples each)...")

all_profiles = random_profiles + anchor_profiles  # indices 0-503
N_TOTAL = len(all_profiles)

X_all = np.zeros((N_TOTAL, 10), dtype=np.float32)
Y_all = np.zeros((N_TOTAL, SAMPLES_PER_PROFILE, 6), dtype=np.float32)

imu_error_log: list[dict] = []

for idx, prof in enumerate(all_profiles):
    seq_seed = idx * 1000 + 42
    seq_rng  = np.random.default_rng(seq_seed)

    strikes = 0
    success = False
    while strikes < 3:
        try:
            seq = generate_imu_sequence(prof, n_steps=1, rng=seq_rng)
            break
        except Exception as exc:
            strikes += 1
            seq_rng = np.random.default_rng(seq_seed + strikes * 999983)
            warnings.warn(f"IMU seq idx={idx} strike {strikes}: {exc}")
    else:
        imu_error_log.append({"profile_idx": idx, "reason": str(exc)})
        warnings.warn(f"IMU seq idx={idx} all 3 strikes failed — filling with zeros.")
        seq = np.zeros((SAMPLES_PER_PROFILE, 6), dtype=np.float32)

    # seq shape may be != SAMPLES_PER_PROFILE depending on timing variability
    # Pad or truncate to exactly SAMPLES_PER_PROFILE
    n_samples = seq.shape[0]
    if n_samples >= SAMPLES_PER_PROFILE:
        seq_fixed = seq[:SAMPLES_PER_PROFILE, :]
    else:
        # Pad by repeating last row
        pad = np.tile(seq[-1:, :], (SAMPLES_PER_PROFILE - n_samples, 1))
        seq_fixed = np.concatenate([seq, pad], axis=0)
    seq_fixed = seq_fixed.astype(np.float32)

    X_all[idx] = profile_to_x_vector(prof)
    Y_all[idx] = seq_fixed

    if (idx + 1) % 51 == 0 or (idx + 1) == N_TOTAL:
        pct = (idx + 1) / N_TOTAL * 100
        print(f"  {pct:5.1f}% — {idx+1}/{N_TOTAL} sequences done")

print(f"  X_all shape: {X_all.shape}   Y_all shape: {Y_all.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step D — Train/val/test split
# ─────────────────────────────────────────────────────────────────────────────

print()
print("Step D — Train/val/test split (70/15/15, seed=42)...")

# Anchor indices are always val-only
anchor_indices = list(range(len(random_profiles), N_TOTAL))  # [500,501,502,503]
random_indices = list(range(len(random_profiles)))           # [0..499]

split_rng = np.random.default_rng(SEED)
shuffled = np.array(random_indices, dtype=int)
split_rng.shuffle(shuffled)

n_train = 350
n_val   = 75
n_test  = 75
assert n_train + n_val + n_test == len(random_indices), "Split counts must sum to 500"

train_idx = shuffled[:n_train].tolist()
val_idx   = shuffled[n_train:n_train + n_val].tolist()
test_idx  = shuffled[n_train + n_val:].tolist()

# Val includes anchors
val_idx_full = val_idx + anchor_indices

print(f"  train: {len(train_idx)}  val: {len(val_idx_full)} ({len(val_idx)} random + {len(anchor_indices)} anchors)  test: {len(test_idx)}")

# ── Slice arrays ──────────────────────────────────────────────────────────────
X_train = X_all[train_idx]
Y_train = Y_all[train_idx]
X_val   = X_all[val_idx_full]
Y_val   = Y_all[val_idx_full]
X_test  = X_all[test_idx]
Y_test  = Y_all[test_idx]

t_vec   = np.linspace(0, 1, SAMPLES_PER_PROFILE, dtype=np.float32)  # (208,)

# ── Save anchor IMU sequences separately ─────────────────────────────────────
for i, name in enumerate(ANCHOR_NAMES):
    anchor_seq = Y_all[anchor_indices[i]]  # (208, 6)
    np.save(OUTPUT_DIR / f"anchor_{name}.npy", anchor_seq)
    print(f"  Saved anchor_{name}.npy  shape={anchor_seq.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step E — Save all .npy files
# ─────────────────────────────────────────────────────────────────────────────

print()
print("Step E — Saving .npy files to training_data/...")

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "t_train.npy", t_vec)
np.save(OUTPUT_DIR / "Y_train.npy", Y_train)

np.save(OUTPUT_DIR / "X_val.npy",   X_val)
np.save(OUTPUT_DIR / "t_val.npy",   t_vec)
np.save(OUTPUT_DIR / "Y_val.npy",   Y_val)

np.save(OUTPUT_DIR / "X_test.npy",  X_test)
np.save(OUTPUT_DIR / "t_test.npy",  t_vec)
np.save(OUTPUT_DIR / "Y_test.npy",  Y_test)

print("  All .npy files saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Step F — Write dataset_manifest.json
# ─────────────────────────────────────────────────────────────────────────────

print()
print("Step F — Writing dataset_manifest.json...")

# Count terrain distribution in training set
terrain_counts_train = {"flat": 0, "slope": 0, "stairs": 0}
for i in train_idx:
    t_int = int(X_all[i, 9])
    terrain_counts_train[TERRAIN_NAMES[t_int]] += 1

manifest = {
    "n_train": int(X_train.shape[0]),
    "n_val": int(X_val.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_anchors": len(ANCHOR_NAMES),
    "samples_per_profile": SAMPLES_PER_PROFILE,
    "X_shape_train": list(X_train.shape),
    "Y_shape_train": list(Y_train.shape),
    "X_shape_val": list(X_val.shape),
    "Y_shape_val": list(Y_val.shape),
    "X_shape_test": list(X_test.shape),
    "Y_shape_test": list(Y_test.shape),
    "input_fields": INPUT_FIELDS,
    "output_fields": OUTPUT_FIELDS,
    "seed": SEED,
    "terrain_counts_train": terrain_counts_train,
    "bill_data_config": "bill_data_config_v1",
    "bill_data_config_status": "RATIFIED 2026-04-03",
    "generated_by": "synthetic-data-generator",
    "date": str(date.today()),
    "n_profiles_skipped": len(skip_log) + len(imu_error_log),
    "skip_log": skip_log + imu_error_log,
}

with open(OUTPUT_DIR / "dataset_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("  Saved dataset_manifest.json")

# ─────────────────────────────────────────────────────────────────────────────
# Shape verification
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("Shape verification")
print("=" * 60)

expected = {
    "anchor_flat.npy":      (208, 6),
    "anchor_bad_wear.npy":  (208, 6),
    "anchor_stairs.npy":    (208, 6),
    "anchor_slope.npy":     (208, 6),
    "X_train.npy":          (350, 10),
    "t_train.npy":          (208,),
    "Y_train.npy":          (350, 208, 6),
    "X_val.npy":            (79, 10),
    "t_val.npy":            (208,),
    "Y_val.npy":            (79, 208, 6),
    "X_test.npy":           (75, 10),
    "t_test.npy":           (208,),
    "Y_test.npy":           (75, 208, 6),
}

all_ok = True
header = f"{'File':<25} {'Expected':<20} {'Actual':<20} {'Status'}"
print(header)
print("-" * len(header))

for fname, exp_shape in expected.items():
    fpath = OUTPUT_DIR / fname
    if not fpath.exists():
        print(f"  {fname:<25} {str(exp_shape):<20} {'MISSING':<20} FAIL")
        all_ok = False
        continue
    arr = np.load(fpath)
    actual_shape = arr.shape
    ok = actual_shape == exp_shape
    if not ok:
        all_ok = False
    status = "OK" if ok else "SHAPE MISMATCH"
    print(f"  {fname:<25} {str(exp_shape):<20} {str(actual_shape):<20} {status}")

print()
if all_ok:
    print("All shapes verified. Dataset generation complete.")
else:
    print("ERROR: One or more shape mismatches detected. See above.")
    sys.exit(1)

print()
print("Manifest summary:")
print(f"  n_train={manifest['n_train']}  n_val={manifest['n_val']}  n_test={manifest['n_test']}")
print(f"  terrain_counts_train: {manifest['terrain_counts_train']}")
print(f"  n_profiles_skipped: {manifest['n_profiles_skipped']}")
print(f"  date: {manifest['date']}")
