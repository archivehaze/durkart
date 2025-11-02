# Imports 
import re
import numpy as np
import pandas as pd
import itertools
import random
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error


TOP_N_BODIES = None
TOP_N_TIRES  = None
TARGET_MAP   = "Rainbow Road" 

track_df = pd.read_csv("mario kart 8 deluxe track terrains - Sheet1.csv")

CANON = {
    "map": "Map",
    "wr": "WR",
    "total": "Total",
    "totals": "Total", 
    "turns": "Turns",
    "turn": "Turns",
    "ground": "Ground",
    "water": "Water",
    "air": "Air",
    "anti-gravity": "Anti-Gravity",
    "anti gravity": "Anti-Gravity",
    "anti–gravity": "Anti-Gravity",
    "anti—gravity": "Anti-Gravity",
    "anti − gravity": "Anti-Gravity",
    "anti - gravity": "Anti-Gravity",
}

def norm_token(s: str) -> str:
    s = str(s)
    s = s.replace("\u00A0", " ") 
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s) 
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def canonicalize_track_columns(df: pd.DataFrame) -> pd.DataFrame:
    buckets = {}
    for col in df.columns:
        key = norm_token(col)
        buckets.setdefault(key, []).append(col)

    out = pd.DataFrame(index=df.index)

    required = ["Map","WR","Ground","Water","Anti-Gravity","Air","Turns","Total"]

    for req in required:
        sources = []
        for k, cols in buckets.items():
            canon = CANON.get(k, k)  
            if canon == req:
                sources.extend(cols)

        if not sources:
            raise ValueError(f"Track CSV missing a column for '{req}' (variants tried).")

        if len(sources) == 1:
            series = df[sources[0]]
        else:
            series = df[sources].bfill(axis=1).iloc[:, 0]

        out[req] = series

    for k, cols in buckets.items():
        canon = CANON.get(k, k)
        if canon in out.columns:
            continue
        out[cols[0]] = df[cols[0]]

    return out

track_df = canonicalize_track_columns(track_df)


def parse_wr_to_seconds(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", s) 
    if m: return float(m.group(1))*60 + float(m.group(2))
    try:   return float(s)
    except ValueError: return np.nan

track_df["WR"] = track_df["WR"].apply(parse_wr_to_seconds)

for c in ["Ground","Water","Anti-Gravity","Air","Turns","Total","WR"]:
    track_df[c] = pd.to_numeric(track_df[c], errors="coerce")

wcols = ["Ground","Water","Anti-Gravity","Air"]
wsum = track_df[wcols].sum(axis=1).replace(0, 1.0)
track_df[wcols] = track_df[wcols].div(wsum, axis=0)

track_df["turn_density"]     = track_df["Turns"] / track_df["Total"].replace(0, np.nan)
track_df["air_ground_ratio"] = track_df["Air"]   / (track_df["Ground"] + 1e-6)

FEATURES = [
    "Ground","Water","Anti-Gravity","Air","Turns","Total","turn_density","air_ground_ratio"
]


ml_df = track_df.dropna(subset=FEATURES + ["WR"]).copy()
X = ml_df[FEATURES]
y = ml_df["WR"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

wr_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
wr_model.fit(X_train, y_train)

y_pred = wr_model.predict(X_test)
print("Holdout R²:", round(r2_score(y_test, y_pred), 4))
print("Holdout MAE (s):", round(mean_absolute_error(y_test, y_pred), 4))

print("\nWR feature importances:")
print(pd.Series(wr_model.feature_importances_, index=FEATURES).sort_values(ascending=False))

def remove_dupes(df):
    if df.columns.duplicated().any():

        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def safe_series(df, col, default=0.0):
    if col in df.columns:
        obj = df.loc[:, col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        return obj
    else:
        return pd.Series(default, index=df.index, dtype="float64")


def load_dataset(filepath):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "marlowspringmeier/mario-kart-8-deluxe-ingame-statistics",
        filepath,
        pandas_kwargs={"sep": ";"}
    )

drivers_df = load_dataset("drivers.csv")
bodies_df  = load_dataset("bodies_karts.csv")
gliders_df = load_dataset("gliders.csv")
tires_df   = load_dataset("tires.csv")

base_stats = [

    "Weight","Acceleration","On-Road traction","Off-Road Traction","Mini-Turbo",
    "Ground Speed","Water Speed","Anti-Gravity Speed","Air Speed",
    "Ground Handling","Water Handling","Anti-Gravity Handling","Air Handling"
]

def prefix_stats(df, key_col, prefix):
    rename = {c: f"{prefix}{c}" for c in df.columns if c != key_col and c in base_stats}
    return df.rename(columns=rename)

drivers_df = prefix_stats(drivers_df, "Driver", "drv_")
bodies_df  = prefix_stats(bodies_df,  "Body",   "body_")
gliders_df = prefix_stats(gliders_df, "Glider", "glider_")
tires_df   = prefix_stats(tires_df,   "Tire",   "tire_")

def _avg_speed(df, prefix):
    spd_cols = [c for c in df.columns if c.startswith(prefix) and "Speed" in c]
    return df.assign(_spd=df[spd_cols].mean(axis=1))

if TOP_N_BODIES:
    bodies_df = _avg_speed(bodies_df, "body_").nlargest(TOP_N_BODIES, "_spd").drop(columns="_spd")
if TOP_N_TIRES:
    tires_df  = _avg_speed(tires_df,  "tire_").nlargest(TOP_N_TIRES,  "_spd").drop(columns="_spd")

combos = itertools.product(
    drivers_df["Driver"].unique(),
    bodies_df["Body"].unique(),
    gliders_df["Glider"].unique(),
    tires_df["Tire"].unique()
)
builds = pd.DataFrame(list(combos), columns=["Driver","Body","Glider","Tire"])
builds = remove_dupes(builds)

builds = (builds
    .merge(drivers_df, on="Driver", how="left")
    .merge(bodies_df,  on="Body",   how="left")
    .merge(gliders_df, on="Glider", how="left")
    .merge(tires_df,   on="Tire",   how="left")
)

for stat in base_stats:
    cols = [f"drv_{stat}", f"body_{stat}", f"tire_{stat}", f"glider_{stat}"]
    cols = [c for c in cols if c in builds.columns]
    builds[f"TOTAL {stat}"] = builds[cols].sum(axis=1, numeric_only=True)

for c in ["Driver","Body","Glider","Tire"]:
    builds[c] = builds[c].astype("category")
float_cols = builds.select_dtypes(include=["float64","int64"]).columns
builds[float_cols] = builds[float_cols].astype("float32")

def predict_time_for_builds_turns(
    builds_df,
    tracks_df,
    c_turn=0.08,
    c_mt=0.005,
    c_acc=0.003,
    k_surface=None,
    clip_gain=(-0.15, 0.35)
):
    b   = remove_dupes(builds_df.copy())
    tdf = remove_dupes(tracks_df.copy())

    speed_cols  = ["TOTAL Ground Speed","TOTAL Water Speed","TOTAL Anti-Gravity Speed","TOTAL Air Speed"]
    handle_cols = ["TOTAL Ground Handling","TOTAL Water Handling","TOTAL Anti-Gravity Handling","TOTAL Air Handling"]

    for col in speed_cols + handle_cols:
        if col not in b.columns:
            b[col] = 0.0

    b[speed_cols + handle_cols] = (
        b[speed_cols + handle_cols]
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .fillna(0.0)
    )

    for col in ["TOTAL Mini-Turbo", "TOTAL Acceleration", "TOTAL Off-Road Traction"]:
        s = safe_series(b, col, default=0.0)
        b[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)

    for c in ["Ground","Water","Anti-Gravity","Air","Total","Turns","OffroadShare"]:
        if c in tdf.columns:
            t = safe_series(tdf, c, default=0.0)
            tdf[c] = pd.to_numeric(t, errors="coerce").fillna(0.0)

    wcols = ["Ground","Water","Anti-Gravity","Air"]
    row_sum = tdf[wcols].sum(axis=1).replace(0, 1.0)
    tdf[wcols] = tdf[wcols].div(row_sum, axis=0)

    if k_surface is None:
        k_surface = {"ground":1.0, "water":0.8, "ag":1.0, "air":0.25}

    H_ref = b[handle_cols].quantile(0.95).replace(0, np.nan)
    if H_ref.isna().any():
        H_ref = H_ref.fillna(H_ref.mean() or 1.0)
    MT_ref = max(float(b["TOTAL Mini-Turbo"].mean()), 1e-6)
    A_ref  = max(float(b["TOTAL Acceleration"].mean()), 1e-6)
    TR_ref = max(float(b["TOTAL Off-Road Traction"].mean()), 1e-6)

    tdf["_turns_per_s"] = tdf.apply(
        lambda r: (float(r.get("Turns", 0.0)) / float(r["Total"])) if float(r["Total"]) > 0 else 0.0,
        axis=1
    )
    turn_ref = np.nanpercentile(
        tdf["_turns_per_s"].replace([np.inf, -np.inf], np.nan).fillna(0.0),
        75
    )
    turn_ref = max(turn_ref, 1e-6)

    out = []
    for _, tr in tdf.iterrows():
        wg, ww, wag, wair = float(tr["Ground"]), float(tr["Water"]), float(tr["Anti-Gravity"]), float(tr["Air"])
        T  = float(tr.get("Turns", 0.0))
        offshare = float(tr.get("OffroadShare", 0.0))

        S_eff = (
            b["TOTAL Ground Speed"]*wg +
            b["TOTAL Water Speed"]*ww +
            b["TOTAL Anti-Gravity Speed"]*wag +
            b["TOTAL Air Speed"]*wair
        ).clip(lower=1e-6)

        Tg, Tw, Tag, Tair = T*wg, T*ww, T*wag, T*wair
        turns_per_s       = (T / float(tr["Total"])) if float(tr["Total"]) > 0 else 0.0
        turn_intensity    = np.clip(turns_per_s / turn_ref, 0.0, 2.0)

        Hg   = b["TOTAL Ground Handling"]       / max(float(H_ref["TOTAL Ground Handling"]), 1e-6)
        Hw   = b["TOTAL Water Handling"]        / max(float(H_ref["TOTAL Water Handling"]), 1e-6)
        Hag  = b["TOTAL Anti-Gravity Handling"] / max(float(H_ref["TOTAL Anti-Gravity Handling"]), 1e-6)
        Hair = b["TOTAL Air Handling"]          / max(float(H_ref["TOTAL Air Handling"]), 1e-6)

        handling_deficit = (
            k_surface["ground"] * Tg  * (1 - Hg)  +
            k_surface["water"]  * Tw  * (1 - Hw)  +
            k_surface["ag"]     * Tag * (1 - Hag) +
            k_surface["air"]    * Tair* (1 - Hair)
        )

        pen =  c_turn * turn_intensity * handling_deficit
        credit_mt  = c_mt  * (b["TOTAL Mini-Turbo"]   / MT_ref)
        credit_acc = c_acc * (b["TOTAL Acceleration"] / A_ref)
        traction_credit = 0.20 * offshare * ((b["TOTAL Off-Road Traction"] - TR_ref) / TR_ref)
        gamma = 1.0 + np.clip(pen - (credit_mt + credit_acc) - traction_credit, clip_gain[0], clip_gain[1])

        S_ref     = float(S_eff.mean())
        base_time = float(tr["Total"]) * (S_ref / S_eff)
        pred_time = base_time * gamma

        tmp = b[["Driver","Body","Tire","Glider"]].copy()
        tmp["Map"] = tr["Map"]
        tmp["predicted_time"] = pred_time.values
        out.append(tmp)

    return pd.concat(out, ignore_index=True)

def predict_time_for_single_build(build, *, use_ml_blend=True, alpha=0.15):
    if isinstance(build, (list, tuple)):
        if len(build) != 5:
            raise ValueError("Expected [Driver, Body, Tire, Glider, Map].")
        driver, body, tire, glider, map_name = build
    elif isinstance(build, dict):
        driver  = build["Driver"]; body = build["Body"]; tire = build["Tire"]; glider = build["Glider"]
        map_name = build["Map"]
    else:
        raise TypeError("build must be list/tuple of len 5 or dict with required keys")

    single = pd.DataFrame([{"Driver": driver, "Body": body, "Tire": tire, "Glider": glider}])

    single = (single
        .merge(drivers_df, on="Driver", how="left")
        .merge(bodies_df,  on="Body",   how="left")
        .merge(gliders_df, on="Glider", how="left")
        .merge(tires_df,   on="Tire",   how="left")
    )

    if single.isna().all(axis=1).any():
        missing = [c for c in ["Driver","Body","Tire","Glider"] if single.filter(like=c).isna().values.any()]
        raise ValueError(f"One or more part names didn’t match your datasets: {missing}")

    for stat in base_stats:
        cols = [f"drv_{stat}", f"body_{stat}", f"tire_{stat}", f"glider_{stat}"]
        cols = [c for c in cols if c in single.columns]
        single[f"TOTAL {stat}"] = single[cols].sum(axis=1, numeric_only=True)

    tdf = track_df.loc[track_df["Map"] == map_name].copy()
    if tdf.empty:
        raise ValueError(f"Map '{map_name}' not found in track_df['Map'].")

    preds = predict_time_for_builds_turns(single, tdf) 
    if "calibration" in globals():
        preds = preds.merge(calibration[["Map","scale"]], on="Map", how="left")
        preds["scale"] = preds["scale"].fillna(1.0)
    else:
        preds["scale"] = 1.0
    preds["predicted_time_calibrated"] = preds["predicted_time"] * preds["scale"]

    if use_ml_blend and "wr_model" in globals() and wr_model is not None:
        feats_row = track_df.loc[track_df["Map"] == map_name, FEATURES]
        if not feats_row.empty:
            ml_wr = float(wr_model.predict(feats_row.iloc[[0]])[0])
            preds["predicted_time_calibrated"] = (
                (1.0 - alpha) * preds["predicted_time_calibrated"] + alpha * ml_wr
            )
            wr_val = float(track_df.loc[track_df["Map"] == map_name, "WR"].iloc[0])
            preds["predicted_time_calibrated"] = np.maximum(
                preds["predicted_time_calibrated"], 0.98 * wr_val
            )
    phrases = ["Mario", "Wahoo!", "Let's-a go!", "It's-a me", "Oh yeah!", "Boing!", "Here we go!"]
    
    return f"{random.choice(phrases)},  {build["Driver"]} has a predicted time of {preds["predicted_time_calibrated"]} {random.choice(phrases)}"


def pick_single_winner_with_rules(preds, builds_df, track_df):
    id_cols = ["Driver","Body","Tire","Glider"]
    needed_stats = [
        "TOTAL Acceleration","TOTAL Mini-Turbo",
        "TOTAL Ground Handling","TOTAL Water Handling","TOTAL Anti-Gravity Handling","TOTAL Air Handling",
        "TOTAL Weight",
    ]

    enriched = (remove_dupes(preds)
        .merge(remove_dupes(builds_df[id_cols + needed_stats]), on=id_cols, how="left", validate="many_to_one")
        .merge(remove_dupes(track_df[["Map","Ground","Water","Anti-Gravity","Air"]]), on="Map", how="left")
    )

    enriched["predicted_time"] = pd.to_numeric(safe_series(enriched, "predicted_time", default=0.0), errors="coerce").fillna(0.0)
    for c in needed_stats:
        enriched[c] = pd.to_numeric(safe_series(enriched, c, default=0.0), errors="coerce").fillna(0.0)

    wsum = enriched[["Ground","Water","Anti-Gravity","Air"]].sum(axis=1).replace(0, 1.0)
    for c in ["Ground","Water","Anti-Gravity","Air"]:
        enriched[c] = pd.to_numeric(safe_series(enriched, c, default=0.0), errors="coerce").fillna(0.0)
        enriched[c] = enriched[c] / wsum

    enriched["TOTAL Handling Weighted"] = (
        enriched["TOTAL Ground Handling"]*enriched["Ground"] +
        enriched["TOTAL Water Handling"]*enriched["Water"] +
        enriched["TOTAL Anti-Gravity Handling"]*enriched["Anti-Gravity"] +
        enriched["TOTAL Air Handling"]*enriched["Air"]
    )

    winner = (enriched.sort_values(
        by=["predicted_time","TOTAL Acceleration","TOTAL Mini-Turbo","TOTAL Handling Weighted","TOTAL Weight",
            "Driver","Body","Tire","Glider"],
        ascending=[ True,           False,              False,                 False,                 True,
                    True,  True, True, True]
    ).iloc[0])
    return winner



sim_preds = predict_time_for_builds_turns(builds, track_df)

sim_means = (sim_preds.groupby("Map", as_index=False)["predicted_time"]
                        .mean().rename(columns={"predicted_time":"sim_mean"}))

calibration = (track_df[["Map","WR"]]
               .merge(sim_means, on="Map", how="left"))

calibration["scale"] = calibration["WR"] / calibration["sim_mean"].replace({0: np.nan})
calibration["scale"] = calibration["scale"].replace([np.inf,-np.inf], np.nan).fillna(1.0)

sim_preds = sim_preds.merge(calibration[["Map","scale"]], on="Map", how="left")
sim_preds["predicted_time_calibrated"] = sim_preds["predicted_time"] * sim_preds["scale"]

if 'wr_model' != None:
    feats_row = track_df.loc[track_df["Map"] == TARGET_MAP, FEATURES]
    if not feats_row.empty:
        ml_wr = float(wr_model.predict(feats_row.iloc[[0]])[0])  
        alpha = 0.15  
        m = (sim_preds["Map"] == TARGET_MAP)

        sim_preds.loc[m, "predicted_time_calibrated"] = (
            (1.0 - alpha) * sim_preds.loc[m, "predicted_time_calibrated"] + alpha * ml_wr
        )

        wr_val = float(track_df.loc[track_df["Map"] == TARGET_MAP, "WR"].iloc[0])
        sim_preds.loc[m, "predicted_time_calibrated"] = np.maximum(
            sim_preds.loc[m, "predicted_time_calibrated"], 0.98 * wr_val
        )

rr_preds = sim_preds.loc[sim_preds["Map"] == TARGET_MAP].copy()
if rr_preds.empty:
    raise ValueError(f"Track '{TARGET_MAP}' not found. "
                     f"Available: {sorted(sim_preds['Map'].unique().tolist())[:10]} ...")

rr_winner = pick_single_winner_with_rules(
    rr_preds.rename(columns={"predicted_time_calibrated": "predicted_time"}),
    builds, track_df
)




print(f"\n=== {TARGET_MAP} — Winner ===")
print(f"Driver: {rr_winner['Driver']}")
print(f"Body:   {rr_winner['Body']}")
print(f"Tire:   {rr_winner['Tire']}")
print(f"Glider: {rr_winner['Glider']}")
print(f"Pred (s): {rr_winner['predicted_time']:.2f}")
