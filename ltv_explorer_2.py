# ltv_explorer.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import re
from urllib.parse import urlparse, parse_qs
import io, requests
import os, tempfile

def _read_csv_like(fobj, COLS):
    fobj.seek(0)
    header = pd.read_csv(fobj, nrows=0)
    fobj.seek(0)
    parse_dates = [c for c in [COLS["created"], COLS["cancel_ts"]] if c in header.columns]
    dtype_overrides = {
        COLS["trainer"]: "string",
        COLS["gender"]: "string",
        COLS["region"]: "string",
        COLS["billing"]: "string",
        COLS["price_tier"]: "string",
    }
    df = pd.read_csv(
        fobj,
        parse_dates=parse_dates,
        dtype={k: v for k, v in dtype_overrides.items() if k in header.columns},
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "NULL"],
        low_memory=False
    )
    df.columns = df.columns.map(lambda c: c.strip())
    return df

def normalize_csv_url(u: str) -> str:
    if not isinstance(u, str):
        return u
    u = u.strip()
    if not u:
        return u

    # /file/d/<ID>/... → direct download
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    # any drive link with ?id=<ID> (e.g. open?id=..., uc?id=..., view?id=...)
    if "drive.google.com" in u:
        parsed = urlparse(u)
        qs = parse_qs(parsed.query)
        fid = (qs.get("id") or [None])[0]
        if fid:
            return f"https://drive.google.com/uc?export=download&id={fid}"

    # drive.usercontent.google.com/download?id=<ID> → normalize
    if "drive.usercontent.google.com" in u:
        qs = parse_qs(urlparse(u).query)
        fid = (qs.get("id") or [None])[0]
        if fid:
            return f"https://drive.google.com/uc?export=download&id={fid}"

    # Google Sheets → CSV (first sheet unless &gid specified)
    m = re.search(r"docs\.google\.com/spreadsheets/d/([^/]+)/", u)
    if m:
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"

    # Dropbox share → force download
    if "dropbox.com" in u:
        return u.replace("?dl=0", "?dl=1")

    return u

# --- move this above load_data to remove any doubt ---
def extract_gdrive_id(u: str) -> str | None:
    if not isinstance(u, str):
        return None
    u = u.strip()
    if not u:
        return None
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", u)
    if m:
        return m.group(1)
    qs = parse_qs(urlparse(u).query)
    fid = (qs.get("id") or [None])[0]
    if fid:
        return fid
    return None
# -----------------------------------------------------

st.set_page_config(page_title="LTV Explorer", layout="wide")

# -----------------------------
# Sidebar: data & global options
# -----------------------------
st.sidebar.header("Data")
csv_path = st.sidebar.text_input(
    "CSV path or HTTPS URL",
    value=st.secrets.get(
        "DEFAULT_CSV_URL",
        "https://ltv-data-andrew-2025.s3.us-east-1.amazonaws.com/subs.csv"
    ),
    placeholder="Paste a s3 link to subs csv…",
    help="Local path or a direct HTTPS link to a CSV."
)

# Created-since date filter
created_since = st.sidebar.date_input(
    "Only include Subs created since",
    value=date(2019, 1, 1),
    help="Rows with created < this date are excluded before grouping."
)

# LTV horizon (drives rev column & months_active metric)
horizon = st.sidebar.selectbox(
    "LTV horizon",
    options=[6, 12, 24],
    index=1,
    help="We include rows with non-null revenue_Xm_usd for the chosen horizon."
)

# Top creators count
top_n = st.sidebar.number_input(
    "Top creators to display (per view)", min_value=10, max_value=200, value=50, step=10
)

# Min subs per creator per group to show value
min_subs = st.sidebar.number_input(
    "Minimum subs per creator per group to show a value",
    min_value=1, max_value=1000, value=50, step=5,
    help="If a creator has fewer than this number of subscriptions in a group, that cell is hidden (only subs count is shown)."
)

# Single metric selector for the table (default to Mean LTV)
metric_options = ["Mean LTV", "Median LTV", "Sum LTV", "Avg Months Active", "Subs"]
metric_choice = st.sidebar.selectbox(
    "Displayed metric (per group)",
    options=metric_options,
    index=0,
    help="Which metric to display across groups in the top table."
)

# Column mapping to your CSV (must align with SQL export)
COLS = dict(
    id="id",
    trainer="trainer_name",
    gender="gender",
    region="region",
    billing="billing_cycle",
    price_tier="price_tier",
    price="usd_mrr_equiv_adjusted",
    trial_flag="trial_flag",
    trial_converted="trial_converted",
    rev6="revenue_6m_usd",
    rev12="revenue_12m_usd",
    rev24="revenue_24m_usd",
    months6="months_active_6m",
    months12="months_active_12m",
    months24="months_active_24m",
    created="created",
    cancel_ts="cancel_ts",
    trial_days="trial_period_days",
)

# ------------------------------------------------
# Load CSV (local path or HTTPS). Cache for speed.
# ------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(path_or_url: str) -> pd.DataFrame:
    def _looks_like_html_bytes(b: bytes) -> bool:
        sniff = b[:4096].lstrip().lower()
        return (
            sniff.startswith(b"<!doctype html")
            or sniff.startswith(b"<html")
            or b"<body" in sniff
            or b"google drive" in sniff
            or b"download anyway" in sniff
            or b"quota exceeded" in sniff
            or b"sign in" in sniff
        )

    def _read_from_path(csv_path_local: str) -> pd.DataFrame:
        header = pd.read_csv(csv_path_local, nrows=0)
        parse_dates = [c for c in [COLS["created"], COLS["cancel_ts"]] if c in header.columns]
        dtype_overrides = {
            COLS["trainer"]: "string",
            COLS["gender"]: "string",
            COLS["region"]: "string",
            COLS["billing"]: "string",
            COLS["price_tier"]: "string",
        }
        df = pd.read_csv(
            csv_path_local,
            parse_dates=parse_dates,
            dtype={k: v for k, v in dtype_overrides.items() if k in header.columns},
            keep_default_na=True,
            na_values=["", "NA", "N/A", "null", "NULL"],
            low_memory=False,
        )
        df.columns = df.columns.map(lambda c: c.strip())
        return df

    # Robust: download to temp file to avoid huge in-memory buffers
    def _download_to_tempfile(url: str) -> str:
        # First small peek to detect HTML; then stream rest to disk
        with requests.get(
            url,
            stream=True,
            timeout=180,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"},
        ) as r:
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            # Peek a small chunk
            try:
                first_chunk = next(r.iter_content(chunk_size=8192))
            except StopIteration:
                first_chunk = b""

            if "text/html" in ct or _looks_like_html_bytes(first_chunk):
                raise ValueError("HTML interstitial detected")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
                tmp_path = tf.name
                if first_chunk:
                    tf.write(first_chunk)
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        tf.write(chunk)
        return tmp_path

    def _gdown_to_tempfile(any_url: str) -> str:
        import gdown
        fid = extract_gdrive_id(any_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
            tmp_path = tf.name
        try:
            if fid:
                gdown.download(id=fid, output=tmp_path, quiet=True)
            else:
                gdown.download(any_url, tmp_path, quiet=True, fuzzy=True)
            return tmp_path
        except Exception as e:
            # Clean up if gdown failed
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"gdown failed: {e}")

    # ---- URL case ----
    if isinstance(path_or_url, str) and path_or_url.lower().startswith(("http://", "https://")):
        url = path_or_url
        tmp_path = None
        try:
            # Try direct streaming download
            try:
                tmp_path = _download_to_tempfile(url)
            except ValueError:
                # HTML detected → use gdown
                tmp_path = _gdown_to_tempfile(url)

            # Parse CSV from disk
            df = _read_from_path(tmp_path)
        except Exception as e:
            # Last-ditch: gdown even if the first attempt wasn't HTML
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            try:
                tmp_path = _gdown_to_tempfile(url)
                df = _read_from_path(tmp_path)
            except Exception as ge:
                raise RuntimeError(
                    "Failed to obtain a CSV from Google Drive. "
                    "Ensure the file is shared as **Anyone with the link (Viewer)**. "
                    f"(detail: {e}; gdown retry: {ge})"
                )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    else:
        # ---- Local path ----
        df = _read_from_path(path_or_url)

    # ---- Numeric coercions & trial flags (unchanged) ----
    for c in [COLS["price"], COLS["rev6"], COLS["rev12"], COLS["rev24"],
              COLS["months6"], COLS["months12"], COLS["months24"]]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if COLS["trial_days"] in df.columns:
        df[COLS["trial_days"]] = pd.to_numeric(df[COLS["trial_days"]], errors="coerce")
    if COLS["trial_flag"] not in df.columns and COLS["trial_days"] in df.columns:
        df[COLS["trial_flag"]] = (df[COLS["trial_days"]].fillna(0) > 0).astype(int)
    if COLS["trial_converted"] in df.columns:
        df[COLS["trial_converted"]] = pd.to_numeric(df[COLS["trial_converted"]], errors="coerce")

    return df

csv_path_norm = normalize_csv_url(csv_path)
if not csv_path_norm:
    st.info("Paste a CSV link (Drive/Sheets/Dropbox) or provide a local path.")
    st.stop()

# Button-gated load so the app renders fast, then downloads on demand
if "data_token" not in st.session_state:
    st.session_state.data_token = 0

load_now = st.sidebar.button("Load / reload data", type="primary", use_container_width=True)
if load_now:
    st.session_state.data_token += 1

if st.session_state.data_token == 0:
    st.info("Set the CSV link, then click **Load / reload data** to fetch the dataset.")
    st.stop()

try:
    df_raw = load_data(csv_path_norm)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# -----------------
# Base cleaning step
# -----------------
df_base = df_raw.copy()
if COLS["trainer"] not in df_base.columns:
    st.error(
        f"Expected column '{COLS['trainer']}' not found. "
        "This usually means the URL didn’t return a CSV (e.g., a Google login/HTML page). "
        "Fix by: (1) Share the file as 'Anyone with link – Viewer', "
        "and (2) paste any standard Drive link; the app will convert it."
    )
    st.stop()

# Normalize categoricals early; collapse blanks to "Unknown"
for cat_col in [COLS["gender"], COLS["region"], COLS["billing"], COLS["price_tier"], COLS["trainer"]]:
    if cat_col in df_base.columns:
        s = df_base[cat_col].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        df_base[cat_col] = s.fillna("Unknown")

# ---- Master creator list (does NOT depend on created_since) ----
all_creators_master = sorted(df_base[COLS["trainer"]].dropna().astype(str).unique().tolist())

# Apply “created since” and other row filters to produce the working frame
df0 = df_base.copy()
if COLS["created"] in df0.columns and pd.api.types.is_datetime64_any_dtype(df0[COLS["created"]]):
    df0 = df0[df0[COLS["created"]] >= pd.to_datetime(created_since)]

# Always exclude price == 0 / null
df0 = df0[df0[COLS["price"]].notna() & (df0[COLS["price"]] > 0)]

# Horizon eligibility
rev_col = {6: COLS["rev6"], 12: COLS["rev12"], 24: COLS["rev24"]}[horizon]
df0 = df0[df0[rev_col].notna()].copy()

# ---------- Stable option lists for group filters ----------
def distinct_sorted(series):
    return sorted(series.dropna().astype(str).unique().tolist())

gender_opts_base = distinct_sorted(df0[COLS["gender"]])
region_opts_base = distinct_sorted(df0[COLS["region"]])
cycle_opts_base  = distinct_sorted(df0[COLS["billing"]])
tier_opts_base   = distinct_sorted(df0[COLS["price_tier"]])

# Clamp a saved list in session to current options (avoid default-not-in-options)
def clamp_list(key: str, options: list[str]) -> list[str]:
    vals = st.session_state.get(key, options.copy())
    if isinstance(vals, (set, tuple)): vals = list(vals)
    vals = [v for v in vals if v in options]
    if not vals:
        vals = options.copy()
    st.session_state[key] = vals
    return vals

# -------------------------
# Global creator selection
# -------------------------
st.header("LTV Explorer")
st.markdown("#### Creators filter")

if "selected_creators" not in st.session_state:
    st.session_state.selected_creators = all_creators_master.copy()

colL, colR = st.columns([3,1])
with colR:
    if st.button("Select all"):
        st.session_state.selected_creators = all_creators_master.copy()
    if st.button("Clear all"):
        st.session_state.selected_creators = []

selected_creators = colL.multiselect(
    "Creators to include (global)",
    options=all_creators_master,
    key="selected_creators",
    help="Type to search. Use Select all / Clear all to bulk update."
)

if len(selected_creators) == 0:
    st.warning("No creators selected. Use **Select all** or choose one or more creators to proceed.")
    st.stop()

if "restrict_all" not in st.session_state:
    st.session_state.restrict_all = True
restrict_all = st.checkbox(
    f"Restrict creators to those present in **all active groups** with subs ≥ {min_subs}",
    key="restrict_all"
)

df = df0[df0[COLS["trainer"]].isin(selected_creators)]

# -----------------------------------
# Group toggles + names + filters UI
# -----------------------------------
st.markdown("#### Groups")
cA1, cA2 = st.columns([0.18, 0.82])
cA1.write("Group A")
name_A = cA2.text_input("Name for Group A", value="Weekly", key="name_A", label_visibility="collapsed")

cB1, cB2 = st.columns([0.18, 0.82])
show_B = cB1.checkbox("Show Group B", value=True, key="show_B")
name_B = cB2.text_input("Name for Group B", value="Monthly", key="name_B", label_visibility="collapsed") if show_B else "Group B"

cC1, cC2 = st.columns([0.18, 0.82])
show_C = cC1.checkbox("Show Group C", value=True, key="show_C")
name_C = cC2.text_input("Name for Group C", value="Quarterly", key="name_C", label_visibility="collapsed") if show_C else "Group C"

cD1, cD2 = st.columns([0.18, 0.82])
show_D = cD1.checkbox("Show Group D", value=False, key="show_D")
name_D = cD2.text_input("Name for Group D", value="Annual", key="name_D", label_visibility="collapsed") if show_D else "Group D"

active_labels = [name_A] + ([name_B] if show_B else []) + ([name_C] if show_C else []) + ([name_D] if show_D else [])
norm = [lbl.strip() for lbl in active_labels]
dupes = sorted({lbl for lbl in norm if norm.count(lbl) > 1})
if dupes:
    st.error(
        "Group names must be unique. "
        f"Duplicated name(s): {dupes}. Please change one of the group labels."
    )
    st.stop()

def group_banner(label: str, code: str):
    colors = {"A":"#eaf2ff", "B":"#e3edff", "C":"#dce8ff", "D":"#d5e3ff"}
    borders = {"A":"#6ea8fe", "B":"#639bff", "C":"#5991ff", "D":"#4f86ff"}
    st.markdown(
        f"<div class='grp-banner grp-{code}' "
        f"style='background:{colors[code]};border-left:4px solid {borders[code]};"
        f"padding:6px 10px;border-radius:6px;margin-top:6px;margin-bottom:-6px;'>"
        f"<b>{label}</b></div>",
        unsafe_allow_html=True
    )

def group_filter_ui(code: str, label: str, visible: bool):
    if not visible:
        return None, 0
    kG, kR, kC, kT, kTrial = f"gender_{code}", f"region_{code}", f"cycle_{code}", f"tier_{code}", f"trials_{code}"
    st.session_state[kG] = clamp_list(kG, gender_opts_base)
    st.session_state[kR] = clamp_list(kR, region_opts_base)
    st.session_state[kC] = clamp_list(kC, cycle_opts_base)
    st.session_state[kT] = clamp_list(kT, tier_opts_base)
    if kTrial not in st.session_state:
        st.session_state[kTrial] = False

    group_banner(f"Filters — {label}", code)
    with st.expander("", expanded=False):
        if st.button(f"Reset filters for {label}", key=f"reset_{code}"):
            st.session_state[kG] = gender_opts_base.copy()
            st.session_state[kR] = region_opts_base.copy()
            st.session_state[kC] = cycle_opts_base.copy()
            st.session_state[kT] = tier_opts_base.copy()
            st.session_state[kTrial] = False

        g  = st.multiselect(f"Gender ({label})", gender_opts_base, key=kG)
        r  = st.multiselect(f"Region ({label})", region_opts_base, key=kR)
        bc = st.multiselect(f"Billing cycle ({label})", cycle_opts_base,  key=kC)
        pt = st.multiselect(f"Price tier ({label})",  tier_opts_base,   key=kT)

        include_converted_trials = st.checkbox(
            f"Include converted trials in {label} (otherwise exclude all trials)",
            value=st.session_state[kTrial], key=kTrial,
            help="If ON: only include rows where trial_converted = 1. If OFF: exclude any with trial_period_days > 0."
        )

    filters = dict(genders=st.session_state[kG], regions=st.session_state[kR],
                   cycles=st.session_state[kC], tiers=st.session_state[kT])
    return filters, (1 if include_converted_trials else 0)

def apply_group_filters(dfin: pd.DataFrame, flt, trials_mode):
    if flt is None:
        return pd.DataFrame(columns=dfin.columns)
    out = dfin[
        dfin[COLS["gender"]].isin(flt["genders"]) &
        dfin[COLS["region"]].isin(flt["regions"]) &
        dfin[COLS["billing"]].isin(flt["cycles"]) &
        dfin[COLS["price_tier"]].isin(flt["tiers"])
    ].copy()
    if COLS["trial_flag"] in out.columns:
        if trials_mode == 0:
            out = out[out[COLS["trial_flag"]].fillna(0) == 0]
        else:
            if COLS["trial_converted"] in out.columns:
                out = out[
                    (out[COLS["trial_flag"]].fillna(0) == 0) |
                    (out[COLS["trial_converted"]].fillna(0) == 1)
                ]
            else:
                out = out[out[COLS["trial_flag"]].fillna(0) == 0]
    return out

# Build group UIs
flt_A, tmode_A = group_filter_ui("A", name_A, True)
flt_B, tmode_B = group_filter_ui("B", name_B, show_B)
flt_C, tmode_C = group_filter_ui("C", name_C, show_C)
flt_D, tmode_D = group_filter_ui("D", name_D, show_D)

# Apply filters per group (only groups toggled on)
A = apply_group_filters(df, flt_A, tmode_A)
groups_all = [("A", name_A, True,  A)]
if show_B: groups_all.append(("B", name_B, True,  apply_group_filters(df, flt_B, tmode_B)))
if show_C: groups_all.append(("C", name_C, True,  apply_group_filters(df, flt_C, tmode_C)))
if show_D: groups_all.append(("D", name_D, True,  apply_group_filters(df, flt_D, tmode_D)))

# Status bar
st.markdown("#### Status")
cols = st.columns(1 + len(groups_all))
cols[0].metric("Rows after cleaning", len(df))
for i, (_, label, _, gdf) in enumerate(groups_all, start=1):
    cols[i].metric(f"{label} rows", len(gdf))

if all(gdf.empty for _,_,_,gdf in groups_all):
    st.warning("All groups are empty with current filters.")
    st.stop()

# -----------------------------------------
# Aggregation helpers & metric preparation
# -----------------------------------------
def agg_by_creator(dfin: pd.DataFrame) -> pd.DataFrame:
    if dfin.empty:
        return pd.DataFrame(columns=[COLS["trainer"], "subs", "ltv_mean", "ltv_median", "ltv_sum", "months_avg"])
    months_col = {6: COLS["months6"], 12: COLS["months12"], 24: COLS["months24"]}[horizon]
    have_months = months_col in dfin.columns
    grp = (dfin.groupby(COLS["trainer"], as_index=False)
           .agg(subs=(rev_col, "size"),
                ltv_mean=(rev_col, "mean"),
                ltv_median=(rev_col, "median"),
                ltv_sum=(rev_col, "sum"),
                months_avg=(months_col, "mean") if have_months else (rev_col, "size")))
    if not have_months:
        grp["months_avg"] = np.nan
    return grp

per_group = {}
for code, label, _, gdf in groups_all:
    per_group[code] = agg_by_creator(gdf)

# Creators universe & total LTV across visible groups (for default sort/top list)
all_creators = pd.Index([])
for code, _, _, _ in groups_all:
    if not per_group[code].empty:
        all_creators = all_creators.union(per_group[code][COLS["trainer"]])

rank_base = pd.DataFrame({COLS["trainer"]: all_creators})
rev_cols_for_total = []
for code, label, _, _ in groups_all:
    t = per_group[code][[COLS["trainer"], "ltv_sum"]].copy()
    scol = f"sum_{horizon}m ({label})"
    t = t.rename(columns={"ltv_sum": scol})
    rev_cols_for_total.append(scol)
    rank_base = rank_base.merge(t, on=COLS["trainer"], how="left")

rank_base["_total_revenue_all_groups"] = rank_base[rev_cols_for_total].sum(axis=1, skipna=True)

if restrict_all and len(groups_all) >= 1:
    eligible_sets = []
    for code, _, _, _ in groups_all:
        pg = per_group[code]
        keep = set(pg.loc[pg["subs"].fillna(0) >= min_subs, COLS["trainer"]])
        eligible_sets.append(keep)
    if eligible_sets:
        must_have = set.intersection(*eligible_sets) if eligible_sets else set()
        rank_base = rank_base[rank_base[COLS["trainer"]].isin(must_have)]
        for code in list(per_group.keys()):
            per_group[code] = per_group[code][per_group[code][COLS["trainer"]].isin(must_have)]

sort_choice = st.selectbox(
    "Sort rows by",
    options=["Total LTV (across visible groups)", "Creator (A→Z)"],
    index=0
)

if rank_base.empty:
    st.warning(
        "No creators meet the current filters. This often happens when "
        f"**Restrict to all groups ≥ {min_subs} subs** is enabled and at least one group "
        "has too few subs per creator. Try lowering the subs threshold or disabling the restriction."
    )
    st.stop()

if sort_choice.startswith("Total LTV"):
    order_series = rank_base.set_index(COLS["trainer"])["_total_revenue_all_groups"]
    creators_sorted = order_series.sort_values(ascending=False).index.tolist()
else:
    creators_sorted = sorted(rank_base[COLS["trainer"]].tolist())

top_creators = creators_sorted[:top_n]
if len(top_creators) == 0:
    st.warning(
        "No creators to display after sorting/top-N selection. "
        "Try increasing **Top creators** or relaxing filters."
    )
    st.stop()

# ---------------------------------------
# Build the wide matrix for the top table
# ---------------------------------------
metric_map = {
    "Mean LTV": "ltv_mean",
    "Median LTV": "ltv_median",
    "Sum LTV": "ltv_sum",
    "Avg Months Active": "months_avg",
    "Subs": "subs",
}
metric_col = metric_map[metric_choice]

top_creators_series = pd.Series(top_creators, dtype="string")
wide_num = pd.DataFrame({"Creator": top_creators_series})
wide_subs = pd.DataFrame({"Creator": top_creators_series})
group_cols_rendered = []

for code, label, _, _ in groups_all:
    t = per_group[code][per_group[code][COLS["trainer"]].isin(top_creators)].copy()
    t = t.rename(columns={COLS["trainer"]: "Creator"})
    mcol = f"{label}"
    scol = f"{label}__subs"
    wide_num = wide_num.merge(t[["Creator", metric_col]], how="left", on="Creator").rename(columns={metric_col: mcol})
    wide_subs = wide_subs.merge(t[["Creator", "subs"]],     how="left", on="Creator").rename(columns={"subs": scol})
    group_cols_rendered.append(mcol)

for label in group_cols_rendered:
    subs_col = f"{label}__subs"
    mask_low = (wide_subs[subs_col].fillna(0) < min_subs)
    wide_num.loc[mask_low, label] = np.nan

numeric_for_style = wide_num.set_index("Creator")[group_cols_rendered].copy()
subs_df = wide_subs.set_index("Creator")[[f"{g}__subs" for g in group_cols_rendered]].copy()
subs_df.columns = group_cols_rendered

delta_label = f"Median % uplift vs. base ({name_A})"
if name_A in group_cols_rendered:
    base_vals = numeric_for_style[name_A]
    deltas = {}
    for label in group_cols_rendered:
        if label == name_A:
            deltas[label] = 0.0
        else:
            comp_vals = numeric_for_style[label]
            pct = (comp_vals - base_vals) / base_vals.replace(0, np.nan)
            deltas[label] = pct.replace([np.inf, -np.inf], np.nan).median(skipna=True)
    delta_numeric_row = pd.Series(deltas, index=group_cols_rendered, name=delta_label)
else:
    delta_numeric_row = pd.Series([np.nan] * len(group_cols_rendered), index=group_cols_rendered, name=delta_label)

def fmt_cell(val, n):
    if pd.isna(val):
        v = ""
    elif metric_choice == "Avg Months Active":
        v = f"{float(val):.1f}"
    else:
        v = f"{int(np.round(val, 0))}"
    n_str = "" if pd.isna(n) else f"<i>{int(n)}</i>"
    return f"<div class='cell'><span class='subs'>{n_str}</span><span class='val'>{v}</span></div>"

def fmt_pct_cell(p):
    if pd.isna(p):
        return "<div class='cell'><span class='subs'></span><span class='val'></span></div>"
    return f"<div class='cell'><span class='subs'></span><span class='val'>{int(np.round(p*100,0))}%</span></div>"

render_html = pd.DataFrame(index=numeric_for_style.index, columns=group_cols_rendered, data="")
for col in group_cols_rendered:
    render_html[col] = [fmt_cell(numeric_for_style.loc[idx, col], subs_df.loc[idx, col]) for idx in numeric_for_style.index]

delta_html_row = pd.Series({col: fmt_pct_cell(delta_numeric_row[col]) for col in group_cols_rendered}, name=delta_label)

combined_html = pd.concat([pd.DataFrame([delta_html_row]), render_html])
combined_numeric = pd.concat([pd.DataFrame([delta_numeric_row]), numeric_for_style])

display_table = combined_html.copy()
display_table.insert(0, "Creator", [delta_label] + numeric_for_style.index.tolist())

def row_extrema_styles(values_row):
    vals = values_row.values.astype(float)
    finite_mask = np.isfinite(vals)
    styles = []
    if finite_mask.any():
        maxv = np.nanmax(vals); minv = np.nanmin(vals)
        for v in vals:
            if np.isfinite(v) and v == maxv and v == minv:
                styles.append("")
            elif np.isfinite(v) and v == maxv:
                styles.append("background-color: #d6f5d6;")
            elif np.isfinite(v) and v == minv:
                styles.append("background-color: #f8d7da;")
            else:
                styles.append("")
    else:
        styles = ["" for _ in vals]
    return styles

styler = display_table.style.apply(
    lambda r: row_extrema_styles(combined_numeric.loc[r.name]),
    axis=1,
    subset=group_cols_rendered
).hide(axis="index")

def emphasize_delta(df: pd.DataFrame):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if delta_label in styles.index:
        styles.loc[delta_label, :] = "border-top: 2px solid #111; border-bottom: 2px solid #111; font-weight: 600;"
    return styles

styler = styler.apply(emphasize_delta, axis=None)

title_metric = {"Mean LTV":"Mean LTV", "Median LTV":"Median LTV", "Sum LTV":"Sum LTV",
                "Avg Months Active":"Avg Months Active", "Subs":"Subs"}[metric_choice]
st.subheader(f"Top {top_n} creators — {title_metric} by group")
st.caption("Row-wise max = green, min = red. Values hidden when subs < threshold; subs count is shown at the left of each cell.")

colors = ["#eaf2ff", "#e3edff", "#dce8ff", "#d5e3ff"]
borders = ["#6ea8fe", "#639bff", "#5991ff", "#4f86ff"]

header_css = ""
for i, _ in enumerate(group_cols_rendered, start=2):  # col 1 = Creator
    j = i - 2
    header_css += f".dataframe thead th:nth-child({i}) {{ background: {colors[j]}; border-bottom: 2px solid {borders[j]}; }}\n"

css = f"""
<style>
.dataframe td {{ vertical-align: middle; padding: 6px 10px; }}
.cell {{ display: flex; justify-content: space-between; align-items: baseline; }}
.cell .subs {{ color: #9aa0a6; font-size: 0.9em; font-weight: 400; margin-right: 8px; }}
.cell .val  {{ color: #111;  font-size: 1.05em; font-weight: 600; }}
{header_css}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown(styler.to_html(), unsafe_allow_html=True)

out_numeric = combined_numeric.copy()
out_numeric.insert(0, "Creator", [delta_label] + numeric_for_style.index.tolist())
st.download_button(
    "Download numeric table (CSV)",
    data=out_numeric.to_csv(index=False),
    file_name=f"creators_top{top_n}_{metric_map[metric_choice]}_{horizon}m_numeric.csv",
    mime="text/csv"
)

def strip_html_cell(html_cell: str) -> str:
    if not isinstance(html_cell, str): return ""
    txt = (html_cell
           .replace("<div class='cell'>", "")
           .replace("</div>", "")
           .replace("<span class='subs'>", "")
           .replace("</span>", " ")
           .replace("<span class='val'>", ""))
    txt = txt.replace("<i>", "").replace("</i>", "")
    return " ".join(txt.split())

out_display = display_table.copy()
out_display[group_cols_rendered] = out_display[group_cols_rendered].apply(lambda col: col.map(strip_html_cell))
st.download_button(
    "Download display table (CSV)",
    data=out_display.to_csv(index=False),
    file_name=f"creators_top{top_n}_{metric_map[metric_choice]}_{horizon}m_display.csv",
    mime="text/csv"
)




