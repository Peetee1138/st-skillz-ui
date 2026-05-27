# app15.py - v0.15
#   0.2   - Implement 2-Skill Explorer and Skill Combo Detail locally for G2-G4 (20251207)
#         - initial deployment to GitHub / OnRender
#         - transition to .npz for 
#   0.3   - Implement Tab 3 and enhancements
#   0.4   - Back out efforts to highlight the heatmap on Tab 1 (failed), add flex sizing to 2-skill tab; update to use _unique data (deploy / deploy2)
#   0.5   - TBD
#   0.6   - Updating Tab2, Frame3 to show alternative histogram of rating percentiles
#   0.7   - Uploading all classes, adjusting base text, fix behaviors
#   0.8   - Rely only on xx_all_data_ui.npz (remove use of xx_final_data.npz
#   0.9   - Clean up and move to .npy for data efficency, add Using this Tool
#   0.10  - Add "tab 4" for Class Info
#   0.11  - Add Discord login
#   0.12  - (from 11a_deploy) tweaks based on Alpha Feeedback
#   0.13  - Add enhancements to Tab 1 (selection of skills), add whatif (Frame 4) to tab2 (Skill Combo Detail)
#   0.14  - [abandoned: Reddit login] Add 3-free logins to give time to add users to white list
#   0.15  - Updated code to use consistent terminology, modified to be mobile friendly
#   0.15a - Tweaks to make charts more mobile friendly - sub-version to avoid damaging v0.15


import os
from pathlib import Path

import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import json

from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update, callback_context
from dash.dependencies import ALL

# Related to User Authorization

from flask import Flask, session, redirect, url_for, request, make_response
from authlib.integrations.flask_client import OAuth


import csv
import requests
from datetime import datetime, timezone
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
# DATA_DIR  = Path(".")  # adjust if you keep CSVs elsewhere

CLASS_DIR = DATA_DIR / Path("classes")
HERO_CODES_CSV = DATA_DIR / "db_hero_codes.csv"
SKILL_CODES_CSV = DATA_DIR / "db_skill_codes.csv"
SKILL_SORT_ORDER_CSV = DATA_DIR / "db_skill_sort_order.csv"
SKILL_INCOMPAT_CSV = DATA_DIR / "db_skill_codes_incompat.csv"
SLOT_ODDS_CSV = DATA_DIR / "db_slot_odds.csv"
DETAIL_ICON_PATH = DATA_DIR / "assets" / "detail_icons"
TREND_ICON_PATH = DATA_DIR / "assets" / "trend_icons"
TAB5_MD_PATH = DATA_DIR / "assets" / "using_this_tool.md"

DETAIL_ICON_URL = "/assets/detail_icons"
TREND_ICON_URL = "/assets/trend_icons"

AUTHORIZED_USERS_FILE = BASE_DIR / "authorized_users.txt"
LOGIN_LOG_FILE = BASE_DIR / "login_attempts.csv"

FREE_USES_ALLOWED = 3

def get_class_bundle_dir(class_code: str) -> Path:
    return CLASS_DIR / str(class_code).strip()
    
CLASS_SKILL_ASSESS_FILES = {
    "G2": CLASS_DIR / "G2_data_assess.csv",
    "G3": CLASS_DIR / "G3_data_assess.csv",
    "G4": CLASS_DIR / "G4_data_assess.csv",
    "G5": CLASS_DIR / "G5_data_assess.csv",
    "G6": CLASS_DIR / "G6_data_assess.csv",
    "G7": CLASS_DIR / "G7_data_assess.csv",
    "B1": CLASS_DIR / "B1_data_assess.csv",
    "B2": CLASS_DIR / "B2_data_assess.csv",
    "B3": CLASS_DIR / "B3_data_assess.csv",
    "B4": CLASS_DIR / "B4_data_assess.csv",
    "B5": CLASS_DIR / "B5_data_assess.csv",
    "B6": CLASS_DIR / "B6_data_assess.csv",
    "B7": CLASS_DIR / "B7_data_assess.csv",
    "R1": CLASS_DIR / "R1_data_assess.csv",
    "R2": CLASS_DIR / "R2_data_assess.csv",
    "R3": CLASS_DIR / "R3_data_assess.csv",
    "R4": CLASS_DIR / "R4_data_assess.csv",
    "R5": CLASS_DIR / "R5_data_assess.csv",
    "R6": CLASS_DIR / "R6_data_assess.csv",
    "R7": CLASS_DIR / "R7_data_assess.csv",}

_single_skill_assess_cache = {}

# ---------- SHARED UI STYLES ----------

# -- Background Colors --

APP_BG   = "#f5f3ef"   # warm soft off-white
PANEL_BG = "#fbfaf7"   # slightly lighter for boxes/cards
CHART_BG = "#fcfbf9"   # for plotly paper/plot backgrounds

# -- other styles --

LEFT_FRAME_STYLE = {
    "width": "100%",
    "maxWidth": "350px",
    "flex": "0 1 350px",
}

LEFT_FRAME_DROPDOWN_STYLE = {
    "width": "100%",
    "minWidth": "0",
}

TITLE_BANNER_STYLE = {
    "textAlign": "center",
    "fontWeight": "bold",
    "backgroundColor": "#444444",
    "color": "white",
    "padding": "6px",
    "borderRadius": "4px",
    "marginBottom": "4px",
}

ROW_CLASS_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "marginTop": "2px",
    "width": "100%",
}

ROW_SKILL_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "marginTop": "2px",
    "width": "100%",
}

CLAMP_2_LINES = {
    "display": "-webkit-box",
    "WebkitBoxOrient": "vertical",
    "WebkitLineClamp": "2",
    "overflow": "hidden",
}

CLICKABLE_ICON_BUTTON_STYLE = {
    "border": "2px solid #0645AD",  # hyperlink blue
    "borderRadius": "2px",
    "padding": "0px",
    "background": "transparent",
    "cursor": "pointer",
    "display": "flex",                 # allow flex centering
    "alignItems": "center",            # vertical centering
    "justifyContent": "center",        # horizontal centering
}

TAB4_LINK_BUTTON_STYLE = {
    "border": "2px solid #0645AD",
    "borderRadius": "4px",
    "padding": "2px 6px",
    # "background": "white",
    "background": PANEL_BG,
    "cursor": "pointer",
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "gap": "6px",
    "minHeight": "38px",
    "whiteSpace": "nowrap",
}

TAB4_ICON_ROW_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "gap": "4px",
    "flexWrap": "nowrap",
}

TAB4_CODE_INLINE_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "gap": "4px",
    "fontSize": "12px",
    "lineHeight": "1.0",
    "whiteSpace": "nowrap",
}

TAB4_TABLE_STYLE = {
    "borderCollapse": "collapse",
    "width": "100%",
    "backgroundColor": "white",
}

TAB4_TH_STYLE_CLASS = {
    "border": "1px solid #444",
    "backgroundColor": "#444444",
    "color": "white",
    "padding": "6px 8px",
    "fontSize": "13px",
    "textAlign": "center",
    "whiteSpace": "nowrap",
}

TAB4_TH_STYLE_KEY = {
    "border": "1px solid #8a6d00",
    "backgroundColor": "#b8860b",
    "color": "white",
    "padding": "6px 8px",
    "fontSize": "13px",
    "textAlign": "center",
    "whiteSpace": "nowrap",
}

TAB4_TH_STYLE_CORE = {
    "border": "1px solid #1f3a5f",
    "backgroundColor": "#274c77",
    "color": "white",
    "padding": "6px 8px",
    "fontSize": "13px",
    "textAlign": "center",
    "whiteSpace": "nowrap",
}

TAB4_TH_STYLE_BUILD = {
    "border": "1px solid #5a1f2b",
    "backgroundColor": "#7a2838",
    "color": "white",
    "padding": "6px 8px",
    "fontSize": "13px",
    "textAlign": "center",
    "whiteSpace": "nowrap",
}

TAB4_TD_STYLE = {
    "border": "1px solid #888",
    "padding": "4px 6px",
    "fontSize": "13px",
    "textAlign": "center",
    "verticalAlign": "middle",
    "whiteSpace": "nowrap",
}

TAB4_CLASS_CELL_STYLE = {
    **TAB4_TD_STYLE,
    "textAlign": "left",
    "fontWeight": "700",
    "fontSize": "26px",
}

TAB4_MAX_CELL_STYLE = {
    **TAB4_TD_STYLE,
    "fontWeight": "700",
    "fontSize": "26px",
}

TAB4_CLASS_LINE_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "flex-start",
    "gap": "8px",
    "whiteSpace": "nowrap",
}

TAB4_ROW_BG_GREEN = "#d9ead3"
TAB4_ROW_BG_GREEN_CLASS = "#b6d7a8"

TAB4_ROW_BG_BLUE = "#d9eaf7"
TAB4_ROW_BG_BLUE_CLASS = "#9fc5e8"

TAB4_ROW_BG_RED = "#f4d6d6"
TAB4_ROW_BG_RED_CLASS = "#ea9999"
RATING_HIST_BINS = list(range(0, 101, 5))  # 0,5,10,...,100
RATING_HIST_COLS_ORDERED = [f"rating_hist_{b}" for b in RATING_HIST_BINS]
RATING_BIN_CENTERS = {f"rating_hist_{b}": float(b) for b in RATING_HIST_BINS}

T2_F3_MIN_TO_SHOW = 30

T2_F2_SCORE_COL_W = "120px"
T2_F2_LABEL_COL_W = "calc(100% - 120px)"

# ---------- DATA LOAD HELPERS ----------

def load_hero_codes():
    df = pd.read_csv(HERO_CODES_CSV)
    return df

def load_skill_metadata():
    df = pd.read_csv(SKILL_CODES_CSV)
    print("Loaded skill codes with columns:", list(df.columns))

    # expected columns
    # short_name, full_name, rarity, skill_id
    if "short_name" not in df.columns:
        raise ValueError("[load_skill_metadata] missing column: short_name")
    if "full_name" not in df.columns:
        raise ValueError("[load_skill_metadata] missing column: full_name")
    if "skill_id" not in df.columns:
        raise ValueError("[load_skill_metadata] missing column: skill_id")

    name_map = df.set_index("short_name")["full_name"].to_dict()
    rarity_map = df.set_index("short_name")["rarity"].to_dict() if "rarity" in df.columns else {}
    code_to_id = df.set_index("short_name")["skill_id"].astype(int).to_dict()
    id_to_code = df.set_index("skill_id")["short_name"].to_dict()

    return name_map, rarity_map, code_to_id, id_to_code
    
def skill_label(code: str) -> str:
    """
    Canonical label: 'ABC - Full Skill Name'
    Used everywhere we build dropdown options.
    """
    if not code:
        return ""
    full = strip_parens(get_full_skill_name(code))
    return f"{code} - {full}"

def load_skill_sort_order():
    """
    Load db_skill_sort_order.csv as a mapping:
      class_code -> [skill_code1, skill_code2, ...] in desired order.

    Assumes:
      - Column 0: class code (e.g. 'G2')
      - Columns 1+ : 3-letter skill codes for that class, in order
      - There is a header row.
    """
    df = pd.read_csv(SKILL_SORT_ORDER_CSV)
    # First column is class code, rest are skills
    class_col = df.columns[0]
    skill_cols = df.columns[1:]

    mapping = {}
    for _, row in df.iterrows():
        code = str(row[class_col])
        skills = [str(row[c]) for c in skill_cols if pd.notna(row[c])]
        mapping[code] = skills
    return mapping

def load_slot_odds():
    """
    Load db_slot_odds.csv into:
      slot_odds_map = {
          1: {"Common": 0.833, "Rare": 0.125, "Epic": 0.042},
          ...
      }

    CSV expected columns:
      slot, pct_com, pct_rare, pct_epic
    Percentages in file are 0-100 scale.
    """
    df = pd.read_csv(SLOT_ODDS_CSV)

    req = {"slot", "pct_com", "pct_rare", "pct_epic"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"[load_slot_odds] missing columns: {sorted(missing)}")

    out = {}
    for _, row in df.iterrows():
        try:
            slot = int(row["slot"])
        except Exception:
            continue

        out[slot] = {
            "Common": float(row["pct_com"]) / 100.0,
            "Rare":   float(row["pct_rare"]) / 100.0,
            "Epic":   float(row["pct_epic"]) / 100.0,
        }

    return out

def parse_skill_codes(skill_code: str):
    """
    Example: 'G2AcrAllAntAss' -> ('G2', ['Acr','All','Ant','Ass'])
    """
    cls = skill_code[:2]
    codes = [skill_code[2 + i*3 : 2 + (i+1)*3] for i in range(4)]
    return cls, codes

def canonical_skill_string(s1, s2, s3, s4) -> str:
    """
    Return a canonical, alphabetized 4-skill string, e.g.
      ('Ass','All','Dan','Des') -> 'AllAssDanDes'
    """
    skills = [s1, s2, s3, s4]
    skills = [s for s in skills if s]   # defensive
    return "".join(sorted(skills))

hero_codes_df = load_hero_codes()
skill_name_map, skill_rarity_map, skill_code_to_id, skill_id_to_code = load_skill_metadata()
skill_sort_order = load_skill_sort_order()
slot_odds_map = load_slot_odds()

_class_bundle_cache = {}
_single_skill_assess_cache = {}

def canonical_full_skill_code(class_code: str, skills: list[str]) -> str | None:
    """
    Build canonical full skill_code for exact bundle lookup.
    Example:
      class_code='R3', skills=['EWa','EPl','Swo','Whi']
      -> 'R3EPlEWaSwoWhi'
    """
    if not class_code or not skills or len(skills) != 4:
        return None

    vals = [str(s).strip() for s in skills if s]
    if len(vals) != 4:
        return None

    return f"{str(class_code).strip()}{''.join(sorted(vals))}"

def get_class_bundle(class_code: str):
    """
    Lazy-load per-class numeric bundle from:
      classes/<class_code>/*.npy
      classes/<class_code>/skill_index.npz
    """
    if not class_code:
        return None

    class_code = str(class_code).strip()

    if class_code in _class_bundle_cache:
        return _class_bundle_cache[class_code]

    class_dir = get_class_bundle_dir(class_code)
    if not class_dir.exists():
        print(f"[get_class_bundle] Missing class dir for {class_code}: {class_dir}")
        _class_bundle_cache[class_code] = None
        return None

    req = [
        "s1.npy", "s2.npy", "s3.npy", "s4.npy",
        "raw_rating.npy", "net_rating.npy", "rating_pctile.npy",
        "c_r1.npy", "c_r2.npy", "c_r3.npy", "rating_cat.npy",
    ]
    missing = [fn for fn in req if not (class_dir / fn).exists()]
    if missing:
        print(f"[get_class_bundle] Missing files for {class_code}: {missing}")
        _class_bundle_cache[class_code] = None
        return None

    bundle = {
        "class_dir": class_dir,
        "s1": np.load(class_dir / "s1.npy", mmap_mode="r"),
        "s2": np.load(class_dir / "s2.npy", mmap_mode="r"),
        "s3": np.load(class_dir / "s3.npy", mmap_mode="r"),
        "s4": np.load(class_dir / "s4.npy", mmap_mode="r"),
        "raw_rating": np.load(class_dir / "raw_rating.npy", mmap_mode="r"),
        "net_rating": np.load(class_dir / "net_rating.npy", mmap_mode="r"),
        "rating_pctile": np.load(class_dir / "rating_pctile.npy", mmap_mode="r"),
        "c_r1": np.load(class_dir / "c_r1.npy", mmap_mode="r"),
        "c_r2": np.load(class_dir / "c_r2.npy", mmap_mode="r"),
        "c_r3": np.load(class_dir / "c_r3.npy", mmap_mode="r"),
        "rating_cat": np.load(class_dir / "rating_cat.npy", mmap_mode="r"),
        "skill_index": None,
        "skill_code_index": None,   # NEW
    }
    
    idx_path = class_dir / "skill_index.npz"
    if idx_path.exists():
        bundle["skill_index"] = np.load(idx_path, allow_pickle=False)

    sc_idx_path = class_dir / "skill_code_index.npz"
    if sc_idx_path.exists():
        bundle["skill_code_index"] = np.load(sc_idx_path, allow_pickle=True)

    _class_bundle_cache[class_code] = bundle
    return bundle


def get_skill_rows_for_class(bundle: dict, skill_code: str) -> np.ndarray:
    """
    Fast row lookup for 'all rows containing this skill'.
    Uses skill_index.npz if present, else falls back to scanning s1..s4.
    """
    if bundle is None or not skill_code:
        return np.array([], dtype=np.int32)

    skill_id = skill_code_to_id.get(skill_code)
    if skill_id is None:
        return np.array([], dtype=np.int32)

    key = f"s{int(skill_id):02d}"

    idx_file = bundle.get("skill_index")
    if idx_file is not None and key in idx_file.files:
        return idx_file[key].astype(np.int32, copy=False)

    s1 = bundle["s1"]
    s2 = bundle["s2"]
    s3 = bundle["s3"]
    s4 = bundle["s4"]

    mask = (s1 == skill_id) | (s2 == skill_id) | (s3 == skill_id) | (s4 == skill_id)
    return np.where(mask)[0].astype(np.int32)


def net_rating_display(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "n/q"
    return "n/q" if x < 0 else f"{x:.2f}"


def get_skill_codes_for_row(bundle: dict, row_idx: int) -> list[str]:
    skill_ids = [
        int(bundle["s1"][row_idx]),
        int(bundle["s2"][row_idx]),
        int(bundle["s3"][row_idx]),
        int(bundle["s4"][row_idx]),
    ]
    return [skill_id_to_code[i] for i in skill_ids]


def find_combo_index(bundle: dict, class_code: str, skills: list[str]):
    """
    Exact combo match using the exported skill_code_index.npz.
    Returns row index or None.
    """
    if bundle is None or not class_code or not skills or len(skills) != 4:
        return None

    full_code = canonical_full_skill_code(class_code, skills)
    if not full_code:
        return None

    idx_file = bundle.get("skill_code_index")
    if idx_file is None:
        return None

    try:
        codes = idx_file["skill_code"]
        rows = idx_file["row_idx"]
    except Exception:
        return None

    matches = np.where(codes == full_code)[0]
    if matches.size == 0:
        return None

    return int(rows[matches[0]])

def build_combo_row(bundle: dict, class_code: str, row_idx: int) -> dict:
    skill_codes = get_skill_codes_for_row(bundle, row_idx)

    return {
        "class_code": class_code,
        "skill_code": f"{class_code}{''.join(skill_codes)}",
        "skill_list": skill_codes,
        "raw_rating": float(bundle["raw_rating"][row_idx]),
        "net_rating": net_rating_display(bundle["net_rating"][row_idx]),
        "rating_pctile": float(bundle["rating_pctile"][row_idx]),
        "c_r1": float(bundle["c_r1"][row_idx]),
        "c_r2": float(bundle["c_r2"][row_idx]),
        "c_r3": float(bundle["c_r3"][row_idx]),
        "rating_cat": int(bundle["rating_cat"][row_idx]),
    }

# Map hero class codes -> name + icon
# Assumes hero class icons are named like G2.png, G3.png etc in assets/detail_icons
class_meta = {
    row["Code"]: {
        "name": row["Hero_Class"],
        # full src path for the class icon
        "icon_src": f"/assets/hero_classes/{row['Code']}.png",
    }
    for _, row in hero_codes_df.iterrows()
}

skill_lookup = {
    code: {
        "full_name": skill_name_map.get(code, code),
        # full src path for the skill icon
        "icon_src": f"/assets/skill_icons/{code}.png",
    }
    for code in skill_name_map.keys()
}


RARITY_COLORS = {
    "Epic":   "rgb(255,242,204)",  # soft yellow
    "Rare":   "rgb(207,226,243)",  # soft blue
    "Common": "rgb(239,239,239)",  # light copper/gray
}

rarity_styles = []
for code, rarity in skill_rarity_map.items():
    color = RARITY_COLORS.get(rarity)
    if not color:
        continue

    # Style for Skill 3 cell when this code is used there
    rarity_styles.append(
        {
            "if": {
                "filter_query": f'{{_s3_code}} = "{code}"',
                "column_id": "s3_full",
            },
            "backgroundColor": color,
        }
    )

    # Style for Skill 4 cell when this code is used there
    rarity_styles.append(
        {
            "if": {
                "filter_query": f'{{_s4_code}} = "{code}"',
                "column_id": "s4_full",
            },
            "backgroundColor": color,
        }
    )

def get_base_skills_for_class(class_code: str) -> list[str]:
    """
    Robust skill pool for a class.

    Priority:
      1) assess csv for this class
      2) skill_sort_order.csv
      3) all known skills
    """
    if not class_code:
        return []

    df_assess = get_single_skill_assess_df(class_code)
    if df_assess is not None and not df_assess.empty:
        if "sk_name" in df_assess.columns:
            vals = sorted(set(df_assess["sk_name"].dropna().astype(str).str.strip()))
            if vals:
                return vals
        if "skill_code" in df_assess.columns:
            vals = sorted(set(df_assess["skill_code"].dropna().astype(str).str.strip()))
            if vals:
                return vals

    order = skill_sort_order.get(class_code, [])
    if order:
        return sorted(set(order))

    return sorted(skill_name_map.keys())
    
def get_full_skill_name(short_code: str) -> str:
    """Return full skill name from 3-letter code, or the code if missing."""
    return skill_name_map.get(short_code, short_code)


def strip_parens(name: str) -> str:
    """Remove ' ( ... )' parts from a skill name for titles."""
    return re.sub(r"\s*\([^)]*\)", "", name).strip()

# ---------- SKILL INCOMPATIBILITIES ----------

def build_incompat_dict_from_long_form(df: pd.DataFrame) -> dict:
    """
    Turn a 'long form' incompat table into a symmetric dict:
      {'Acr': {'Blu', 'Xxx'}, 'Blu': {'Acr', 'Yyy'}, ...}
    """
    incompat_dict: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        base_skill = str(row["skill"]).strip()
        # all other non-NaN columns in that row are incompatibles
        incompat_skills = (
            row[1:]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        if base_skill not in incompat_dict:
            incompat_dict[base_skill] = set()
        incompat_dict[base_skill].update(incompat_skills)

        # make it symmetric
        for other in incompat_skills:
            if other not in incompat_dict:
                incompat_dict[other] = set()
            incompat_dict[other].add(base_skill)

    return incompat_dict


# Load once at startup
incompat_df = pd.read_csv(SKILL_INCOMPAT_CSV)
incompat_dict = build_incompat_dict_from_long_form(incompat_df)

def filtered_skill_pool(
    base_skills: list[str],
    fixed_skills: list[str],
) -> list[str]:
    """
    Starting from base_skills, remove:
      - any already-picked skills in fixed_skills
      - any skills incompatible with any of fixed_skills
    """
    fixed = [s for s in fixed_skills if s]
    pool = [s for s in base_skills if s not in fixed]

    for fixed_skill in fixed:
        bad = incompat_dict.get(fixed_skill)
        if bad:
            pool = [s for s in pool if s not in bad]

    return pool

# Tab 1: Frame 3 - build skill order

ASSESS_RANK_COL = "rank_mx_95ile_r90_avg"
TIEBREAK_COL    = "95ile_pct"

def build_skill_order_from_assess(df_assess: pd.DataFrame, skills_in_scope: list[str]) -> list[str]:
    """
    Returns skills ordered by:
      1) ASSESS_RANK_COL (ascending)
      2) TIEBREAK_COL    (descending)
      3) sk_name         (ascending, deterministic fallback)

    Any skills missing from assess are appended alphabetically at the end.
    """
    if df_assess is None or df_assess.empty:
        return sorted(skills_in_scope)

    tmp = df_assess[df_assess["sk_name"].isin(skills_in_scope)].copy()

    # Required column must exist
    if ASSESS_RANK_COL not in tmp.columns:
        return sorted(skills_in_scope)

    # Coerce numeric columns safely
    tmp[ASSESS_RANK_COL] = pd.to_numeric(tmp[ASSESS_RANK_COL], errors="coerce")

    if TIEBREAK_COL in tmp.columns:
        tmp[TIEBREAK_COL] = pd.to_numeric(tmp[TIEBREAK_COL], errors="coerce")
    else:
        # If missing, create neutral values so sort still works
        tmp[TIEBREAK_COL] = 0

    # Drop rows with no primary rank
    tmp = tmp.dropna(subset=[ASSESS_RANK_COL])

    # ---- SORT ORDER ----
    tmp = tmp.sort_values(
        by=[ASSESS_RANK_COL, TIEBREAK_COL, "sk_name"],
        ascending=[True, False, True],   # ← key line
        kind="mergesort"                 # stable sort (important for Dash UI consistency)
    )

    ordered = tmp["sk_name"].tolist()

    # Append missing skills deterministically
    missing = [s for s in skills_in_scope if s not in set(ordered)]
    return ordered + sorted(missing)

    
def detail_icon(filename, class_name="detail-icon", title=None):
    """
    Small helper to render an <img> from assets/detail_icons.
    Used for quality & special icons in the detail view.
    """
    return html.Img(
        src=f"{DETAIL_ICON_URL}/{filename}",
        className=class_name,
        title=title or filename,
    )
    
def get_quality_icons(rating_pctile_raw, net_rating):
    """
    Returns a list of html.Img components according to the quality rules.

    - rating_pctile_raw is 0–1 (e.g. 0.9967)
    - thresholds are applied on the 0–100 percentile scale
    """
    icons = []
    net_str = str(net_rating).strip().lower()

    # Non-qualifying → always D face
    if net_str == "n/q":
        icons.append(detail_icon("icon_shop_face_d.png", class_name="quality-icon"))
        return icons

    try:
        pct = float(rating_pctile_raw) * 100.0  # 0–100
    except (TypeError, ValueError):
        icons.append(detail_icon("icon_shop_face_d.png", class_name="quality-icon"))
        return icons

    # Thresholds on percentile
    if pct < 90.0:
        icons.append(detail_icon("icon_shop_face_C.png", class_name="quality-icon"))
    elif pct < 95.0:
        icons.append(detail_icon("icon_shop_face_B.png", class_name="quality-icon"))
    elif pct < 99.0:
        icons.append(detail_icon("icon_shop_face_A.png", class_name="quality-icon"))
    elif pct < 99.5:
        icons.append(detail_icon("icon_shop_face_S.png", class_name="quality-icon"))
    elif pct < 99.8:
        icons.append(detail_icon("icon_global_gem.png", class_name="quality-icon"))
    elif pct < 99.9:
        icons.extend(
            [
                detail_icon("icon_global_gem.png", class_name="quality-icon"),
                detail_icon("icon_global_gem.png", class_name="quality-icon"),
            ]
        )
    else:
        icons.extend(
            [
                detail_icon("icon_global_gem.png", class_name="quality-icon"),
                detail_icon("icon_global_gem.png", class_name="quality-icon"),
                detail_icon("icon_global_gem.png", class_name="quality-icon"),
            ]
        )

    return icons



def format_skill_name_with_info(full_name):
    """
    Splits 'SkillName (Some info)' into:
      'SkillName ' + '(' + 'Some info' + ')'
    with the info part in a smaller span.
    If no '(', returns as a single span.
    """
    text = str(full_name)
    if "(" not in text:
        return html.Span(text, className="skill-name-main")

    base, rest = text.split("(", 1)
    rest = rest.rstrip(")")
    return html.Span([
        html.Span(base.strip() + " ", className="skill-name-main"),
        html.Span(f"({rest})", className="skill-name-info"),
    ])

def build_class_and_skills_line(row, class_meta, skill_lookup):
    """
    Render the class + skills line for Tab 2, Frame 2.

    Format:

        [Class Icon] ClassName :
            [SkillIcon] Skill1 | [SkillIcon] Skill2
            [SkillIcon] Skill3 | [SkillIcon] Skill4

    Skills stay in ENTERED order.
    """

    class_code = row["class_code"]
    class_info = class_meta.get(class_code, {})
    class_name = class_info.get("name", class_code)
    class_icon_src = class_info.get("icon_src", f"/assets/hero_classes/{class_code}.png")

    skill_codes = row.get("skill_list", [])
    skill_codes = [c for c in skill_codes if c]
    
    def make_skill_chunk(sc):
        meta = skill_lookup.get(sc, {})
        full_name = meta.get("full_name", sc)
        icon_src = meta.get("icon_src", f"/assets/skill_icons/{sc}.png")
    
        return html.Div(
            [
                html.Button(
                    html.Img(
                        src=icon_src,
                        className="skill-icon",
                        title=full_name,  # <-- fix: was "full_name" string
                    ),
                    id={  # <-- fix: use sc, not undefined code
                        "type": "detail-skill-icon-btn",
                        "skill": sc,
                        "context": "tab2-headline",
                    },
                    n_clicks=0,
                    style=CLICKABLE_ICON_BUTTON_STYLE,
                ),
                html.Span(" ", className="skill-label-icon-space"),
                format_skill_name_with_info(full_name),
            ],
            className="headline-skill-chunk",
        )

    # -------- FORCE ROW SPLIT AFTER SKILL 2 --------

    line1 = []
    line2 = []

    if len(skill_codes) > 0:
        line1.append(make_skill_chunk(skill_codes[0]))
    if len(skill_codes) > 1:
        line1.append(html.Span(" | ", className="skill-separator"))
        line1.append(make_skill_chunk(skill_codes[1]))

    if len(skill_codes) > 2:
        line2.append(make_skill_chunk(skill_codes[2]))
    if len(skill_codes) > 3:
        line2.append(html.Span(" | ", className="skill-separator"))
        line2.append(make_skill_chunk(skill_codes[3]))

    skills_block = html.Div(
        [
            html.Div(line1, className="skills-line"),
            html.Div(line2, className="skills-line"),
        ],
        className="class-skills-block",
    )

    return html.Div(
        [
            html.Span(
                [
                    html.Img(
                        src=class_meta[row["class_code"]]["icon_src"],
                        className="class-icon",
                    ),
                    html.Span(" ", className="class-name-space"),
                    html.Span(
                        class_meta[row["class_code"]]["name"],
                        className="class-name",
                    ),
                ],
                className="class-label-block",
            ),
            skills_block,
        ],
        className="class-skills-line",
    )
    
def build_build_headline(row):
    raw_rating = float(row["raw_rating"])
    rating_pctile_raw = float(row["rating_pctile"])  # 0–1 scale in data
    rating_pctile_pct = rating_pctile_raw * 100.0    # 0–100
    net_rating = row["net_rating"]

    # Rating number with tiny "Rating" tag
    rating_block_children = [
        html.Span(f"{raw_rating:.1f}", className="headline-rating-number"),
        html.Span(" ", className="headline-space"),
        html.Span("Rating", className="headline-sub-label"),
    ]

    # Broken quest icon if N/Q
    if str(net_rating).strip().lower() == "n/q":
        rating_block_children.append(html.Span(" ", className="headline-space"))
        rating_block_children.append(
            detail_icon(
                "icon_quest_broken.png",
                class_name="headline-nq-icon",
                title="Non-Qualifying (Broken Quest)",
            )
        )

    rating_block = html.Span(
        rating_block_children,
        className="headline-rating-block",
    )

    # Percentile block – show as e.g. 99.68 %ile
    pct_block = html.Span(
        [
            html.Span(f"{rating_pctile_pct:.2f}", className="headline-rating-number"),
            html.Span(" ", className="headline-space"),
            html.Span("%ile", className="headline-sub-label"),
        ],
        className="headline-pct-block",
    )

    # Quality + trend icons (trend driven by mr_rating_cat / rating_cat)
    quality_icons = get_quality_icons(rating_pctile_raw, net_rating)
    trend_icons = get_trend_icons(row)

    # Insert a | between quality_icons and trend_icons (only if trend exists)
    icons_children = list(quality_icons)
    if trend_icons:
        icons_children += [
            html.Span(" | ", className="headline-separator"),
            html.Span("Trend: ", className="headline-trend-text"),
        ] + trend_icons

    return html.Div(
        [
            rating_block,
            html.Span(" | ", className="headline-separator"),
            pct_block,
            html.Span(" | ", className="headline-separator"),
            html.Span(icons_children, className="headline-icons-block"),
        ],
        className="build-headline-row",
    )


def get_trend_icons(row):
    raw = row.get("rating_cat", row.get("mr_rating_cat", None))
    try:
        cat = int(raw)
    except (TypeError, ValueError):
        return []

    icon_map = {
        1: ("positive-dynamic.png", "Trend: Positive / Dynamic"),
        2: ("muscle.png", "Trend: Muscle"),
        3: ("fragile.png", "Trend: Fragile"),
    }
    if cat not in icon_map:
        return []

    fname, title = icon_map[cat]

    return [
        html.Img(
            src=f"{TREND_ICON_URL}/{fname}",
            className="trend-icon",   # <-- IMPORTANT: image class
            title=title,
        )
    ]
    
def build_tab2_frame2_section1(row, class_meta, skill_lookup):
    """
    Frame 2 layout for Tab 2:

        (1) Class + Skills line
        (1) Headline row

        (2) Build attributes table   (left)
        (3) Single-skill table       (right)
    """
    return html.Div(
        [
            build_class_and_skills_line(row, class_meta, skill_lookup),
            build_build_headline(row),
            html.Hr(style={"margin": "8px 0"}),

            # two tables → one flex row, top-aligned
            html.Div(
                [
                    # LEFT: Per-skill table / green Skill Summary table
                    html.Div(
                        build_single_skill_table(row),
                        style={
                            "flex": "0 0 50%",
                            "minWidth": "420px",
                            "maxWidth": "50%",
                        },
                    ),
            
                    # RIGHT: Rating Components table — hidden for now, not deleted
                    # html.Div(
                    #     [
                    #         build_rating_components_table(row),
                    #     ],
                    #     style={
                    #         "display": "flex",
                    #         "flexDirection": "column",
                    #         "gap": "8px",
                    #         "flex": "0 0 50%",
                    #         "minWidth": "420px",
                    #         "maxWidth": "50%",
                    #     },
                    # ),
                ],
                className="build-subtables-row",
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "16px",
                    "flexWrap": "wrap",
                },
            ),
        ],
        className="tab2-frame2-build-section",
    )

def format_pctile_01_to_str(p):
    """
    Convert a 0–1 percentile to a '99.68 %ile'-style string.
    Returns '—' if p is None/NaN.
    """
    if p is None or pd.isna(p):
        return "—"
    try:
        return f"{float(p) * 100.0:.2f} %ile"
    except (TypeError, ValueError):
        return "—"

def build_build_substats_table(row):
    """
    Build the Attribute | Result | Result %ile table
    for Frame 2 (Tab 2).
    """
    qsr = row.get("qsr", np.nan)
    avg_rds = row.get("avg_rds", np.nan)
    min_h = row.get("min_h1h2h3", np.nan)
    min_sur_margin = row.get("min_sur_margin", np.nan)
    r95 = row.get("r95", np.nan)

    qsr_pct = row.get("qsr_pctile", np.nan)
    avg_rds_pct = row.get("avg_rds_pctile", np.nan)
    min_h_pct = row.get("min_h1h2h3_pctile", np.nan)
    min_sur_margin_pct = row.get("min_sur_margin_pctile", np.nan)

    rows = [
        {
            "attr": "Quest Survival Rate",
            "result": f"{float(qsr):.3f}%",
            "pct": format_pctile_01_to_str(qsr_pct),
        },
        {
            "attr": "Average Rounds to Win",
            "result": f"{float(avg_rds):.3f} rounds",
            "pct": format_pctile_01_to_str(avg_rds_pct),
        },
        {
            "attr": "Lowest Hero Survival Rate",
            "result": f"{float(min_h):.3f}%",
            "pct": format_pctile_01_to_str(min_h_pct),
        },
        {
            "attr": "Lowest Hero Survival Margin",
            "result": f"{float(min_sur_margin):.2f} rounds",
            "pct": format_pctile_01_to_str(min_sur_margin_pct),
        },
        {
            "attr": "Rounds to Win 95% of the Time",
            "result": f"{float(r95):.3f} rounds",
            "pct": "",
        },
    ]

    # --- styles (light navy header, dark navy borders) ---
    header_bg = "#34495e"   # light-ish navy
    border_col = "#001f3f"  # dark navy

    header_cell_style = {
        "backgroundColor": header_bg,
        "color": "white",
        "border": f"1px solid {border_col}",
        "textAlign": "center",
        "padding": "4px 6px",
        "fontWeight": "bold",
    }

    body_cell_style = {
        "border": f"1px solid {border_col}",
        "padding": "3px 6px",
        "fontSize": "12px",
        "verticalAlign": "middle",
    }

    header = html.Tr(
        [
            html.Th("Build Attribute", style=header_cell_style),
            html.Th("Result", style=header_cell_style),
            html.Th("Class %ile", style=header_cell_style),
        ]
    )

    body_rows = []
    for r in rows:
        body_rows.append(
            html.Tr(
                [
                    html.Td(r["attr"], style=body_cell_style),
                    html.Td(r["result"], style=body_cell_style),
                    html.Td(r["pct"], style=body_cell_style),
                ]
            )
        )

    return html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        className="build-attr-table",
        style={
            "borderCollapse": "collapse",
            "minWidth": "380px",
        },
    )

def build_rating_components_table(row):
    """
    New table under Build Attribute table (Tab 2, Frame 2).

    Uses rating components:
      - c_r1 → Speed (R1)
      - c_r2 → Survival (R2)
      - c_r3 → Efficiency (R3)
    """

    def fmt_score(v):
        if v is None or pd.isna(v):
            return "—"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    r1 = row.get("c_r1", np.nan)
    r2 = row.get("c_r2", np.nan)
    r3 = row.get("c_r3", np.nan)

    rows = [
        {
            "title": "Reliability (R1) [max 1.0]",
            "line2": "How often does a team with this hero win the quest?",
            "line3a": (
                "R1 scores build's Quest Success Rate (win %), proportionally "
                "against all other builds for this class."
            ),
            "line3b": "100% win rate = max R1.   [Uses Trial Data]",
            "val": fmt_score(r1),
        },
        {
            "title": "Survival (R2) [max 1.0]",
            "line2": "How the build contributes to the heroes all safely surviving?",
            "line3a": (
                "R2 evaluates minimum of survival % across all heroes and compares "
                "survival margins across this class."
            ),
            "line3b": "Builds that stay stable under pressure \u2192 higher R2.   [Uses Trial Data]",
            "val": fmt_score(r2),
        },
        {
            "title": "Skill Efficiency (R3) [max varies by class (1.0-1.15)]",
            "line2": "How balanced and optimal are the skills in this build?",
            "line3a": (
                "R3 measures how efficiently the skills in the build contribute to the projected rounds-to-live as well as the rounds-to-win. "
            ),
            "line3b": (
                "This stat relies on stats, and projects onto Ancient Jungle Extreme Legendary & Huge mini=bosses"
            ),
            "val": fmt_score(r3),
        },
    ]


    # Distinct palette from the existing navy & green
    header_bg = "#8e44ad"   # purple-ish
    border_col = "#4a235a"  # darker purple

    header_cell_style = {
        "backgroundColor": header_bg,
        "color": "white",
        "border": f"1px solid {border_col}",
        "textAlign": "center",
        "padding": "4px 6px",
        "fontWeight": "bold",
        "fontSize": "13px",
    }

    body_cell_style = {
        "border": f"1px solid {border_col}",
        "padding": "4px 6px",
        "fontSize": "12px",
        "verticalAlign": "top",
    }

    label_cell_style = {
        **body_cell_style,
        "textAlign": "left",
    }

    value_cell_style = {
        **body_cell_style,
        "textAlign": "center",
        "verticalAlign": "middle",
        "fontWeight": "600",
        "fontSize": "13px",
        "whiteSpace": "nowrap",
        "fontVariantNumeric": "tabular-nums",
    }
    
    header = html.Tr(
        [
            html.Th("Rating Component", style={**header_cell_style, "width": T2_F2_LABEL_COL_W}),
            html.Th("Score",           style={**header_cell_style, "width": T2_F2_SCORE_COL_W}),
        ]
    )

    body_rows = []
    for r in rows:
        label_children = [
            # Line 1: title, biggest + bold
            html.Div(
                r["title"],
                style={
                    "fontWeight": "700",
                    "fontSize": "13px",
                    "marginBottom": "1px",
                },
            ),
            # Line 2: short description
            html.Div(
                r["line2"],
                style={
                    "fontSize": "11px",
                    "marginBottom": "1px",
                },
            ),
            # Line 3 (part 1)
            html.Div(
                r.get("line3a", ""),
                style={
                    "fontSize": "10px",
                    "color": "#555",
                },
            ),
            # Line 3 (part 2)
            html.Div(
                r.get("line3b", ""),
                style={
                    "fontSize": "10px",
                    "color": "#555",
                },
            ),
        ]


        body_rows.append(
            html.Tr(
                [
                    html.Td(label_children, style={**label_cell_style, "width": T2_F2_LABEL_COL_W}),
                    html.Td(
                        html.Div(
                            r["val"],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "height": "100%",
                                "width": "100%",
                            },
                        ),
                        style={**value_cell_style, "width": T2_F2_SCORE_COL_W},
                    ),
                ]
            )
        )

    return html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        className="build-rating-components-table",
        style={
            "borderCollapse": "collapse",
            "width": "100%",           # NEW: let it fill the frame
            "tableLayout": "fixed",    # NEW: enforce widths above
            "minWidth": "380px",
        },
    )

def build_single_skill_table(row):
    """
    Right-hand table: per-skill stats for the 4 skills in this build.

      Columns: [icon] | Skill Name | Rating & Tier | Sparkline
    """
    class_code = row["class_code"]
    df_assess = get_single_skill_assess_df(class_code)

    # Index assess table by 'sk_name' if present
    assess_index = {}
    if df_assess is not None and "sk_name" in df_assess.columns:
        for _, r in df_assess.iterrows():
            assess_index[str(r["sk_name"])] = r

    skill_codes = row.get("skill_list", [])
    skill_codes = [c for c in skill_codes if c]

    # --- styles (forest green) ---
    header_bg = "#2e8b57"   # light forest green
    border_col = "#2e8b57"

    header_cell_style = {
        "backgroundColor": header_bg,
        "color": "white",
        "border": f"1px solid {border_col}",
        "textAlign": "center",
        "padding": "4px 6px",
        "fontWeight": "bold",
    }

    body_cell_style = {
        "border": f"1px solid {border_col}",
        "padding": "3px 6px",
        "fontSize": "12px",
        "verticalAlign": "middle",
    }

    icon_cell_style = {
        **body_cell_style,
        "padding": "6px 3px",
        "textAlign": "center",
        "verticalAlign": "middle",
    }
    rating_cell_style = {
        "textAlign": "center",
        "verticalAlign": "middle",
        "padding": "2px 3px",
        "whiteSpace": "nowrap",
        "border": f"1px solid {border_col}",
    }
    # Header row (with new sparkline title)
    header = html.Tr(
        [
            html.Th("", style=header_cell_style),
            html.Th("Skill Name", style=header_cell_style),
            html.Th("Tier", style=header_cell_style),
            html.Th("n/q | <80 %ile | 80–95 %ile | 95+ %ile", style=header_cell_style),
        ]
    )

    body_rows = []

    for sc in skill_codes:
        # Icon + label
        meta = skill_lookup.get(sc, {})
        full_name = meta.get("full_name", sc)
        skill_icon_src = meta.get("icon_src", f"/assets/skill_icons/{sc}.png")

        icon_cell = html.Button(
            html.Img(
                src=skill_icon_src,
                className="single-skill-icon",
                title=full_name,
            ),
            id={
                "type": "detail-skill-icon-btn",
                "skill": sc,
                "context": "headline",
            },
            n_clicks=0,
            style=CLICKABLE_ICON_BUTTON_STYLE,   # ✅ add this
        )

        skill_label_cell = html.Span(
            [
                html.Span(skill_label(sc), className="skill-label-full"),
                html.Span(sc, className="skill-label-code"),
            ]
        )
        
        # Default values
        # r_max = None
        tier = None
        nq_pct = sub80_pct = pct80_95 = pct95 = 0.0

        srow = assess_index.get(sc)
        if srow is not None:
            # r_max = srow.get("r_max", None)
            tier = srow.get("skill_tier", None)
            nq_pct = srow.get("nq_pct", 0.0)
            sub80_pct = srow.get("sub80_pct", 0.0)
            pct80_95 = srow.get("80_95_pct", 0.0)
            pct95 = srow.get("95ile_pct", 0.0)

        # Rating + tier icon (rating text 20% larger)
        rating_children = []
        # if r_max is not None and not pd.isna(r_max):
        #     rating_children.append(
        #         html.Span(
        #             f"{float(r_max):.1f} ",
        #             className="single-skill-rating",
        #             style={"fontSize": "1.6em"},  
        #         )
        #     )
        # else:
        #     rating_children.append(
        #         html.Span(
        #             "—",
        #             className="single-skill-rating",
        #             style={"fontSize": "1.2em"},
        #         )
        #     )

        tier_icon = single_skill_tier_icon(tier)
        if tier_icon is not None:
            rating_children.append(tier_icon)

        spark = single_skill_sparkline(nq_pct, sub80_pct, pct80_95, pct95)

        body_rows.append(
            html.Tr(
                [
                    html.Td(icon_cell, style=icon_cell_style),
                    html.Td(
                        skill_label_cell,
                        style={**body_cell_style, "fontSize": "16px"},  # roughly 2× base
                    ),
                    html.Td(
                        html.Div(
                            rating_children,
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "width": "100%",
                            },
                        ),
                        style=rating_cell_style,
                    ),
                    html.Td(spark, style={**body_cell_style, "fontSize": "36px"}),
                ]
            )
        )

    return html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        className="single-skill-table",
        style={
            "borderCollapse": "collapse",
            "minWidth": "420px",
        },
    )

# ---------- Build Detail View: Build Per-Skill Table
def get_single_skill_assess_df(class_code: str):
    if not class_code:
        return None

    if class_code in _single_skill_assess_cache:
        return _single_skill_assess_cache[class_code]

    path = CLASS_SKILL_ASSESS_FILES.get(class_code)
    if not path or not path.exists():
        print(f"[single_skill] No assess file for {class_code}: {path}")
        _single_skill_assess_cache[class_code] = None
        return None

    df = pd.read_csv(path)
    _single_skill_assess_cache[class_code] = df
    return df

def single_skill_tier_icon(tier):
    """
    Map skill_tier -> face icon.
      1 -> icon_shop_face_SSS
      2 -> icon_shop_face_S
      3 -> icon_shop_face_A
      4 -> icon_shop_face_B
      6 -> icon_shop_face_D
    """
    mapping = {
        1: "icon_shop_face_SSS.png",
        2: "icon_shop_face_S.png",
        3: "icon_shop_face_A.png",
        4: "icon_shop_face_B.png",
        6: "icon_shop_face_D.png",
    }
    try:
        t = int(tier)
    except (TypeError, ValueError):
        return None

    fn = mapping.get(t)
    if not fn:
        return None

    return detail_icon(fn, class_name="single-skill-tier-icon")

def single_skill_sparkline(nq, sub80, pct80_95, pct95):
    """
    Build an inline 'sparkline' bar made of 4 colored segments:
      purple (nq), red (sub80), yellow (80-95), green (95+)
    Each argument is expected to be 0–1 (fraction of builds).
    """
    vals = []
    for v in [nq, sub80, pct80_95, pct95]:
        try:
            vals.append(max(float(v), 0.0))
        except (TypeError, ValueError):
            vals.append(0.0)

    # Avoid all zeros
    if sum(vals) <= 0:
        vals = [0.25, 0.25, 0.25, 0.25]

    colors = ["purple", "red", "gold", "green"]

    segments = []
    for v, c in zip(vals, colors):
        segments.append(
            html.Div(
                style={
                    "flex": v + 0.01,  # +0.01 so tiny values still show
                    "backgroundColor": c,
                }
            )
        )

    return html.Div(segments, className="single-skill-sparkline")

# -------------------------------
# Tab2 Frame4: What-If Reroll Model
# -------------------------------

def fmt_pct_01(p):
    """0-1 -> '99.6%' style string."""
    if p is None or pd.isna(p):
        return "—"
    try:
        return f"{float(p) * 100.0:.1f}%"
    except Exception:
        return "—"

def fmt_pct_100(p):
    """0-100 -> '12.5%' style string."""
    if p is None or pd.isna(p):
        return "—"
    try:
        return f"{float(p):.1f}%"
    except Exception:
        return "—"

def fmt_expected_rolls(prob_01):
    """Expected rolls = 1/p."""
    try:
        p = float(prob_01)
    except Exception:
        return "—"

    if p <= 0:
        return "—"

    return f"{(1.0 / p):.1f}"

def get_reroll_slot_options(s1, s2, s3, s4):
    """
    Build dropdown options in the exact current Frame 1 order.
    Example:
      S1: All
      S2: Stu
      S3: Des
      S4: Whi
    """
    vals = [s1, s2, s3, s4]
    out = []

    for idx, sc in enumerate(vals, start=1):
        if not sc:
            continue
        out.append({
            "label": f"S{idx}: {sc}",
            "value": idx,
        })

    return out

def get_valid_reroll_targets(class_code: str, selected_skills: list[str], reroll_slot: int) -> list[str]:
    """
    Valid reroll targets for a given slot:
      - base skills for class
      - exclude the skill being replaced
      - exclude incompatibles to the other 3 fixed skills
      - exclude duplicates already present in the other 3 fixed skills
    """
    if not class_code or not selected_skills or len(selected_skills) != 4 or not reroll_slot:
        return []

    if reroll_slot not in (1, 2, 3, 4):
        return []

    base_skills = get_base_skills_for_class(class_code)
    if not base_skills:
        return []

    idx = reroll_slot - 1
    skill_being_replaced = selected_skills[idx]
    fixed_others = [s for i, s in enumerate(selected_skills) if i != idx and s]

    pool = filtered_skill_pool(base_skills, fixed_others)

    # remove duplicates already present in the other 3 slots
    used_elsewhere = set(fixed_others)
    pool = [s for s in pool if s not in used_elsewhere]

    # explicitly exclude the skill being replaced
    pool = [s for s in pool if s != skill_being_replaced]

    return sorted(set(pool))

def calc_target_roll_probability(class_code: str, selected_skills: list[str], reroll_slot: int, target_skill: str) -> float:
    """
    One-reroll probability of hitting exactly target_skill:

      P(target) =
          P(rarity for slot)
          *
          1 / (# valid skills of that rarity for this slot)

    VALID skills are:
      - valid for class
      - compatible with other 3 fixed skills
      - not already used in the other 3 fixed skills
      - not the skill being replaced
    """
    if not class_code or not selected_skills or len(selected_skills) != 4:
        return 0.0

    if reroll_slot not in (1, 2, 3, 4):
        return 0.0

    if not target_skill:
        return 0.0

    valid_targets = get_valid_reroll_targets(class_code, selected_skills, reroll_slot)
    if target_skill not in valid_targets:
        return 0.0

    target_rarity = str(skill_rarity_map.get(target_skill, "")).strip()
    if target_rarity not in ("Common", "Rare", "Epic"):
        return 0.0

    rarity_odds = slot_odds_map.get(int(reroll_slot), {})
    p_rarity = float(rarity_odds.get(target_rarity, 0.0))
    if p_rarity <= 0:
        return 0.0

    same_rarity_targets = [
        sc for sc in valid_targets
        if str(skill_rarity_map.get(sc, "")).strip() == target_rarity
    ]
    n_same_rarity = len(same_rarity_targets)
    if n_same_rarity <= 0:
        return 0.0

    return p_rarity / n_same_rarity

def build_reroll_result_row(class_code: str, selected_skills: list[str], reroll_slot: int, target_skill: str) -> dict:
    """
    Build one modeled result row for Table B.
    """
    out_skills = list(selected_skills)
    out_skills[reroll_slot - 1] = target_skill

    bundle = get_class_bundle(class_code)
    row_idx = find_combo_index(bundle, class_code, out_skills) if bundle is not None else None
    if row_idx is None:
        return {
            "skill_list": out_skills,
            "raw_rating": None,
            "rating_pctile": None,
            "delta_rating": None,
            "delta_pctile": None,
            "roll_prob": calc_target_roll_probability(class_code, selected_skills, reroll_slot, target_skill),
            "changed_slot": reroll_slot,
            "target_skill": target_skill,
        }

    modeled = build_combo_row(bundle, class_code, row_idx)

    # current build lookup
    current_idx = find_combo_index(bundle, class_code, selected_skills)
    current_row = build_combo_row(bundle, class_code, current_idx) if current_idx is not None else None

    raw_rating = modeled.get("raw_rating")
    rating_pctile = modeled.get("rating_pctile")

    delta_rating = None
    delta_pctile = None

    if current_row is not None:
        try:
            delta_rating = float(raw_rating) - float(current_row["raw_rating"])
        except Exception:
            delta_rating = None

        try:
            delta_pctile = float(rating_pctile) - float(current_row["rating_pctile"])
        except Exception:
            delta_pctile = None

    return {
        "skill_list": out_skills,
        "raw_rating": raw_rating,
        "rating_pctile": rating_pctile,
        "delta_rating": delta_rating,
        "delta_pctile": delta_pctile,
        "roll_prob": calc_target_roll_probability(class_code, selected_skills, reroll_slot, target_skill),
        "changed_slot": reroll_slot,
        "target_skill": target_skill,
    }

def make_skill_chip(sc: str, changed: bool = False):
    """
    Compact icon + 3-letter code chip.
    """
    if not sc:
        return html.Span("—")

    border = "2px solid #c0392b" if changed else "1px solid #999"
    bg = "#fdecea" if changed else "#f7f7f7"

    return html.Div(
        [
            html.Img(
                src=f"/assets/skill_icons/{sc}.png",
                title=get_full_skill_name(sc),
                style={"height": "16px", "width": "16px"},
            ),
            html.Span(sc, style={"fontWeight": "bold", "fontSize": "11px"}),
        ],
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "4px",
            "padding": "2px 3px",
            "border": border,
            "borderRadius": "5px",
            "backgroundColor": bg,
            "whiteSpace": "nowrap",
            "lineHeight": "1.0",
        },
    )
    
def build_tab2_frame4_current_summary(class_code, s1, s2, s3, s4):
    """
    Table A: current build summary.
    """
    if not class_code or not s1 or not s2 or not s3 or not s4:
        return html.Div("Select a full 4-skill build to model rerolls.")

    bundle = get_class_bundle(class_code)
    if bundle is None:
        return html.Div(f"No data available for class {class_code}.")

    skills_in_order = [s1, s2, s3, s4]
    row_idx = find_combo_index(bundle, class_code, skills_in_order)
    if row_idx is None:
        return html.Div("Current build was not found in the class bundle.")

    row = build_combo_row(bundle, class_code, row_idx)

    header_style = {
        "backgroundColor": "#34495e",
        "color": "white",
        "padding": "4px 3px",
        "border": "1px solid #001f3f",
        "textAlign": "center",
        "fontWeight": "bold",
    }
    cell_style = {
        "padding": "4px 3px",
        "border": "1px solid #001f3f",
        "textAlign": "center",
        "verticalAlign": "middle",
        "fontSize": "14px",
    }
    
    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("S1", style=header_style),
                        html.Th("S2", style=header_style),
                        html.Th("S3", style=header_style),
                        html.Th("S4", style=header_style),
                        html.Th("Rating", style=header_style),
                        html.Th("%ile", style=header_style),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(make_skill_chip(s1), style=cell_style),
                            html.Td(make_skill_chip(s2), style=cell_style),
                            html.Td(make_skill_chip(s3), style=cell_style),
                            html.Td(make_skill_chip(s4), style=cell_style),
                            html.Td(f"{float(row['raw_rating']):.1f}", style=cell_style),
                            html.Td(fmt_pct_01(row["rating_pctile"]), style=cell_style),
                        ]
                    )
                ]
            ),
        ],
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "minWidth": "0px",
        }
    )

def build_tab2_frame4_results_table(class_code, s1, s2, s3, s4, reroll_slot, target_skills):
    """
    Table B: modeled outcomes.
    """

    def status_box(message):
        return html.Div(
            message,
            style={
                "display": "inline-block",
                "width": "fit-content",
                "maxWidth": "100%",
                "border": "1px solid #ccc",
                "borderRadius": "4px",
                "padding": "8px 12px",
                "backgroundColor": PANEL_BG,
            },
        )

    if not class_code or not s1 or not s2 or not s3 or not s4:
        return status_box("Complete the build above to see modeled outcomes.")

    if not reroll_slot:
        return status_box("Select a slot to reroll.")

    if not target_skills:
        return status_box("Select up to 3 target skills to model.")

    selected_skills = [s1, s2, s3, s4]
    results = [
        build_reroll_result_row(class_code, selected_skills, reroll_slot, t)
        for t in target_skills[:3]
    ]

    # sort best rating first, blanks last
    results = sorted(
        results,
        key=lambda r: (r["raw_rating"] is None, -(r["raw_rating"] or -9999)),
    )

    header_style = {
        "backgroundColor": "#2e8b57",
        "color": "white",
        "padding": "4px 6px",
        "border": "1px solid #1f5f3b",
        "textAlign": "center",
        "fontWeight": "bold",
        "fontSize": "12px",
    }
    cell_style = {
        "padding": "4px 3px",
        "border": "1px solid #1f5f3b",
        "textAlign": "center",
        "verticalAlign": "middle",
        "fontSize": "12px",
    }

    body_rows = []
    for r in results:
        skills = r["skill_list"]
        changed_slot = r["changed_slot"]

        body_rows.append(
            html.Tr(
                [
                    html.Td(make_skill_chip(skills[0], changed=(changed_slot == 1)), style=cell_style),
                    html.Td(make_skill_chip(skills[1], changed=(changed_slot == 2)), style=cell_style),
                    html.Td(make_skill_chip(skills[2], changed=(changed_slot == 3)), style=cell_style),
                    html.Td(make_skill_chip(skills[3], changed=(changed_slot == 4)), style=cell_style),
                    html.Td("—" if r["raw_rating"] is None else f"{float(r['raw_rating']):.1f}", style=cell_style),
                    html.Td(fmt_pct_01(r["rating_pctile"]), style=cell_style),
                    html.Td("—" if r["delta_rating"] is None else f"{float(r['delta_rating']):+.1f}", style=cell_style),
                    html.Td("—" if r["delta_pctile"] is None else f"{float(r['delta_pctile']) * 100.0:+.1f}%", style=cell_style),
                    html.Td(fmt_pct_100(float(r["roll_prob"]) * 100.0), style=cell_style),
                    html.Td(fmt_expected_rolls(r["roll_prob"]), style=cell_style),
                ]
            )
        )

    cumulative_prob = calc_any_target_roll_probability(results)
    cumulative_exp_rolls = fmt_expected_rolls(cumulative_prob)

    summary_label_style = {
        **cell_style,
        "textAlign": "right",
        "fontWeight": "bold",
        "backgroundColor": "#eef6f0",
    }

    summary_value_style = {
        **cell_style,
        "fontWeight": "bold",
        "backgroundColor": "#eef6f0",
    }

    body_rows.append(
        html.Tr(
            [
                html.Td("", style=cell_style),
                html.Td("", style=cell_style),
                html.Td("", style=cell_style),
                html.Td("", style=cell_style),
                html.Td("", style=cell_style),
                html.Td("", style=cell_style),
                html.Td("", style=cell_style),
                html.Td("Any Selected Target:", style=summary_label_style),
                html.Td(fmt_pct_100(cumulative_prob * 100.0), style=summary_value_style),
                html.Td(cumulative_exp_rolls, style=summary_value_style),
            ]
        )
    )

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("S1", style=header_style),
                        html.Th("S2", style=header_style),
                        html.Th("S3", style=header_style),
                        html.Th("S4", style=header_style),
                        html.Th("Rating", style=header_style),
                        html.Th("%ile", style=header_style),
                        html.Th("Δ Rating", style=header_style),
                        html.Th("Δ %ile", style=header_style),
                        html.Th("Roll %", style=header_style),
                        html.Th("Exp. Rolls", style=header_style),
                    ]
                )
            ),
            html.Tbody(body_rows),
        ],
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "minWidth": "860px",
        }
    )
    
def calc_any_target_roll_probability(results: list[dict]) -> float:
    """
    Because each target skill is a distinct mutually exclusive reroll outcome,
    cumulative one-roll probability is the sum of the individual probabilities.
    """
    total = 0.0
    for r in results:
        try:
            total += float(r.get("roll_prob", 0.0))
        except Exception:
            pass
    return total
    

# -------------------------------
# Tab2 Frame3: Class rating histogram columns + selected build percentile
# -------------------------------

# Matches: cl_rating_hist_9950
_CL_RATING_HIST_RE = re.compile(r"^cl_rating_hist_(\d+)$")


def _tab2_f3_class_hist_cols(hero_codes_df: pd.DataFrame):
    """
    Returns list of (bp_int, colname) sorted by bp_int.
    bp_int is 0..10000 (basis points of percentile).
    """
    cols = []
    for c in hero_codes_df.columns:
        m = _CL_RATING_HIST_RE.match(str(c))
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return cols


def _extract_class_hist_xy(row: pd.Series, hist_cols):
    """
    hist_cols: list of (bp_int, colname).
    x returned in percent (0..100), y as float counts.
    """
    xs = []
    ys = []
    for bp, c in hist_cols:
        xs.append(bp / 100.0)  # bp -> %
        v = row.get(c, 0)
        v = 0 if pd.isna(v) else v
        ys.append(float(v))
    return xs, ys


def _safe_pctile_from_rawratings(raw_ratings: np.ndarray, target_val: float) -> float:
    """
    Percentile rank in [0,100]. Uses <= for tie-friendly behavior.
    """
    if raw_ratings is None or len(raw_ratings) == 0 or np.isnan(target_val):
        return float("nan")
    rr = raw_ratings
    rr = rr[~np.isnan(rr)]
    if rr.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(rr <= float(target_val)))


def _get_raw_ratings_for_class(class_code: str):
    bundle = get_class_bundle(class_code)
    if bundle is None:
        return np.array([], dtype=np.float32)
    return np.asarray(bundle["raw_rating"], dtype=np.float32)
    
def build_tab2_f3_rating_percentile_histogram(
    hero_codes_df: pd.DataFrame,
    class_code: str,
    s1: str, s2: str, s3: str, s4: str,
):
    fig = go.Figure()

    # -------------------------
    # Local helpers (frills)
    # -------------------------
    def _rag_for_pct(pct: float):
        """
        Return (line_rgba, box_rgba, font_rgb) based on percentile.
          Red   : < 90
          Amber : 90.. < 99
          Green : >= 99
        """
        if np.isnan(pct):
            return ("rgba(80,80,80,0.6)", "rgba(230,230,230,0.7)", "rgb(0,0,0)")

        if pct >= 99.0:
            return ("rgba(0,140,0,0.85)", "rgba(0,140,0,0.12)", "rgb(0,90,0)")
        elif pct >= 90.0:
            return ("rgba(200,140,0,0.85)", "rgba(200,140,0,0.14)", "rgb(120,80,0)")
        else:
            return ("rgba(180,0,0,0.85)", "rgba(180,0,0,0.12)", "rgb(120,0,0)")

    # --- Guards ---
    if hero_codes_df is None or hero_codes_df.empty or not class_code:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=420,
            margin=dict(l=40, r=20, t=70, b=40),
        )
        fig.add_annotation(
            text="Missing class selection.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    # --- Class row from db_hero_codes.csv ---
    class_col = "Code"
    class_row_df = hero_codes_df.loc[
        hero_codes_df[class_col].astype(str).str.strip() == str(class_code).strip()
    ]
    if class_row_df.empty:
        fig.update_layout(template="plotly_white", xaxis_title="Percentile", yaxis_title="Count")
        fig.add_annotation(
            text=f"Class '{class_code}' not found in db_hero_codes.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    class_row = class_row_df.iloc[0]

    # --- Histogram columns (ONLY from hero_codes_df) ---
    hist_cols = _tab2_f3_class_hist_cols(hero_codes_df)
    if not hist_cols:
        fig.update_layout(template="plotly_white", xaxis_title="Percentile", yaxis_title="Count")
        fig.add_annotation(
            text="No cl_rating_hist_#### columns found in db_hero_codes.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    xs, ys = _extract_class_hist_xy(class_row, hist_cols)

    selected_pct = float("nan")
    target_raw   = float("nan")
    skill_code   = None
    
    # ----------------------------------------------------
    # NEW: Filter histogram to only show >= T2_F3_MIN_TO_SHOW
    # ----------------------------------------------------
    x_min = float(T2_F3_MIN_TO_SHOW)
    xs_f = []
    ys_f = []
    for x, y in zip(xs, ys):
        if x >= x_min:
            xs_f.append(x)
            ys_f.append(y)

    # Guard if filter removes everything (shouldn't, but safe)
    if not xs_f:
        xs_f, ys_f = xs, ys

    # --- Bar histogram: filtered ---
    fig.add_trace(
        go.Bar(
            x=xs_f,
            y=ys_f,
            name="All builds",
        )
    )

    # --- THIS build vertical marker (lookup in classes/XX_final_data.npz) ---
    selected_pct = float("nan")
    skill_code = None

    if class_code and s1 and s2 and s3 and s4:
        skill_label = canonical_skill_string(s1, s2, s3, s4)
        skill_code = f"{class_code}{skill_label}"

        bundle = get_class_bundle(class_code)
        if bundle is not None:
            row_idx = find_combo_index(bundle, class_code, [s1, s2, s3, s4])
            rr_all = _get_raw_ratings_for_class(class_code)

            if row_idx is not None:
                target_raw = float(bundle["raw_rating"][row_idx])

            if not np.isnan(target_raw):
                selected_pct = _safe_pctile_from_rawratings(rr_all, target_raw)
                
    # --- Vertical line + annotation (RAG shading + right-justified) ---
    shapes = []
    annotations = []

    y_max = max(ys_f) if ys_f else 0
    y_line = y_max * 1.05 if y_max > 0 else 1

    # draw the line if we found the raw rating
    if np.isfinite(target_raw):
        line_rgba, box_rgba, font_rgb = _rag_for_pct(selected_pct)
    
        # Vertical line at RAW RATING
        shapes.append(
            dict(
                type="line",
                x0=target_raw, x1=target_raw,
                y0=0, y1=y_line,
                line=dict(color=line_rgba, width=4),
            )
        )
    
        # Label shows both raw rating and percentile (percentile is computed vs rr_all)
        pct_txt = "n/a" if not np.isfinite(selected_pct) else f"{selected_pct:.2f}%"
        annotations.append(
            dict(
                x=target_raw,
                y=y_line,
                xanchor="right",
                yanchor="bottom",
                xshift=-6,
                text=f"This build: {target_raw:.1f}  ({pct_txt})",
                showarrow=False,
                font=dict(size=12, color=font_rgb),
                bgcolor=box_rgba,
                bordercolor=line_rgba,
                borderwidth=1,
            )
        )

    # --- Layout ---
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=420,
        margin=dict(l=40, r=20, t=70, b=50),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f"Percentile Distribution by Rating"
        + (f"<br>{' | '.join([strip_parens(get_full_skill_name(s)) for s in [s1, s2, s3, s4] if s])}" if skill_code else "")
    )

    # X-axis now starts at T2_F3_MIN_TO_SHOW
    fig.update_xaxes(
        title_text="Raw Rating Percentile Bucket",
        range=[x_min, 100],
        tickmode="linear",
        tick0=(int(x_min / 5) * 5),  # keeps ticks aligned to 5s
        dtick=5,
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="Count of Builds",
        rangemode="tozero",
        showticklabels=False,
    )

    return fig

# ========== Helpers for Tab 3 ==========
def build_single_skill_summary_block(class_code: str, skill_code: str):
    df = get_single_skill_assess_df(class_code)
    if df is None or df.empty or "sk_name" not in df.columns:
        return f"No assessment data found for {class_code}."

    row = df.loc[df["sk_name"] == skill_code]
    if row.empty:
        return f"No data for skill {skill_code} in class {class_code}."
    row = row.iloc[0]

    full_name_raw = get_full_skill_name(skill_code)

    # Core metrics
    tier      = row.get("skill_tier", None)
    r_max     = row.get("r_max", None)
    r90_raw   = (row.get("r90_pct", 0.0) or 0.0)
    r95builds_raw = (
        row.get("pct_95", None)
        if "pct_95" in row
        else (row.get("95ile_pct", 0.0) or 0.0)
    )
    roland_tag = row.get("roland_tag", None) 

    # Bucket breakdown (0–100 values)
    nq_raw    = (row.get("nq_pct", 0.0) or 0.0)
    sub80_raw = (row.get("sub80_pct", 0.0) or 0.0)
    pct80_95_raw = (
        row.get("pct_80_95", None)
        if "pct_80_95" in row
        else (row.get("80_95_pct", 0.0) or 0.0)
    )
    pct95_raw = (
        row.get("pct_95", None)
        if "pct_95" in row
        else (row.get("95ile_pct", 0.0) or 0.0)
    )

    # Extra stats for P3
    sk_count   = row.get("sk_count", None)
    rank_mx    = row.get("rank_mx", None)
    rank_95ile = row.get("rank_95ile", None)
    rank_r90   = row.get("rank_r90", None)
    rank_avg   = row.get("rank_mx_95ile_r90_avg", None)

    total_skills = len(df)  # for "of <N>" text

    # ---------- P1: class + skill line + rating row ----------

    class_info = class_meta.get(class_code, {})
    class_name = class_info.get("name", class_code)
    class_icon_src = class_info.get("icon_src", f"/assets/hero_classes/{class_code}.png")
    skill_icon_src = f"/assets/skill_icons/{skill_code}.png"

    tier_icon = single_skill_tier_icon(tier)


    class_skill_line = html.Div(
        [
            html.Span(
                [
                    html.Img(
                        src=class_icon_src,
                        className="class-icon",
                        title=class_name,
                    ),
                    html.Span(" ", className="class-name-space"),
                    html.Span(class_name, className="class-name"),
                    html.Span(" ", className="class-name-space"),
                ],
                className="class-label-block",
            ),
            html.Span(
                [
                    html.Button(
                        html.Img(
                            src=skill_icon_src,
                            className="skill-icon",
                            title=full_name_raw,
                            style={"width": "32px", "height": "32px"},
                        ),
                        id={  # optional but recommended: lets Tab3 headline icon route too
                            "type": "detail-skill-icon-btn",
                            "skill": skill_code,
                            "context": "tab3-headline",
                        },
                        n_clicks=0,
                        style=CLICKABLE_ICON_BUTTON_STYLE,  # <-- removes gray box
                    ),
                    html.Span(" ", className="skill-label-icon-space"),
                    format_skill_name_with_info(full_name_raw),
                ],
                className="single-skill-class-skill-block",
            ),
        ],
        className="single-skill-class-line ssi-headline-row",
    )


    # Roland tag info (0=No, 5=Maybe, else Yes)
    roland_val = row.get("roland_tag", 0) or 0
    
    if roland_val == 0:
        roland_text = "No"
        roland_color = "#c0392b"
        roland_sub = ""
    elif roland_val == 5:
        roland_text = "Maybe"
        roland_color = "#f1c40f"
        roland_sub = "Priority 5"
    else:
        roland_text = "YES"
        roland_color = "#27ae60"
        roland_sub = f"Priority {roland_val}"
    
    roland_block = html.Span(
        [
            html.Img(
                src="/assets/detail_icons/veteran_head.png",
                className="roland-icon",
                title="Roland Priority",
            ),
            html.Span(
                [
                    html.Span(
                        roland_text,
                        className="headline-rating-number",
                        style={"color": roland_color},
                    ),
                    html.Span(
                        roland_sub,
                        className="headline-sub-label",
                        style={"marginLeft": "8px"},
                    ),
                ],
                style={"display": "inline-flex", "alignItems": "flex-end"},
            ),
        ],
        className="roland-block",
    )

    # Rating row: r_max, r90_pct, 95+ builds, tier icon
    def fmt(val):
        if val is None or pd.isna(val):
            return "—"
        return f"{float(val):.1f}"

    rating_block = html.Div(
        [
            html.Span(
                [
                    html.Span(fmt(r_max), className="headline-rating-number"),
                    html.Span(" ", className="headline-space"),
                    html.Span("Rating (max)", className="headline-sub-label"),
                ],
                className="headline-rating-block",
            ),
    
            html.Span("|", className="headline-separator"),
    
            html.Span(
                [
                    html.Span(f"{r90_raw:.1f}%", className="headline-rating-number"),
                    html.Span(" ", className="headline-space"),
                    html.Span("Good Builds (>90 Rating)", className="headline-sub-label"),
                ],
                className="headline-pct-block",
            ),
    
            html.Span("|", className="headline-separator"),
    
            html.Span(
                [
                    html.Span(f"{r95builds_raw:.1f}%", className="headline-rating-number"),
                    html.Span(" ", className="headline-space"),
                    html.Span("Builds in 95+ %ile for Class", className="headline-sub-label"),
                ],
                className="headline-pct-block",
            ),
    
            html.Span("|", className="headline-separator"),
    
            # tier icon (kept as-is)
            (tier_icon if tier_icon is not None else html.Span()),
    
            html.Span("|", className="headline-separator"),
    
            roland_block,
        ],
        className="ssi-rating-row",
    )

    # ---------- P2: bucket bar chart (left) ----------

    labels = [
        "Non-Qualified Build",
        "Qualified, < 80th %ile",
        "80–95th %ile",
        "95th+ %ile",
    ]
    values = [nq_raw, sub80_raw, pct80_95_raw, pct95_raw]
    colors = ["purple", "red", "gold", "green"]

    text = [f"{v:.1f}% of builds" for v in values]
    textpos = ["inside" if v >= 25 else "outside" for v in values]

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=colors),
                text=text,
                textposition=textpos,
            )
        ]
    )

    # Compute a dynamic X-axis upper bound: snap to the next 20% above
    # the largest bucket (but never above 100).
    max_val = max(values) if values else 0.0
    if max_val <= 0:
        x_max = 100.0
    else:
        # snap to nearest 20% above the max value
        x_max = float(20.0 * np.ceil(max_val / 20.0))
        x_max = min(100.0, max(20.0, x_max))

    bar_fig.update_layout(
        margin=dict(l=160, r=20, t=10, b=40),
        xaxis_title="% of Builds",
        yaxis_title="",
        xaxis=dict(range=[0, x_max]),
        yaxis=dict(ticklabelstandoff=10),
        bargap=0.25,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    bar_block = html.Div(
        [
            html.Div(
                "How this Skill Appears in Rated Builds (Rating Percentile | Percent of Builds)",
                style={
                    "backgroundColor": "#34495e",
                    "color": "white",
                    "padding": "4px 8px",
                    "fontWeight": "bold",
                    "fontSize": "13px",
                    "textAlign": "center",
                    "borderTopLeftRadius": "4px",
                    "borderTopRightRadius": "4px",
                },
            ),
            dcc.Graph(
                figure=bar_fig,
                config={"displayModeBar": False},
                style={"height": "260px", "width": "100%"},
            ),
        ],
        style={
            "border": "1px solid #cccccc",
            "borderRadius": "4px",
            "padding": "0px 4px 4px 4px",
            "backgroundColor": "#fafafa",
        },
    )


    # ---------- P3: stats table (right) ----------

    def fmt_int_or_dash(v):
        if v is None or pd.isna(v):
            return "—"
        try:
            return f"{int(v)}"
        except (TypeError, ValueError):
            return str(v)

    header_style = {
        "border": "1px solid #2f4f4f",
        "padding": "6px 8px",
        "fontWeight": "bold",
        "textAlign": "center",
        "backgroundColor": "#2f4f4f",
        "color": "white",
        "fontSize": "13px",
    }
    cell_style = {
        "border": "1px solid #2f4f4f",
        "padding": "6px 8px",
        "textAlign": "center",
        "fontSize": "13px",
    }
    label_style = {**cell_style, "textAlign": "left"}

    stats_rows = [
        ("Unique Combinations", fmt_int_or_dash(sk_count)),
        ("Available Skills", fmt_int_or_dash(total_skills)),
        (
            "Rank: Max Rating",
            f"{fmt_int_or_dash(rank_mx)} of {fmt_int_or_dash(total_skills)}"
            if rank_mx is not None and not pd.isna(rank_mx)
            else "—",
        ),
        (
            "Rank: 95%ile Builds",
            f"{fmt_int_or_dash(rank_95ile)} of {fmt_int_or_dash(total_skills)}"
            if rank_95ile is not None and not pd.isna(rank_95ile)
            else "—",
        ),
        (
            "Rank: \"Good\" Builds",
            f"{fmt_int_or_dash(rank_r90)} of {fmt_int_or_dash(total_skills)}"
            if rank_r90 is not None and not pd.isna(rank_r90)
            else "—",
        ),
        (
            "Avg Rank",
            f"{float(rank_avg):.2f}"
            if rank_avg is not None and not pd.isna(rank_avg)
            else "—",
        ),
    ]

    stats_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Attribute", style=header_style),
                        html.Th("Data", style=header_style),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(label, style=label_style),
                            html.Td(val, style=cell_style),
                        ]
                    )
                    for label, val in stats_rows
                ]
            ),
        ],
        className="single-skill-summary-table",
        style={
            "borderCollapse": "collapse",
            "minWidth": "260px",
        },
    )

    # ---------- assemble P1 + P2/P3 ----------

    return html.Div(
        [
            class_skill_line,
            rating_block,
            html.Hr(style={"margin": "8px 0"}),
            html.Div(
                [
                    html.Div(bar_block, className="t3-f2-bar-wrap"),
                    html.Div(stats_table, className="t3-f2-stats-wrap"),
                ],
                className="t3-f2-summary-row",
            ),
        ],
        className="single-skill-summary-block",
    )

# Tab 3 - Frame 3 - Single Skill Histogram:

_HIST_RE = re.compile(r"^rating_hist_(\d+)$")

def _tab3_skill_col(df: pd.DataFrame) -> str:
    if "sk_name" in df.columns:
        return "sk_name"
    if "skill_code" in df.columns:
        return "skill_code"
    raise KeyError("[tab3_hist] missing skill id column (expected sk_name or skill_code)")

def _tab3_hist_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        m = _HIST_RE.match(str(c))
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return cols  # list of (bin_int, colname)

def _extract_xy_rating(row: pd.Series, hist_cols):
    # hist_cols is list of (bin_int, colname)
    xs = [float(b) for b, _ in hist_cols]
    ys = []
    for _, c in hist_cols:
        v = row.get(c, 0)
        v = 0 if pd.isna(v) else v
        ys.append(float(v))
    return xs, ys

def build_tab3_skill_ranking_distribution(skill_df: pd.DataFrame, selected_skill: str):
    fig = go.Figure()

    # --- Guards ---
    if skill_df is None or skill_df.empty or not selected_skill:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=420,
            margin=dict(l=10, r=20, t=70, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.add_annotation(
            text="Missing class or skill selection.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    sk_col = _tab3_skill_col(skill_df)

    # --- Histogram columns (auto-detect) ---
    hist_cols = _tab3_hist_cols(skill_df)
    if not hist_cols:
        fig.update_layout(
            title="Skill Ranking Distribution",
            template="plotly_white",
            xaxis_title="Rating",
            yaxis_title="Count",
        )
        fig.add_annotation(
            text="No rating_hist_* columns found in assess file.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    # --- Selected row ---
    s = str(selected_skill).strip()
    sel_df = skill_df.loc[skill_df[sk_col].astype(str).str.strip() == s]
    if sel_df.empty:
        fig.update_layout(
            title="Skill Ranking Distribution",
            template="plotly_white",
            xaxis_title="Rating",
            yaxis_title="Count",
        )
        fig.add_annotation(
            text=f"Selected skill '{s}' not found in assess file (col={sk_col}).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    sel_row = sel_df.iloc[0]

    # --- Tier 1 rows ---
    tier1_df = skill_df.copy()
    if "skill_tier" in tier1_df.columns:
        tier1_df["skill_tier"] = pd.to_numeric(tier1_df["skill_tier"], errors="coerce")
        tier1_df = tier1_df.loc[tier1_df["skill_tier"] == 1]
    else:
        tier1_df = tier1_df.iloc[0:0]

    tier1_df = tier1_df.loc[tier1_df[sk_col].astype(str).str.strip() != s].head(4)

    # --- Add selected trace (SOLID) ---
    xs, ys = _extract_xy_rating(sel_row, hist_cols)
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=f"{s}: ratings of builds",
            line=dict(width=5),
        )
    )

    # --- Add Tier 1 traces (FAINT DOTTED) ---
    for _, r in tier1_df.iterrows():
        sc = str(r[sk_col]).strip()
        xs2, ys2 = _extract_xy_rating(r, hist_cols)
        fig.add_trace(
            go.Scatter(
                x=xs2, y=ys2,
                mode="lines",
                name=f"{sc}: ratings of builds",
                line=dict(width=3, dash="dot"),
                opacity=0.5,
            )
        )

    # --- Dynamic Y axis: next 500 above max ---
    # Collect y-values from selected + tier1
    y_max_raw = 0.0
    y_max_raw = max(y_max_raw, max(ys) if ys else 0.0)

    for _, r in tier1_df.iterrows():
        _, ys2 = _extract_xy_rating(r, hist_cols)
        if ys2:
            y_max_raw = max(y_max_raw, max(ys2))

    # Round up to next 500, add 5% headroom
    y_target = y_max_raw * 1.05
    y_max = int(np.ceil(y_target / 500.0) * 500.0)
    if y_max <= 0:
        y_max = 500
    
    # --- Title + tips bar (like Tab2 vibe) ---
    fig.update_layout(
        template="plotly_white",
    
        # --- Force white background ---
        paper_bgcolor="white",
        plot_bgcolor="white",
    
        # --- Force black text everywhere ---
        font=dict(color="black"),
        title_font=dict(color="black"),
        legend_font=dict(color="black"),
    
        # Keep your existing layout values
        height=420,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # --- Custom X-axis ticks (35–100) ---
    tick_vals = list(range(35, 101, 5))
    tick_text = []
    
    for v in tick_vals:
        if v == 35 or v == 100:
            tick_text.append("")          # no label
        elif v == 40:
            tick_text.append("< 40")
        elif v == 95:
            tick_text.append("95+")
        else:
            tick_text.append(f"{v-5}–{v}")


    # --- Force axes exactly as requested ---
    fig.update_xaxes(
        range=[35, 100],
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        showline=True,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )
    
    # Keep the y-axis readable: no more than ~4 major labels.
    if y_max <= 0:
        y_dtick = 500
    else:
        y_dtick = float(np.ceil((y_max / 4.0) / 500.0) * 500.0)
        y_dtick = max(500.0, y_dtick)
    
    fig.update_yaxes(
        range=[0, y_max],
        tickmode="linear",
        tick0=0,
        dtick=y_dtick,
        rangemode="tozero",
    )
    
    fig.update_xaxes(title_text="Raw Rating")
    fig.update_yaxes(title_text="Count of Builds per Rating Group")

    return fig

def tab4_split_codes(blob: str, n_skills: int) -> list[str]:
    """
    Split a 3/6/12-char skill blob into 3-letter codes.
    Example:
      'AcrAll' -> ['Acr', 'All']
      'AcrAllEPlWhi' -> ['Acr', 'All', 'EPl', 'Whi']
    """
    if blob is None or pd.isna(blob):
        return []

    s = str(blob).strip()
    # Some build strings may include the class prefix, e.g. "G2AcrAll..."
    if len(s) == 2 + (n_skills * 3):
        s = s[2:]
    
    expected_len = n_skills * 3
    if len(s) != expected_len:
        return []
    
    return [s[i:i+3] for i in range(0, expected_len, 3)]

# ===== TAB4:  Class Summaries and Example Builds ======

def tab4_skill_icon_img(skill_code: str, size_px: int = 28):
    return html.Img(
        src=f"/assets/skill_icons/{skill_code}.png",
        title=strip_parens(get_full_skill_name(skill_code)),
        style={"height": f"{size_px}px", "width": f"{size_px}px"},
    )

def tab4_row_colors(class_code: str):
    """
    GT = green, BT = blue, RT = red
    """
    c = str(class_code).strip().upper()
    if c.startswith("G"):
        return TAB4_ROW_BG_GREEN, TAB4_ROW_BG_GREEN_CLASS
    if c.startswith("B"):
        return TAB4_ROW_BG_BLUE, TAB4_ROW_BG_BLUE_CLASS
    if c.startswith("R"):
        return TAB4_ROW_BG_RED, TAB4_ROW_BG_RED_CLASS
    return "white", "#eeeeee"

def tab4_skill_button(class_code: str, skill_code: str):
    if not skill_code:
        return "—"

    return html.Button(
        [
            html.Div(
                [
                    tab4_skill_icon_img(skill_code, size_px=30),
                    html.Span(skill_code, style={"fontSize": "12px", "fontWeight": "600"}),
                ],
                style=TAB4_ICON_ROW_STYLE,
            ),
        ],
        id={
            "type": "tab4-key-skill-btn",
            "class_code": class_code,
            "skill": skill_code,
        },
        n_clicks=0,
        title=skill_label(skill_code),
        style=TAB4_LINK_BUTTON_STYLE,
    )
    
def tab4_core_button(class_code: str, core_blob: str):
    codes = tab4_split_codes(core_blob, 2)
    if len(codes) != 2:
        return "—"

    parts = []
    for c in codes:
        parts.append(tab4_skill_icon_img(c, size_px=26))
        parts.append(html.Span(c, style={"fontSize": "12px", "fontWeight": "600"}))

    return html.Button(
        [
            html.Div(parts, style=TAB4_ICON_ROW_STYLE),
        ],
        id={
            "type": "tab4-core-btn",
            "class_code": class_code,
            "core": core_blob,
        },
        n_clicks=0,
        title=" / ".join([skill_label(c) for c in codes]),
        style=TAB4_LINK_BUTTON_STYLE,
    )

def tab4_build_button(class_code: str, build_blob: str):
    codes = tab4_split_codes(build_blob, 4)
    if len(codes) != 4:
        return "—"

    parts = []
    for c in codes:
        parts.append(tab4_skill_icon_img(c, size_px=22))
        parts.append(html.Span(c, style={"fontSize": "11px", "fontWeight": "600"}))

    return html.Button(
        [
            html.Div(parts, style=TAB4_ICON_ROW_STYLE),
        ],
        id={
            "type": "tab4-build-btn",
            "class_code": class_code,
            "build": build_blob,
        },
        n_clicks=0,
        title=" / ".join([skill_label(c) for c in codes]),
        style=TAB4_LINK_BUTTON_STYLE,
    )

def tab4_class_label_cell(class_code: str, class_name: str):
    return html.Div(
        [
            html.Img(
                src=f"/assets/hero_classes/{class_code}.png",
                style={"height": "34px", "width": "34px"},
                title=class_name,
            ),
            html.Span(class_name),
        ],
        style=TAB4_CLASS_LINE_STYLE,
    )

def tab4_fmt_rating(v):
    if v is None or pd.isna(v):
        return "—"
    try:
        return f"{float(v):.1f}"
    except Exception:
        return str(v)


def tab4_class_summary_table(sort_mode: str = "class_asc"):
    """
    Build the desktop class-summary table from db_hero_codes.csv.
    """
    df = hero_codes_df.copy()

    if "rating_max" in df.columns:
        df["rating_max_num"] = pd.to_numeric(df["rating_max"], errors="coerce")
    else:
        df["rating_max_num"] = np.nan

    sort_mode = (sort_mode or "class_tier").lower()

    # derive family tier order from class code, e.g. G2 -> family G, tier 2
    df["class_family"] = df["Code"].astype(str).str[0]
    df["class_tier_num"] = pd.to_numeric(df["Code"].astype(str).str[1:], errors="coerce")

    if sort_mode == "rating_desc":
        df = df.sort_values(
            ["rating_max_num", "Hero_Class"],
            ascending=[False, True],
            kind="mergesort",
        )
    elif sort_mode == "class_alpha":
        df = df.sort_values(
            ["Hero_Class"],
            ascending=[True],
            kind="mergesort",
        )
    else:
        # default: By Class & Tier
        # family order B, G, R to match your code conventions
        family_order = {"B": 1, "G": 2, "R": 3}
        df["class_family_order"] = df["class_family"].map(family_order).fillna(99)

        df = df.sort_values(
            ["class_family_order", "class_tier_num", "Hero_Class"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        
    rows = []

    for _, row in df.iterrows():
        class_code = row["Code"]
        class_name = row["Hero_Class"]

        row_bg, class_bg = tab4_row_colors(class_code)

        rows.append(
            html.Tr(
                [
                    html.Td(
                        tab4_class_label_cell(class_code, class_name),
                        style={**TAB4_CLASS_CELL_STYLE, "backgroundColor": class_bg},
                    ),
                    html.Td(
                        tab4_fmt_rating(row.get("rating_max")),
                        style={**TAB4_MAX_CELL_STYLE, "backgroundColor": row_bg},
                    ),
                    html.Td(
                        tab4_skill_button(class_code, row.get("key_skill")),
                        style={**TAB4_TD_STYLE, "backgroundColor": row_bg},
                    ),
                    html.Td(
                        tab4_core_button(class_code, row.get("ex_core1")),
                        style={**TAB4_TD_STYLE, "backgroundColor": row_bg},
                    ),
                    html.Td(
                        tab4_core_button(class_code, row.get("ex_core2")),
                        style={**TAB4_TD_STYLE, "backgroundColor": row_bg},
                    ),
                    html.Td(
                        tab4_build_button(class_code, row.get("build_apex")),
                        style={**TAB4_TD_STYLE, "backgroundColor": row_bg},
                    ),
                    html.Td(
                        tab4_build_button(class_code, row.get("build_ex1")),
                        style={**TAB4_TD_STYLE, "backgroundColor": row_bg},
                    ),
                    html.Td(
                        tab4_build_button(class_code, row.get("build_ex2")),
                        style={**TAB4_TD_STYLE, "backgroundColor": row_bg},
                    ),
                ]
            )
        )

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Class", style=TAB4_TH_STYLE_CLASS),
                        html.Th("Max Rating", style=TAB4_TH_STYLE_CLASS),
                        html.Th("Key Skill", style=TAB4_TH_STYLE_KEY),
                        html.Th("Example Core 1", style=TAB4_TH_STYLE_CORE),
                        html.Th("Example Core 2", style=TAB4_TH_STYLE_CORE),
                        html.Th("APEX Build", style=TAB4_TH_STYLE_BUILD),
                        html.Th("Example Build 1", style=TAB4_TH_STYLE_BUILD),
                        html.Th("Example Build 2", style=TAB4_TH_STYLE_BUILD),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style=TAB4_TABLE_STYLE,
    )
    
def tab5_load_markdown() -> str:
    try:
        return TAB5_MD_PATH.read_text(encoding="utf-8")
    except Exception as e:
        return (
            "# Using This Tool\n\n"
            f"_Help text file not found: {TAB5_MD_PATH}_\n\n"
            f"_Error: {type(e).__name__}: {e}_"
        )

def tab5_header():
    return html.H4("Using This Tool", style=TITLE_BANNER_STYLE)

def tab5_body_markdown():
    return dcc.Markdown(
        tab5_load_markdown(),
        style={
            "background": "white",
            "color": "black",
            "border": "1px solid #ccc",
            "borderRadius": "4px",
            "padding": "12px 14px",
            "maxWidth": "980px",
            "lineHeight": "1.35",
            "fontSize": "13px",
        },
    )

# ---------- DASH APP ----------

# --- Login / Authorized Users

# OLD
# app = Dash(__name__, suppress_callback_exceptions=True)

# With Discord Integration
server = Flask(__name__)
server.secret_key = os.environ["FLASK_SECRET_KEY"]

# Tell Flask it is behind Render's proxy
server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Safer session cookie settings for OAuth redirect flows
server.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

app = Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True
)

oauth = OAuth(server)

discord = oauth.register(
    name="discord",
    client_id=os.environ["DISCORD_CLIENT_ID"],
    client_secret=os.environ["DISCORD_CLIENT_SECRET"],
    access_token_url="https://discord.com/api/oauth2/token",
    authorize_url="https://discord.com/oauth2/authorize",
    api_base_url="https://discord.com/api/",
    client_kwargs={"scope": "identify"},
    token_endpoint_auth_method="client_secret_post",
)

# ALLOWED_DISCORD_IDS = {
#     "779058036936802384", # Peetee1138
# }

def load_authorized_discord_ids() -> set[str]:
    if not AUTHORIZED_USERS_FILE.exists():
        return set()

    authorized_ids = set()

    with open(AUTHORIZED_USERS_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # skip blank lines
            if not line:
                continue

            # skip full-line comments
            if line.startswith("#"):
                continue

            # allow inline comments, e.g.:
            # 123456789012345678  # Peetee1138
            line = line.split("#", 1)[0].strip()

            if line:
                authorized_ids.add(line)

    return authorized_ids

def finalize_login(discord_id: str, username: str):
    authorized_ids = load_authorized_discord_ids()
    is_approved = discord_id in authorized_ids

    session["discord_user"] = {
        "id": discord_id,
        "username": username,
    }

    # Approved users always get in
    if is_approved:
        session["authorized"] = True
        session["guest_mode"] = False
        session["guest_uses"] = 0

        log_login_attempt(
            username=username,
            discord_id=discord_id,
            approved=True,
            note="approved",
        )
        return redirect("/")

    # Not approved yet -> allow a few free uses
    guest_uses = int(session.get("guest_uses", 0))

    if guest_uses < FREE_USES_ALLOWED:
        guest_uses += 1
        session["guest_uses"] = guest_uses
        session["authorized"] = True
        session["guest_mode"] = True

        log_login_attempt(
            username=username,
            discord_id=discord_id,
            approved=False,
            note=f"guest_use_{guest_uses}_of_{FREE_USES_ALLOWED}",
        )
        return redirect("/")

    # Free uses exhausted
    session["authorized"] = False
    session["guest_mode"] = False

    log_login_attempt(
        username=username,
        discord_id=discord_id,
        approved=False,
        note="guest_limit_reached",
    )
    return redirect("/not-approved")
    
    
def log_login_attempt(username: str, discord_id: str, approved: bool, note: str = "") -> None:
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    print(f"[DEBUG] entering log_login_attempt", flush=True)
    print(f"[DEBUG] LOGIN_LOG_FILE={LOGIN_LOG_FILE}", flush=True)

    file_exists = LOGIN_LOG_FILE.exists()

    with open(LOGIN_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp_utc", "username", "discord_id", "approved", "note"])

        writer.writerow([timestamp_utc, username, discord_id, approved, note])
        f.flush()

    print(
        f"[LOGIN ATTEMPT] time={timestamp_utc} "
        f"user={username} id={discord_id} approved={approved} note={note}",
        flush=True
    )

@server.route("/login", methods=["GET", "HEAD"])
def login():
    # Ignore HEAD requests so they don't create a fake OAuth state
    if request.method == "HEAD":
        return ("", 200)

    # Clear any stale OAuth/session state before starting a fresh login
    session.clear()

    redirect_uri = url_for("callback", _external=True)
    print(
        f"[LOGIN START] path=/login method={request.method} remote_addr={request.remote_addr}",
        flush=True
    )
    return discord.authorize_redirect(redirect_uri)
    
@server.route("/callback")
def callback():
    try:
        token = discord.authorize_access_token()
        print(f"[DEBUG] token keys={list(token.keys())}", flush=True)
        print(f"[DEBUG] token_type={token.get('token_type')}", flush=True)

        access_token = token.get("access_token")
        if not access_token:
            print("[LOGIN ERROR] No access_token returned from Discord", flush=True)
            return redirect("/login-failed")

        user_resp = requests.get(
            "https://discord.com/api/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        user_resp.raise_for_status()
        user = user_resp.json()

        print(f"[DEBUG] username={user.get('username')} id={user.get('id')}", flush=True)

    except Exception as e:
        print(f"[LOGIN ERROR] callback failed: {e}", flush=True)
        session.clear()
        return redirect("/login-failed")

    discord_id = str(user.get("id", "")).strip()
    username = user.get("username", "unknown")

    return finalize_login(
        discord_id=discord_id,
        username=username,
    )

@server.route("/login-failed")
def login_failed():
    session.clear()
    return """
    <h3>Login failed</h3>
    <p>Your Discord login session got out of sync.</p>
    <p>Please close this tab and start again from the login link below.</p>
    <p><a href="/login">Start Discord login again</a></p>
    """
    
# PRODUCTION
    # if discord_id in ALLOWED_DISCORD_IDS:
    #     session["authorized"] = True
    #     return redirect("/")

    # session["authorized"] = False
    # return redirect("/not-approved")

# TESTING
    # session["authorized"] = True
    # return redirect("/")

@server.route("/not-approved")
def not_approved():
    return f"""
    <h3>Thanks for signing in.</h3>
    <p>You have used your {FREE_USES_ALLOWED} free guest logins for this alpha.</p>
    <p>Please ping me on Discord if you'd like a reminder added.</p>
    <p>Once I approve your Discord ID, you can log in again and should get access.</p>
    <p><a href="/login">Try again</a></p>
    """
    
@server.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@server.before_request
def protect_app():
    public_paths = {
        "/login",
        "/callback",
        "/not-approved",
        "/logout",
        "/login-failed",
    }

    if request.path in public_paths:
        return

    if request.path.startswith("/assets/"):
        return

    if request.path.startswith("/_dash-"):
        if not session.get("authorized"):
            return redirect("/login")
        return

    if request.path == "/favicon.ico":
        return

    if not session.get("authorized"):
        return redirect("/login")
        
# --- SERVER INFORMATION - Activate below for _deploy versions
server = app.server

# --- Begin body of Dash app

app.title = "Skills UI — Shop Titans Skills Ratings - PoC"

def _is_available(val):
    return str(val).strip().lower() in ("true", "yes", "1", "y")

def make_layout_tab1():
    return html.Div(
        style={
            "margin": "10px 40px 10px 10px",
            "fontFamily": "Arial",
            "backgroundColor": APP_BG,
        },
        children=[
                # html.H2("Skills UI — 2-Skill Explorer (Proof of Concept)"),

            # -------------------------------------------------------
            # ROW 1: Frame 1 (left) + Frame 2 (right)
            # -------------------------------------------------------
            html.Div(
                className="t1-main-row",
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "20px",
                    "alignItems": "flex-start",
                },
                children=[
        
                    # ---------- FRAME 1 (Left Column) ----------
                    html.Div(
                        style=LEFT_FRAME_STYLE,
                        children=[
                            html.H4("Step 1: Pick a Class"),

                            # Class row
                            html.Label("Hero Class"),
                            html.Div(
                                style=ROW_CLASS_STYLE,
                                children=[
                                    dcc.Dropdown(
                                        id="hero-class",
                                        options=[
                                            {
                                                "label": row["Hero_Class"],
                                                "value": row["Code"],
                                                "disabled": not _is_available(row.get("available", True)),
                                            }
                                            for _, row in hero_codes_df.iterrows()
                                        ],
                                        value="G2",
                                        clearable=False,
                                        style=LEFT_FRAME_DROPDOWN_STYLE,
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                    html.Img(
                                        id="hero-class-icon",
                                        style={
                                            "height": "32px",
                                            "width": "32px",
                                        },
                                    ),
                                ],
                            ),

                            html.Br(),

                            html.H4("Step 2: Pick Core Skills"),

                            # Skill 1 row
                            html.Label("Skill 1"),
                            html.Div(
                                style=ROW_SKILL_STYLE,
                                children=[
                                    dcc.Dropdown(
                                        id="skill1",
                                        clearable=False,
                                        placeholder="Select Skill 1...",
                                        style=LEFT_FRAME_DROPDOWN_STYLE,
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                    html.Button(
                                        html.Img(
                                            id="skill1-icon",
                                            style={"height": "32px"},
                                        ),
                                        id="skill1-icon-btn",
                                        n_clicks=0,
                                        style=CLICKABLE_ICON_BUTTON_STYLE,
                                    ),
                                ],
                            ),

                            html.Br(),

                            # Skill 2 row
                            html.Label("Skill 2"),
                            html.Div(
                                style=ROW_SKILL_STYLE,
                                children=[
                                    dcc.Dropdown(
                                        id="skill2",
                                        clearable=False,
                                        placeholder="Select Skill 2...",
                                        style=LEFT_FRAME_DROPDOWN_STYLE,
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                    html.Button(
                                        html.Img(
                                            id="skill2-icon",
                                            style={"height": "32px"},
                                        ),
                                        id="skill2-icon-btn",
                                        n_clicks=0,
                                        style=CLICKABLE_ICON_BUTTON_STYLE,
                                    ),
                                ],
                            ),
                            
                            # Summary of what we picked
                            html.Br(),
                            html.Div(id="selection-summary"),
                            html.Br(),

                            # Dropdown to limit heatmap span
                            html.H4("Step 3: Limit Heat Map (default All)"),
                            dcc.Dropdown(
                                id="heatmap-skill-filter",
                                options=[
                                    {"label": "All", "value": "all"},
                                    {"label": "Top 10", "value": "top10"},
                                    {"label": "Top 20", "value": "top20"},  # NEW
                                    {"label": "Epic & Rare", "value": "epic_rare"},
                                ],
                                value="top10",
                                clearable=False,
                                style={"width": "180px"},
                                className="t1-heatmap-filter",
                            ),
                            html.Br(),
                            html.Label("Exclude Skills (optional)"),
                            dcc.Dropdown(
                                id="heatmap-exclude-skills",
                                multi=True,
                                placeholder="Select skills to exclude...",
                                style={"width": "250px"},
                            ),
                        ],
                    ),

                    # ---------- FRAME 2 (Right Column) ----------
                    html.Div(
                        className="t1-f2-table-panel",
                        style={"flex": "1 1 450px", "minWidth": "450px"},
                        children=[
                            html.H4(id="step3-title", style=TITLE_BANNER_STYLE),

                            dash_table.DataTable(
                                id="combo-table",
                                style_table={
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                    "width": "100%",
                                },
                                style_cell={
                                    "fontSize": 12,
                                    "textAlign": "left",
                                    "color": "black",
                                },
                                style_header={
                                    "fontFamily": "Arial",
                                    "fontSize": 14,
                                    "fontWeight": "bold",
                                    "textAlign": "center",          # center headings
                                    "backgroundColor": "#666666",   # darker gray
                                    "color": "white",
                                },
                                fixed_rows={"headers": True},
                                style_cell_conditional=[
                                    {
                                        "if": {"column_id": "rank"},
                                        "width": "6%",
                                        "textAlign": "center",
                                    },
                                    {
                                        "if": {"column_id": "s3_full"},
                                        "width": "31%",
                                    },
                                    {
                                        "if": {"column_id": "s4_full"},
                                        "width": "31%",
                                    },
                                    {
                                        "if": {"column_id": "raw_rating"},
                                        "width": "12%",
                                        "textAlign": "center",
                                    },
                                    {
                                        "if": {"column_id": "net_rating"},
                                        "width": "13%",
                                        "textAlign": "center",
                                    },
                                ],

                                style_data_conditional=rarity_styles + [
                                    {
                                        "if": {"column_id": "rank"},
                                        "color": "#0645AD",          # link blue
                                        "textDecoration": "underline",
                                        "cursor": "pointer",
                                    },
                                    {
                                        "if": {"column_id": "s3_full"},
                                        "color": "#0645AD",
                                        "textDecoration": "underline",
                                        "cursor": "pointer",
                                    },
                                    {
                                        "if": {"column_id": "s4_full"},
                                        "color": "#0645AD",
                                        "textDecoration": "underline",
                                        "cursor": "pointer",
                                    },
                                ],
                                hidden_columns=["_s3_code", "_s4_code", "s3", "s4"],
                                columns=[
                                    {"name": "Rank", "id": "rank"},
                                    {"name": "Skill 3", "id": "s3_full"},
                                    {"name": "Skill 4", "id": "s4_full"},
                                    {"name": "Raw Rating", "id": "raw_rating"},
                                    {"name": "Net Rating", "id": "net_rating"},
                                    {"name": "_s3_code", "id": "_s3_code"},
                                    {"name": "_s4_code", "id": "_s4_code"},
                                    {"name": "s3", "id": "s3"},
                                    {"name": "s4", "id": "s4"},
                                ],
                            )
                        ],
                    ),
                ],
            ),

            # -------------------------------------------------------
            # ROW 2: FRAME 3 (Bottom full-width chart)
            # -------------------------------------------------------
            html.Br(),
            html.H4(
                id="heatmap-title",
                style={**TITLE_BANNER_STYLE, "borderRadius": "2px"},
            ),
            html.Div(
                [
                    html.Span("Tips: ", style={"fontWeight": "bold"}),
                    html.Span("Click a heatmap cell to open Skill Combo Detail. "),
                    html.Span("Click any skill icon to jump to Single Skill Info. "),
                    html.Span("Click and drag to zoom, double-click to reset view."),
                ],
                style={
                    "fontSize": "11px",
                    "marginTop": "2px",
                    "marginBottom": "4px",
                },
            ),
            dcc.Graph(
                id="combo-heatmap",
                figure=go.Figure(),
                config={"displayModeBar": False},
                style={"height": "800px", "width": "100%"},
            ),
            html.Div(
                "Legend: xx = same skill (diagonal), zz = incompatible or no data for that pair, "
                "n/q = Non-Qualifying build",
                style={
                    "fontSize": "11px",
                    "fontStyle": "italic",
                    "marginTop": "4px",
                },
            ),
        ],
    )


def make_layout_tab2():
    return html.Div(
        style={
            "margin": "10px 40px 10px 10px",
            "paddingBottom": "180px",           
            "fontFamily": "Arial",
            "backgroundColor": APP_BG,
        },  
        id="combo-detail-container",
        children=[
            # -------------------------------------------------------
            # ROW 1: Frame 1 (left) + Frame 2 (right)
            # -------------------------------------------------------
            html.Div(
                style={"display": "flex", "gap": "20px"},
                children=[

                    # ---------- FRAME 1 (Left Column) ----------
                    html.Div(
                        style=LEFT_FRAME_STYLE,
                        children=[
                            html.Label("Hero Class"),
                            html.Div(
                                style=ROW_CLASS_STYLE,
                                children=[
                                    dcc.Dropdown(
                                        id="detail-hero-class",
                                        options=[
                                            {
                                                "label": row["Hero_Class"],
                                                "value": row["Code"],
                                                "disabled": not _is_available(row.get("available", True)),
                                            }
                                            for _, row in hero_codes_df.iterrows()
                                        ],
                                        value="G2",
                                        clearable=False,
                                        style=LEFT_FRAME_DROPDOWN_STYLE,
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                    html.Img(
                                        id="detail-hero-class-icon",
                                        style={
                                            "height": "32px",
                                            "width": "32px",
                                        }
                                    ),
                                ],
                            ),

                            html.Br(),
                            html.Div(
                                "Put skills in slot order to enable What If... analysis",
                                style={
                                    "fontSize": "11px",
                                    "fontStyle": "italic",
                                    "color": "#555",
                                    "marginBottom": "8px",
                                },
                            ),
                            
                            # ----- Skill rows 1–4 -----
                            *[
                                html.Div(
                                    children=[
                                        html.Label(f"Skill {i}"),
                                        html.Div(
                                            style=ROW_SKILL_STYLE,
                                            children=[
                                                dcc.Dropdown(
                                                    id=f"detail-skill{i}",
                                                    placeholder=f"Select Skill {i}...",
                                                    clearable=False,
                                                    style=LEFT_FRAME_DROPDOWN_STYLE,
                                                ),
                                                html.Button(
                                                    html.Img(
                                                        id=f"detail-skill{i}-icon",
                                                        style={"height": "32px"},
                                                    ),
                                                    id=f"detail-skill{i}-btn",
                                                    n_clicks=0,
                                                    style=CLICKABLE_ICON_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        html.Br(),
                                    ]
                                )
                                for i in range(1, 5)
                            ],  # ✅ IMPORTANT: close the list comprehension + comma
                            
                            # ✅ Now add your jump link as a separate element
                            html.Div(
                                [
                                    html.A(
                                        "Jump to What If...",
                                        href="#tab2-whatif-anchor",
                                        style={
                                            "fontSize": "12px",
                                            "fontWeight": "bold",
                                            "color": "#0645AD",
                                            "textDecoration": "underline",
                                            "cursor": "pointer",
                                        },
                                    )
                                ],
                                style={"marginTop": "-6px", "marginBottom": "8px"},
                            ),
                        ],
                    ),

                    # ---------- FRAME 2 (Right Column) ----------
                    html.Div(
                        style={"flex": "1", "minWidth": "400px"},
                        children=[
                            html.H4(
                                "Skill Combo Detail (sorted alphabetically)",
                                style=TITLE_BANNER_STYLE,
                            ),
                            html.Div(
                                id="combo-detail-text",
                                style={
                                    "border": "1px solid #ccc",
                                    "borderRadius": "4px",
                                    "padding": "8px",
                                    "minHeight": "120px",
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # -------------------------------------------------------
            # ROW 2: FRAME 3 (Bottom full-width chart)
            # -------------------------------------------------------
            html.Hr(),
            html.H4(
                "Compare of this Build to all Builds for Class",
                style={**TITLE_BANNER_STYLE, "borderRadius": "2px"},
            ),
            html.Div(
                [
                    html.Span("Tips: ", style={"fontWeight": "bold"}),
                    html.Span("(1) Chart shows a histogram of the Ratings of all unique 4-skill builds. "),
                    html.Span("(2) Vertical line shows the Rating of this build & percentile (in this class). "),
                ],
                style={"fontSize": "11px", "marginTop": "4px"},
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="combo-detail-histogram",
                        figure=go.Figure(),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "500px", "width": "100%"},
                    ),
                ],
                className="t2-f3-chart-wrap",
            ),
            # -------------------------------------------------------
            # ROW 3: FRAME 4 (Bottom full-width chart)
            # -------------------------------------------------------
            html.Hr(),
            html.Div(id="tab2-whatif-anchor"),
            html.H4(
                "What-If Reroll Model",
                style={**TITLE_BANNER_STYLE, "borderRadius": "2px"},
            ),
            html.Div(
                [
                    html.Span("Use this section to test a reroll target for one slot. ", style={"fontSize": "11px"}),
                    html.Span("Odds assume: first roll rarity by slot odds, then pick uniformly from valid skills of that rarity.", style={"fontSize": "11px"}),
                ],
                style={"marginBottom": "8px"},
            ),

            # Table A
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                },
                children=[
                    html.Div(
                        id="tab2-frame4-current-summary",
                        style={
                            "width": "fit-content",
                            "maxWidth": "100%",
                            "minWidth": "0",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "padding": "8px",
                            "marginBottom": "12px",
                            "overflowX": "auto",
                            "backgroundColor": PANEL_BG,
                        },
                    )
                ]
            ),
            
            # Controls
            html.Div(
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "16px",
                    "alignItems": "flex-start",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        [
                            html.Label("1) Select slot to reroll"),
                            html.Div("\u00A0", style={"fontSize": "11px", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="reroll-slot-dropdown",
                                clearable=False,
                                placeholder="Select slot...",
                                style={"width": "220px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("2) Select up to 3 target skills", style={"display": "block", "marginBottom": "2px"}),
                            html.Div("Choose up to 3.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="reroll-target-skills",
                                multi=True,
                                placeholder="Select target skills...",
                                style={"width": "520px", "maxWidth": "100%"},
                            ),
                        ]
                    )
                ],
            ),

            # Table B
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                },
                children=[
                    html.Div(
                        id="tab2-frame4-results",
                        style={
                            "width": "fit-content",
                            "maxWidth": "100%",
                            "minWidth": "0",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "padding": "8px",
                            "overflowX": "auto",
                            "backgroundColor": PANEL_BG,
                        },
                    )
                ]
            ),            
        ],
    )

# ========== Tab 3: Single Skill Info layout ==========

def make_layout_tab3():
    return html.Div(
        style={
            "margin": "10px 40px 10px 10px",
            "fontFamily": "Arial",
            "backgroundColor": APP_BG,
        },
        id="single-skill-tab",
        children=[
            # ROW 1: Frame 1 (left) + Frame 2 (right)
            html.Div(
                style={
                    "display": "flex",
                    "gap": "20px",
                    "flexWrap": "wrap",  # keeps mobile-friendly behavior
                },
                children=[

                    # ---------- FRAME 1: Class + Single Skill ----------
                    html.Div(
                        style=LEFT_FRAME_STYLE,
                        children=[
                            html.H4("Single Skill Explorer"),

                            html.Label("Hero Class"),
                            html.Div(
                                style=ROW_CLASS_STYLE,
                                children=[
                                    dcc.Dropdown(
                                        id="single-skill-class",
                                        options=[
                                            {
                                                "label": row["Hero_Class"],
                                                "value": row["Code"],
                                                "disabled": not _is_available(row.get("available", True)),
                                            }
                                            for _, row in hero_codes_df.iterrows()
                                        ],
                                        value="G2",
                                        clearable=False,
                                        style=LEFT_FRAME_DROPDOWN_STYLE,
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                    html.Img(
                                        id="single-skill-class-icon",
                                        style={"height": "32px", "width": "32px"},
                                    ),
                                ],
                            ),

                            html.Br(),

                            html.Label("Skill"),
                            html.Div(
                                style=ROW_SKILL_STYLE,
                                children=[
                                    dcc.Dropdown(
                                        id="single-skill-select",
                                        placeholder="Select a skill...",
                                        clearable=False,
                                        style=LEFT_FRAME_DROPDOWN_STYLE,
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                    html.Img(
                                        id="single-skill-icon",
                                        style={"height": "32px"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # ---------- FRAME 2: Skill Summary Card ----------
                    html.Div(
                        style={"flex": "2 1 380px", "minWidth": "320px"},
                        children=[
                            html.H4(
                                "Selected Skill Summary",
                                style=TITLE_BANNER_STYLE,
                            ),
                            html.Div(
                                id="single-skill-summary",
                                style={
                                    "border": "1px solid #ccc",
                                    "borderRadius": "4px",
                                    "padding": "8px",
                                    "minHeight": "120px",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            html.Hr(),

            # ---------- ROW 2: FRAME 3: Class-wide Skill Ranking ----------
            html.H4(
                "Skill Rating Distribution",
                id="tab3-frame3-title",
                style={**TITLE_BANNER_STYLE, "borderRadius": "2px"},
            ),
            html.Div(
                [
                    html.Span("Tips: ", style={"fontWeight": "bold"}),
                    html.Span("(1) Solid line = selected skill. "),
                    html.Span("(2) Faint dotted lines = Tier 1 skills for this class. "),
                    html.Span("(3) Click a trace in the legend to hide/show. "),
                    html.Span("(4) Double-click to reset zoom."),
                ],
                style={"fontSize": "11px", "marginTop": "4px"},
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="tab3-frame3-graph",
                        figure=go.Figure(),
                        config={"displayModeBar": False},
                        style={"height": "500px", "width": "100%"},
                    )
                ],
                className="t3-f3-chart-wrap",
            ),
        ],
    )

# ===== TAB4:  Summary of all Classes and example builds/cores

def make_layout_tab4():
    return html.Div(
        style={
            "margin": "10px 20px 10px 10px",
            "padding": "10px",
            "fontFamily": "Arial",
            "backgroundColor": APP_BG,
        },
        id="tab4-class-summary-tab",
        children=[
            html.H4("Class Summary", style=TITLE_BANNER_STYLE),
            html.Div(
                [
                    html.Span("Using this table: ", style={"fontWeight": "bold"}),
                    html.Span("Key Skill: ", style={"fontWeight": "bold"}),
                    html.Span("Skill with the overall highest-rated combinations based on overall data. "),
                    html.Span("Example Cores: ", style={"fontWeight": "bold"}),
                    html.Span("To help you explore, a couple of strong 2-skill cores to consider. "),
                    html.Br(),
                    html.Span("Builds: ", style={"fontWeight": "bold"}),
                    html.Span("Based on the data, Apex = the highest Rated build, "),
                    html.Span("Ex. 1 = strong build with mostly uncommon skills, Ex. 2 = good build with mostly common skills."),
                ],
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Span("Sort Order:", style={"fontWeight": "bold", "marginRight": "8px"}),
                    html.Button(
                        "By Class & Tier",
                        id="tab4-sort-class-tier-btn",
                        n_clicks=0,
                        style={
                            "padding": "6px 10px",
                            "border": "1px solid #666",
                            "borderRadius": "4px",
                            "backgroundColor": "#e8e8e8",
                            "cursor": "pointer",
                            "fontWeight": "600",
                        },
                    ),
                    html.Button(
                        "By Rating",
                        id="tab4-sort-rating-btn",
                        n_clicks=0,
                        style={
                            "padding": "6px 10px",
                            "border": "1px solid #666",
                            "borderRadius": "4px",
                            "backgroundColor": "#f7f7f7",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "By Class (Alpha)",
                        id="tab4-sort-class-alpha-btn",
                        n_clicks=0,
                        style={
                            "padding": "6px 10px",
                            "border": "1px solid #666",
                            "borderRadius": "4px",
                            "backgroundColor": "#f7f7f7",
                            "cursor": "pointer",
                        },
                    ),
                    dcc.Store(id="tab4-sort-mode", data="class_tier"),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "marginBottom": "10px",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                id="tab4-table-wrap",
                style={
                    "overflowX": "auto",
                },
            ),
            html.Br(),
            html.Div(
                "Notes can go here later.",
                style={"fontSize": "12px", "fontStyle": "italic"},
            ),
        ],
    )
    
# ===== TAB5:  User Instructions & FAQ
    
def make_layout_tab5():
    return html.Div(
        style={
            "margin": "10px 40px 10px 10px",
            "fontFamily": "Arial",
            "backgroundColor": APP_BG,
        },
        id="using-this-tool-tab",
        children=[
            tab5_header(),
            html.Div(
                [
                    # Optional: if you later add images to assets, they can live above/between sections
                    # html.Img(src="/assets/help_flow.png", style={"maxWidth": "980px", "width": "100%", "marginBottom": "10px"}),
                    tab5_body_markdown(),
                ],
                style={
                    "display": "flex", 
                    "justifyContent": "flex-start",
                    "backgroundColor": PANEL_BG
                },
            ),
        ],
    )

    
# ========== Overall App Layout ==========

app.layout = html.Div(
    style={"backgroundColor": APP_BG, "minHeight": "100vh"},
    children=[
        dcc.Store(id="selected-combo"),
        dcc.Store(id="pending-tab-nav"),
        dcc.Tabs(
            id="main-tabs",
            value="tab-2skill",
            style={"fontFamily": "Arial"},
            children=[
                dcc.Tab(
                    label="2-Skill Explorer",
                    value="tab-2skill",
                    children=make_layout_tab1(),   # existing UI
                ),
                dcc.Tab(
                    label="Skill Combo Detail",
                    value="tab-combo-detail",
                    children=make_layout_tab2()
                ),
                dcc.Tab(
                    label="Single Skill Info",
                    value="tab-single-skill",
                    children=make_layout_tab3(),
                ),
                dcc.Tab(
                    label="Class Summary",
                    value="tab-class-summary",
                    children=make_layout_tab4(),
                ),
                dcc.Tab(
                    label="Help: Using this Tool",
                    value="using-this-tool-tab",
                    children=make_layout_tab5(),
                ),
            ],
        )
    ]
)

# ---------- CALLBACKS ----------

# 1) Dropdowns for Skill 1 / Skill 2
@app.callback(
    Output("skill1", "options"),
    Output("skill2", "options"),
    Output("skill1", "value"),
    Output("skill2", "value"),
    Input("hero-class", "value"),
    Input("skill1", "value"),
    Input("skill2", "value"),
)
def update_skill_dropdowns(class_code, s1, s2):
    """
    Populate Skill 1 & Skill 2 dropdowns for the 2-skill explorer,
    honoring incompatibilities and always sorting options alphabetically.
    """
    if not class_code:
        return [], [], None, None

    base_skills = get_base_skills_for_class(class_code)
    if not base_skills:
        return [], [], None, None

    # --- initial pools, sorted ---
    pool1 = sorted(filtered_skill_pool(base_skills, [s2] if s2 else []))
    pool2 = sorted(filtered_skill_pool(base_skills, [s1] if s1 else []))

    # --- validate current selections ---
    if s1 not in pool1:
        s1 = pool1[0] if pool1 else None

    if s2 not in pool2:
        defaults2 = [x for x in pool2 if x != s1] if s1 else pool2
        s2 = defaults2[0] if defaults2 else (pool2[0] if pool2 else None)

    # --- rebuild pools in case s1/s2 changed, and sort again ---
    pool1 = sorted(filtered_skill_pool(base_skills, [s2] if s2 else []))
    pool2 = sorted(filtered_skill_pool(base_skills, [s1] if s1 else []))

    opts1 = [{"label": skill_label(s), "value": s} for s in pool1]
    opts2 = [{"label": skill_label(s), "value": s} for s in pool2]

    return opts1, opts2, s1, s2

@app.callback(
    Output("heatmap-exclude-skills", "options"),
    Input("hero-class", "value"),
)
def update_exclude_options(class_code):
    if not class_code:
        return []

    base_skills = get_base_skills_for_class(class_code)
    return [{"label": skill_label(s), "value": s} for s in sorted(base_skills)]

@app.callback(
    Output("step3-title", "children"),
    Output("heatmap-title", "children"),
    Output("selection-summary", "children"),
    Output("combo-table", "data", allow_duplicate=True),
    Output("combo-heatmap", "figure", allow_duplicate=True),
    Input("hero-class", "value"),
    Input("skill1", "value"),
    Input("skill2", "value"),
    Input("heatmap-skill-filter", "value"),
    Input("heatmap-exclude-skills", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_outputs(class_code, skill1, skill2, skill_filter, exclude_skills):
    empty_fig = go.Figure()

    def build_titles():
        full1 = strip_parens(get_full_skill_name(skill1))
        full2 = strip_parens(get_full_skill_name(skill2))

        title3 = [
            "3rd & 4th Skill Options for ",
            html.Span(full1, style={"color": "#FFD700"}),
            " and ",
            html.Span(full2, style={"color": "#FFD700"}),
            " Core",
        ]
        title4 = [
            "Heat Map for ",
            html.Span(full1, style={"color": "#FFD700"}),
            " and ",
            html.Span(full2, style={"color": "#FFD700"}),
            " Core",
        ]
        return title3, title4

    if not class_code or not skill1 or not skill2 or skill1 == skill2:
        step3_title = "3rd & 4th Skill Options"
        heatmap_title = "Heat Map of 3rd/4th Skill Pairs"
        msg = "Pick a hero class plus two different core skills."
        return step3_title, heatmap_title, msg, [], empty_fig

    bundle = get_class_bundle(class_code)
    if bundle is None:
        step3_title, heatmap_title = build_titles()
        msg = f"No data available for class {class_code}."
        return step3_title, heatmap_title, msg, [], empty_fig

    rows1 = get_skill_rows_for_class(bundle, skill1)
    rows2 = get_skill_rows_for_class(bundle, skill2)
    idx = np.intersect1d(rows1, rows2, assume_unique=True)

    step3_title, heatmap_title = build_titles()

    if idx.size == 0:
        msg = f"No builds found containing skills {skill1} and {skill2} for class {class_code}."
        return step3_title, heatmap_title, msg, [], empty_fig

    records = []
    for i in idx:
        skills = get_skill_codes_for_row(bundle, int(i))
        others = [s for s in skills if s not in (skill1, skill2)]
        if len(others) != 2:
            continue

        s3_code, s4_code = sorted(others)
        raw_rating = float(bundle["raw_rating"][i])

        records.append({
            "s3": s3_code,
            "s4": s4_code,
            "raw_rating": raw_rating,
            "net_rating": net_rating_display(bundle["net_rating"][i]),
        })

    if not records:
        msg = f"No valid 3rd/4th-skill rows found for {class_code} with {skill1} + {skill2}."
        return step3_title, heatmap_title, msg, [], empty_fig

    table_df = pd.DataFrame(records).sort_values("raw_rating", ascending=False).reset_index(drop=True)
    table_df["rank"] = range(1, len(table_df) + 1)
    table_df["s3_full"] = table_df["s3"].map(get_full_skill_name).fillna(table_df["s3"])
    table_df["s4_full"] = table_df["s4"].map(get_full_skill_name).fillna(table_df["s4"])

    lookup = {}
    for _, row in table_df.iterrows():
        key = frozenset({row["s3"], row["s4"]})
        lookup[key] = row["net_rating"]

    skills_present = set(table_df["s3"]) | set(table_df["s4"])

    df_assess = get_single_skill_assess_df(class_code)
    axis_skills = build_skill_order_from_assess(df_assess, sorted(list(skills_present)))

    filter_mode = (skill_filter or "all").lower()

    if filter_mode in ("top10", "top20"):
        if df_assess is not None and not df_assess.empty:
            sk_col = "sk_name" if "sk_name" in df_assess.columns else ("skill_code" if "skill_code" in df_assess.columns else None)

            if sk_col and "rank_mx_95ile_r90_avg" in df_assess.columns:
                df_rank = df_assess[[sk_col, "rank_mx_95ile_r90_avg"]].copy()
                df_rank[sk_col] = df_rank[sk_col].astype(str).str.strip()
                df_rank["rank_mx_95ile_r90_avg"] = pd.to_numeric(df_rank["rank_mx_95ile_r90_avg"], errors="coerce")
                df_rank = df_rank[df_rank[sk_col].isin(axis_skills)]
                df_rank = df_rank.dropna(subset=["rank_mx_95ile_r90_avg"])
                df_rank = df_rank.sort_values("rank_mx_95ile_r90_avg", ascending=True)

                top_n = 10 if filter_mode == "top10" else 20
                top_skills = df_rank[sk_col].head(top_n).tolist()
                if len(top_skills) >= 2:
                    axis_skills = top_skills

    elif filter_mode == "epic_rare":
        epic_rare_codes = {
            code for code, rarity in skill_rarity_map.items()
            if str(rarity).strip() in ("Epic", "Rare")
        }
        filtered_axis = [s for s in axis_skills if s in epic_rare_codes]
        if len(filtered_axis) >= 2:
            axis_skills = filtered_axis

    # --- CUSTOM EXCLUDE FILTER ---
    if exclude_skills:
        exclude_set = set(exclude_skills)
        filtered_axis = [s for s in axis_skills if s not in exclude_set]

        if len(filtered_axis) >= 2:
            axis_skills = filtered_axis

    # --- APPLY SAME FILTER TO TABLE (Frame 2) ---
    axis_set = set(axis_skills)

    table_df_filtered = table_df[
        table_df["s3"].isin(axis_set) &
        table_df["s4"].isin(axis_set)
    ].copy()

    # Re-rank after filtering
    table_df_filtered = table_df_filtered.sort_values("raw_rating", ascending=False).reset_index(drop=True)
    table_df_filtered["rank"] = range(1, len(table_df_filtered) + 1)

    # Rebuild UI version
    table_df_ui = table_df_filtered[
        ["rank", "s3", "s4", "s3_full", "s4_full", "raw_rating", "net_rating"]
    ].copy()

    table_df_ui["s3_mobile"] = table_df_ui["s3"].map(
        lambda x: strip_parens(get_full_skill_name(x))
    )
    table_df_ui["s4_mobile"] = table_df_ui["s4"].map(
        lambda x: strip_parens(get_full_skill_name(x))
    )

    table_df_ui["raw_rating"] = table_df_ui["raw_rating"].map(
        lambda v: f"{float(v):.2f}" if pd.notna(v) else "—"
    )

    table_df_ui["_s3_code"] = table_df_filtered["s3"].values
    table_df_ui["_s4_code"] = table_df_filtered["s4"].values

    if len(axis_skills) < 2:
        msg = f"Filter '{filter_mode}' left fewer than 2 skills on the heat map. Try switching back to 'All'."
        return step3_title, heatmap_title, msg, table_df_ui.to_dict("records"), go.Figure()

    n_skills = max(1, len(axis_skills))

    base_text_size = 13
    base_tick_size = 12
    
    if filter_mode == "top10":
        text_font_size = int(base_text_size * 1.5)
        tick_font_size = int(base_tick_size * 1.4)
    elif filter_mode == "top20":
        text_font_size = int(base_text_size * 1.25)
        tick_font_size = int(base_text_size * 1.2)
    else:
        text_font_size = base_text_size
        tick_font_size = base_tick_size
        if n_skills > 16:
            text_font_size = 11
            tick_font_size = 11

    fig_width = 1700
    fig_height = 700
    margin_top = 50

    if filter_mode == "all":
        fig_width = 1700
        fig_height = 700
        margin_top = 50
    elif filter_mode == "epic_rare":
        fig_width = 1250
        fig_height = 675
        margin_top = 60
    elif filter_mode == "top10":
        fig_width = 900
        fig_height = 470
        margin_top = 75
    elif filter_mode == "top20":
        fig_width = 1100
        fig_height = 620
        margin_top = 65

    cell_width_units = 1.0
    icon_sizex = 0.7 * cell_width_units
    icon_sizey = 0.8 / max(n_skills, 8)
    icon_sizey = min(0.12, max(0.05, icon_sizey))
    base_icon_y = 1.03

    if filter_mode == "all":
        icon_y = base_icon_y
    elif filter_mode == "epic_rare":
        icon_y = base_icon_y + 0.01
        icon_sizex *= 0.95
    elif filter_mode == "top10":
        icon_y = base_icon_y + 0.04
        icon_sizex *= 0.80
    elif filter_mode == "top20":
        icon_y = base_icon_y + 0.02
        icon_sizex *= 0.95

    icon_sizey = icon_sizex

    n = len(axis_skills)
    z_numeric = np.full((n, n), np.nan, dtype=float)
    text = [["" for _ in range(n)] for _ in range(n)]

    for i, s_row in enumerate(axis_skills):
        for j, s_col in enumerate(axis_skills):
            if s_row == s_col:
                text[i][j] = "xx"
                continue
            key = frozenset({s_row, s_col})
            val = lookup.get(key)
            if val is None:
                text[i][j] = "zz"
            else:
                if str(val).strip().lower() == "n/q":
                    text[i][j] = "n/q"
                else:
                    num = float(val)
                    text[i][j] = f"{num:.1f}"
                    z_numeric[i, j] = max(50.0, min(100.0, num))

    base_z = np.zeros((n, n), dtype=float)
    heat_base = go.Heatmap(
        z=base_z,
        x=axis_skills,
        y=axis_skills,
        colorscale=[[0.0, "#e0e0e0"], [1.0, "#e0e0e0"]],
        showscale=False,
        xgap=1,
        ygap=2,
    )
    heat_base.meta = "base-heatmap"

    colorscale = [
        [0.0, "rgb(253,141,60)"],
        [0.5, "rgb(255,255,178)"],
        [1.0, "rgb(35,132,67)"],
    ]

    colorbar_cfg = dict(
        title="Rating",
        tickvals=[50, 75, 100],
        ticktext=["50", 75, 100],
        thickness=10,
        len=0.4,
        orientation="h",
        x=0.0,
        xanchor="left",
        y=-0.01,
        yanchor="top",
    )
    if filter_mode in ("top10", "top20"):
        colorbar_cfg["thickness"] = 14
        colorbar_cfg["len"] = 0.45

    heat_num = go.Heatmap(
        z=z_numeric,
        x=axis_skills,
        y=axis_skills,
        text=text,
        texttemplate="%{text}",
        textfont={"size": text_font_size},
        colorscale=colorscale,
        zmin=50,
        zmax=100,
        hovertemplate="S3=%{y}<br>S4=%{x}<br>Rating=%{text}<extra></extra>",
        colorbar=colorbar_cfg,
        xgap=1,
        ygap=1,
    )
    heat_num.meta = "numeric-heatmap"

    fig = go.Figure(data=[heat_base, heat_num])

    fig.update_layout(
        xaxis=dict(
            title="",
            side="top",
            tickmode="array",
            tickvals=axis_skills,
            ticktext=axis_skills,
            tickfont=dict(color="#000000", size=tick_font_size),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickmode="array",
            tickvals=axis_skills,
            ticktext=axis_skills,
            tickfont=dict(color="#000000", size=tick_font_size),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=40, r=20, t=margin_top, b=20),
        width=fig_width,
        height=fig_height,
        clickmode="event+select",
    )

    images = []
    for s in axis_skills:
        images.append(
            dict(
                source=f"/assets/skill_icons/{s}.png",
                xref="x",
                yref="paper",
                x=s,
                y=icon_y,
                sizex=icon_sizex,
                sizey=icon_sizey,
                xanchor="center",
                yanchor="bottom",
                layer="above",
            )
        )

    fig.update_layout(images=images)

    summary = f"Class {class_code}, core skills: {skill1} + {skill2}.  Found {len(table_df_ui)} unique 3rd/4th combos."
    return step3_title, heatmap_title, summary, table_df_ui.to_dict("records"), fig

@app.callback(
    Output("pending-tab-nav", "data", allow_duplicate=True),
    Output("detail-hero-class", "value"),
    Output("detail-skill1", "value"),
    Output("detail-skill2", "value"),
    Output("detail-skill3", "value"),
    Output("detail-skill4", "value"),
    Input("selected-combo", "data"),                # heatmap click → store
    Input("combo-table", "active_cell"),            # rank click
    State("combo-table", "derived_viewport_data"),
    State("hero-class", "value"),
    State("skill1", "value"),
    State("skill2", "value"),
    prevent_initial_call=True,
)
def drive_detail_selection(
    sel_combo,
    active_cell,
    viewport_rows,
    class_code,
    s1,
    s2,
):
    from dash import callback_context, no_update

    try:
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # 1) Rank click in the table
        if trigger_id == "combo-table" and active_cell and active_cell.get("column_id") == "rank":
            if not viewport_rows or not class_code or not s1 or not s2:
                return no_update, no_update, no_update, no_update, no_update, no_update

            row_idx = active_cell.get("row")
            if row_idx is None or row_idx < 0 or row_idx >= len(viewport_rows):
                return no_update, no_update, no_update, no_update, no_update, no_update

            row = viewport_rows[row_idx]
            s3 = row.get("_s3_code")
            s4 = row.get("_s4_code")
            if not s3 or not s4:
                return no_update, no_update, no_update, no_update, no_update, no_update

            return {"tab": "tab-combo-detail"}, class_code, s1, s2, s3, s4

        # 2) Heatmap click (selected-combo updated)
        if trigger_id == "selected-combo" and sel_combo:
            class_code2 = sel_combo.get("class_code")
            core = sel_combo.get("core", [])
            extra = sel_combo.get("extra", [])

            s1_2 = core[0] if len(core) > 0 else None
            s2_2 = core[1] if len(core) > 1 else None
            s3_2 = extra[0] if len(extra) > 0 else None
            s4_2 = extra[1] if len(extra) > 1 else None

            if not (class_code2 and s1_2 and s2_2 and s3_2 and s4_2):
                return no_update, no_update, no_update, no_update, no_update

            return {"tab": "tab-combo-detail"}, class_code2, s1_2, s2_2, s3_2, s4_2

        return no_update, no_update, no_update, no_update, no_update, no_update

    except Exception as e:
        print("[drive_detail_selection] ERROR:", type(e).__name__, e)
        return no_update, no_update, no_update, no_update, no_update, no_update



@app.callback(
    Output("selected-combo", "data"),
    Input("combo-heatmap", "clickData"),
    State("hero-class", "value"),
    State("skill1", "value"),
    State("skill2", "value"),
    prevent_initial_call=True,
)

def on_heatmap_click(clickData, class_code, s1, s2):
    """Store a 4-skill combo when the user clicks a heatmap cell."""
    # Nothing clicked yet
    if not clickData:
        return no_update

    # Missing context
    if not class_code or not s1 or not s2:
        return no_update

    # Extract row/col skill codes from the clicked point
    try:
        pt = clickData["points"][0]
        s4 = pt.get("x")
        s3 = pt.get("y")
    except Exception:
        return no_update

    if not s3 or not s4:
        return no_update

    combo = {
        "class_code": class_code,
        "core": [s1, s2],
        "extra": [s3, s4],
    }

    print("Clicked combo (heatmap):", combo)  # <- watch for this in the terminal

    return combo

@app.callback(
    Output("pending-tab-nav", "data"),
    Output("single-skill-class", "value"),
    Output("single-skill-select", "value"),

    Input("selected-combo", "data"),          # from heatmap click
    Input("combo-table", "active_cell"),      # Rank / skill clicks in table
    Input("skill1-icon-btn", "n_clicks"),     # Tab 1 icons
    Input("skill2-icon-btn", "n_clicks"),
    Input("detail-skill1-btn", "n_clicks"),   # Tab 2 Frame 1 icons
    Input("detail-skill2-btn", "n_clicks"),
    Input("detail-skill3-btn", "n_clicks"),
    Input("detail-skill4-btn", "n_clicks"),
    Input({"type": "detail-skill-icon-btn", "skill": ALL, "context": ALL}, "n_clicks"),  # Tab 2/3 headline & tables

    State("main-tabs", "value"),
    State("hero-class", "value"),
    State("skill1", "value"),
    State("skill2", "value"),
    State("detail-hero-class", "value"),
    State("detail-skill1", "value"),          # selected skills on Tab 2
    State("detail-skill2", "value"),
    State("detail-skill3", "value"),
    State("detail-skill4", "value"),
    State("combo-table", "derived_viewport_data"),  # ✅ page-aware
    State({"type": "detail-skill-icon-btn", "skill": ALL, "context": ALL}, "id"),
    prevent_initial_call=True,
)
def route_tabs_and_single_skill(
    sel_combo,
    active_cell,
    s1_icon_clicks,
    s2_icon_clicks,
    d_s1_btn_clicks,
    d_s2_btn_clicks,
    d_s3_btn_clicks,
    d_s4_btn_clicks,
    detail_icon_clicks,
    current_tab,
    hero_class,
    skill1_code,
    skill2_code,
    detail_class,
    d_s1,
    d_s2,
    d_s3,
    d_s4,
    viewport_rows,
    detail_icon_ids,
):
    """
    Central router for tab changes:

    - Heatmap click or Rank click → Tab 2 (Skill Combo Detail)
    - Skill icons (Tab 1 & Tab 2) → Tab 3 (Single Skill Info)

    IMPORTANT: ignore initial renders (n_clicks == 0).
    """
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger_info = ctx.triggered[0]
    trigger_prop = trigger_info["prop_id"]       # e.g. "skill1-icon-btn.n_clicks" or '{"type":...}.n_clicks'
    trigger_val  = trigger_info["value"]

    trigger_id_str, trigger_attr = trigger_prop.split(".", 1)

    # Helper: treat n_clicks as "real" only when >0
    def _is_real_scalar_click(v):
        try:
            return int(v or 0) > 0
        except Exception:
            return False

    # Helper: pattern matching click list -> any click > 0 (best-effort)
    # NOTE: Dash gives the whole list; without previous state we can’t know which changed.
    def _is_real_pattern_click(vlist):
        if vlist is None:
            return False
        if not isinstance(vlist, (list, tuple)):
            return _is_real_scalar_click(vlist)
        return any(_is_real_scalar_click(v) for v in vlist)

    # ------------------------------------------------------------
    # 0) Tab 2 Frame 1 skill icons → Single Skill Info
    # ------------------------------------------------------------
    if trigger_id_str in ("detail-skill1-btn", "detail-skill2-btn", "detail-skill3-btn", "detail-skill4-btn"):
        if not _is_real_scalar_click(trigger_val):
            return no_update, no_update, no_update
        if not detail_class:
            return no_update, no_update, no_update

        btn_to_skill = {
            "detail-skill1-btn": d_s1,
            "detail-skill2-btn": d_s2,
            "detail-skill3-btn": d_s3,
            "detail-skill4-btn": d_s4,
        }
        sk = btn_to_skill.get(trigger_id_str)
        if not sk:
            return no_update, no_update, no_update

        return {"tab": "tab-single-skill"}, detail_class, sk

    # ------------------------------------------------------------
    # 1) Tab 1 icons → Single Skill Info
    # ------------------------------------------------------------
    if trigger_id_str == "skill1-icon-btn":
        if not _is_real_scalar_click(trigger_val):
            return no_update, no_update, no_update
        if not hero_class or not skill1_code:
            return no_update, no_update, no_update
        return {"tab": "tab-single-skill"}, hero_class, skill1_code

    if trigger_id_str == "skill2-icon-btn":
        if not _is_real_scalar_click(trigger_val):
            return no_update, no_update, no_update
        if not hero_class or not skill2_code:
            return no_update, no_update, no_update
        return {"tab": "tab-single-skill"}, hero_class, skill2_code

    # ------------------------------------------------------------
    # 2) Clicks in S3/S4 table
    # ------------------------------------------------------------
    if trigger_id_str == "combo-table":
        # During paging/sorting, Dash can fire active_cell updates with None column_id
        if not active_cell:
            return no_update, no_update, no_update

        col_id = active_cell.get("column_id")
        row_idx = active_cell.get("row")

        if not col_id or row_idx is None:
            return no_update, no_update, no_update

        if not viewport_rows or row_idx < 0 or row_idx >= len(viewport_rows):
            return no_update, no_update, no_update

        row = viewport_rows[row_idx]

        # 2a) Rank click → go to Combo Detail tab
        if col_id == "rank":
            return {"tab": "tab-combo-detail"}, no_update, no_update

        # 2b) Skill 3 / Skill 4 click → go to Single Skill Info
        if col_id == "s3_full":
            sk = row.get("_s3_code")
        elif col_id == "s4_full":
            sk = row.get("_s4_code")
        else:
            return no_update, no_update, no_update

        if not hero_class or not sk:
            return no_update, no_update, no_update

        return {"tab": "tab-single-skill"}, hero_class, sk

    # ------------------------------------------------------------
    # 3) Heatmap click → Tab 2 (via selected-combo store)
    # ------------------------------------------------------------
    if trigger_id_str == "selected-combo":
        return no_update, no_update, no_update
    
    # ------------------------------------------------------------
    # 0a) User manually switched to the Single Skill tab
    # ------------------------------------------------------------
    if trigger_id_str == "main-tabs" and trigger_val == "tab-single-skill":
        target_class = detail_class or hero_class or "G2"
        base = get_base_skills_for_class(target_class)
        default_skill = sorted(base)[0] if base else None
        return {"tab": "tab-single-skill"}, target_class, default_skill

    # ------------------------------------------------------------
    # 4) Tab 2/3 headline + table icons (pattern IDs)
    # ------------------------------------------------------------
    # trigger_id_str is JSON for pattern IDs
    try:
        comp_id = json.loads(trigger_id_str)
    except Exception:
        comp_id = None

    if isinstance(comp_id, dict) and comp_id.get("type") == "detail-skill-icon-btn":
        if not _is_real_pattern_click(trigger_val):
            return no_update, no_update, no_update

        sk = comp_id.get("skill")
        if not detail_class or not sk:
            return no_update, no_update, no_update

        return {"tab": "tab-single-skill"}, detail_class, sk

    # ------------------------------------------------------------
    # 5) Fallback (NEVER return None)
    # ------------------------------------------------------------
    return no_update, no_update, no_update

@app.callback(
    Output("main-tabs", "value"),
    Input("pending-tab-nav", "data"),
    State("main-tabs", "value"),
    prevent_initial_call=True,
)
def apply_pending_tab_nav(nav_data, current_tab):

    if not nav_data:
        return no_update

    if isinstance(nav_data, str):
        return nav_data

    target = nav_data.get("tab")

    if not target:
        return no_update

    return target
    
@app.callback(
    Output("skill1-icon", "src"),
    Output("skill2-icon", "src"),
    Input("skill1", "value"),
    Input("skill2", "value"),
)
def update_skill_icons(s1, s2):
    def path(code):
        if not code:
            return no_update
        return f"/assets/skill_icons/{code}.png"

    return path(s1), path(s2)

@app.callback(
    Output("hero-class-icon", "src"),
    Input("hero-class", "value"),
)   
def update_class_icon(class_code):
    if not class_code:
        return no_update
    # expects icons like assets/hero_classes/G2.png, G3.png, etc.
    return f"/assets/hero_classes/{class_code}.png"

@app.callback(
    Output("detail-hero-class-icon", "src"),
    Input("detail-hero-class", "value"),
)
def update_detail_class_icon(class_code):
    if not class_code:
        return no_update
    return f"/assets/hero_classes/{class_code}.png"

@app.callback(
    Output("detail-skill1", "options"),
    Output("detail-skill2", "options"),
    Output("detail-skill3", "options"),
    Output("detail-skill4", "options"),
    Output("detail-skill1", "value", allow_duplicate=True),
    Output("detail-skill2", "value", allow_duplicate=True),
    Output("detail-skill3", "value", allow_duplicate=True),
    Output("detail-skill4", "value", allow_duplicate=True),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
    prevent_initial_call=True,
)
def update_detail_skill_options(class_code, s1, s2, s3, s4):
    """
    Apply incompatibilities to the 4-skill selectors on the detail tab.

    IMPORTANT:
    - returns BOTH options and values
    - preserves currently selected values during refresh
    """
    if not class_code:
        return [], [], [], [], None, None, None, None

    base_skills = get_base_skills_for_class(class_code)
    if not base_skills:
        return [], [], [], [], None, None, None, None

    selected = [s1, s2, s3, s4]
    options_out = []
    values_out = []

    for idx in range(4):
        current_val = selected[idx]
        fixed_others = [s for i, s in enumerate(selected) if i != idx and s]

        # apply normal filtering
        pool = filtered_skill_pool(base_skills, fixed_others)

        # remove duplicates already selected elsewhere
        used_elsewhere = set(fixed_others)
        pool = [s for s in pool if s not in used_elsewhere]

        # IMPORTANT: preserve current value even during refresh/race
        if current_val and current_val not in pool:
            pool.append(current_val)

        pool = sorted(set(pool))

        opts = [{"label": skill_label(s), "value": s} for s in pool]
        options_out.append(opts)

        if current_val in pool:
            values_out.append(current_val)
        elif current_val:
            # Preserve routed values during callback timing/race conditions.
            values_out.append(current_val)
        else:
            # Do not auto-fill Skill 1 as Acr/etc. Leave blank until user/routing sets it.
            values_out.append(None)

    return (
        options_out[0], options_out[1], options_out[2], options_out[3],
        values_out[0], values_out[1], values_out[2], values_out[3],
    )

@app.callback(
    Output("detail-skill1-icon", "src"),
    Output("detail-skill2-icon", "src"),
    Output("detail-skill3-icon", "src"),
    Output("detail-skill4-icon", "src"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def update_detail_skill_icons(s1, s2, s3, s4):
    def path(code):
        if not code:
            return no_update
        return f"/assets/skill_icons/{code}.png"

    return path(s1), path(s2), path(s3), path(s4)

@app.callback(
    Output("reroll-slot-dropdown", "options"),
    Output("reroll-slot-dropdown", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def update_reroll_slot_dropdown(s1, s2, s3, s4):
    opts = get_reroll_slot_options(s1, s2, s3, s4)
    default_val = opts[0]["value"] if opts else None
    return opts, default_val
    
@app.callback(
    Output("reroll-target-skills", "options"),
    Output("reroll-target-skills", "value"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
    Input("reroll-slot-dropdown", "value"),
    Input("reroll-target-skills", "value"),
)
def update_reroll_target_options(class_code, s1, s2, s3, s4, reroll_slot, current_targets):
    if not class_code or not s1 or not s2 or not s3 or not s4 or not reroll_slot:
        return [], []

    selected_skills = [s1, s2, s3, s4]
    valid_targets = get_valid_reroll_targets(class_code, selected_skills, int(reroll_slot))

    opts = [{"label": skill_label(sc), "value": sc} for sc in valid_targets]

    current_targets = current_targets or []
    kept = [sc for sc in current_targets if sc in valid_targets][:3]

    return opts, kept

@app.callback(
    Output("tab2-frame4-current-summary", "children"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def update_tab2_frame4_current_summary(class_code, s1, s2, s3, s4):
    return build_tab2_frame4_current_summary(class_code, s1, s2, s3, s4)

@app.callback(
    Output("tab2-frame4-results", "children"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
    Input("reroll-slot-dropdown", "value"),
    Input("reroll-target-skills", "value"),
)
def update_tab2_frame4_results(class_code, s1, s2, s3, s4, reroll_slot, target_skills):
    return build_tab2_frame4_results_table(
        class_code=class_code,
        s1=s1, s2=s2, s3=s3, s4=s4,
        reroll_slot=reroll_slot,
        target_skills=target_skills,
    )


@app.callback(
    Output("combo-detail-text", "children"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def show_combo_detail(class_code, s1, s2, s3, s4):
    if not class_code or not s1 or not s2 or not s3 or not s4:
        return "Select a hero and four skills to see details."

    bundle = get_class_bundle(class_code)
    if bundle is None:
        return f"No all-data found for class {class_code}."

    row_idx = find_combo_index(bundle, class_code, [s1, s2, s3, s4])
    if row_idx is None:
        skill_code = canonical_full_skill_code(class_code, [s1, s2, s3, s4])
        return f"No matching row in numeric bundle for skill_code={skill_code}"
        
    row = build_combo_row(bundle, class_code, row_idx)
    return build_tab2_frame2_section1(row, class_meta, skill_lookup)
    
# ---------- Tab2, Frame3: DETAIL HISTOGRAM CALLBACK ----------

@app.callback(
    Output("combo-detail-histogram", "figure"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def update_detail_rating_percentile_histogram(class_code, s1, s2, s3, s4):
    hero_codes_df = pd.read_csv(DATA_DIR / "db_hero_codes.csv")
    return build_tab2_f3_rating_percentile_histogram(hero_codes_df, class_code, s1, s2, s3, s4)

from dash import callback, Output, Input, State, no_update

@callback(
    Output("single-skill-select", "options"),
    Input("single-skill-class", "value"),
)
def update_single_skill_dropdown_options(class_code):
    if not class_code:
        return []

    base = get_base_skills_for_class(class_code)  # your existing helper
    # options sorted alphabetically by code (matches your preference)
    base = sorted(set(base))

    return [{"label": skill_label(s), "value": s} for s in base]


@app.callback(
    Output("single-skill-class-icon", "src"),
    Input("single-skill-class", "value"),
)
def update_single_skill_class_icon(class_code):
    if not class_code:
        return no_update
    return f"/assets/hero_classes/{class_code}.png"

@app.callback(
    Output("single-skill-icon", "src"),
    Input("single-skill-select", "value"),
)
def update_single_skill_icon(sc):
    if not sc:
        return no_update
    return f"/assets/skill_icons/{sc}.png"

@app.callback(
    Output("single-skill-summary", "children"),
    Input("single-skill-class", "value"),
    Input("single-skill-select", "value"),
)
def update_single_skill_summary(class_code, skill_code):
    if not class_code or not skill_code:
        return "Select a class and a skill to see details."
    return build_single_skill_summary_block(class_code, skill_code)

@app.callback(
    Output("tab3-frame3-title", "children"),
    Input("single-skill-class", "value"),
    Input("single-skill-select", "value"),
)
def update_tab3_frame3_title(class_code, skill_code):
    if not class_code or not skill_code:
        return "Skill Ranking Distribution"

    class_name = class_meta.get(class_code, {}).get("name", class_code)
    skill_name = strip_parens(get_full_skill_name(skill_code))

    return [
        "Skill Ranking Distribution:",
        html.Br(),
        f"{class_name} | {skill_name}",
    ]
    
@app.callback(
    Output("tab3-frame3-graph", "figure"),
    Input("single-skill-class", "value"),
    Input("single-skill-select", "value"),
)
def update_tab3_histogram(class_code, selected_skill):
    skill_df = get_single_skill_assess_df(class_code)
    return build_tab3_skill_ranking_distribution(skill_df, selected_skill)

@app.callback(
    Output("pending-tab-nav", "data", allow_duplicate=True),

    Output("hero-class", "value", allow_duplicate=True),
    Output("skill1", "value", allow_duplicate=True),
    Output("skill2", "value", allow_duplicate=True),

    Output("detail-hero-class", "value", allow_duplicate=True),
    Output("detail-skill1", "value", allow_duplicate=True),
    Output("detail-skill2", "value", allow_duplicate=True),
    Output("detail-skill3", "value", allow_duplicate=True),
    Output("detail-skill4", "value", allow_duplicate=True),

    Output("single-skill-class", "value", allow_duplicate=True),
    Output("single-skill-select", "value", allow_duplicate=True),

    Input({"type": "tab4-key-skill-btn", "class_code": ALL, "skill": ALL}, "n_clicks"),
    Input({"type": "tab4-core-btn", "class_code": ALL, "core": ALL}, "n_clicks"),
    Input({"type": "tab4-build-btn", "class_code": ALL, "build": ALL}, "n_clicks"),

    prevent_initial_call=True,
)
def tab4_route_clicks(key_clicks, core_clicks, build_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return (
            no_update,
            no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update,
        )

    trigger = ctx.triggered[0]
    trigger_id_str = trigger["prop_id"].split(".")[0]
    trigger_val = trigger["value"]

    def _has_real_click(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                try:
                    if int(x or 0) > 0:
                        return True
                except Exception:
                    pass
            return False
        try:
            return int(v or 0) > 0
        except Exception:
            return False

    # IMPORTANT: ignore rerenders / zero-click refreshes
    if not _has_real_click(trigger_val):
        return (
            no_update,
            no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update,
        )

    try:
        comp_id = json.loads(trigger_id_str)
    except Exception:
        comp_id = None

    if not isinstance(comp_id, dict):
        return (
            no_update,
            no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update,
        )

    typ = comp_id.get("type")
    class_code = comp_id.get("class_code")

    # --- Key Skill -> Tab 3 ---
    if typ == "tab4-key-skill-btn":
        skill_code = comp_id.get("skill")
        return (
            {"tab": "tab-single-skill"},
            no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            class_code, skill_code,
        )

    # --- Example Core -> Tab 1 ---
    if typ == "tab4-core-btn":
        core_blob = comp_id.get("core")
        codes = tab4_split_codes(core_blob, 2)
        if len(codes) != 2:
            return (
                no_update,
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update,
            )

        return (
            {"tab": "tab-2skill"},
            class_code, codes[0], codes[1],
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update,
        )

    # --- Example Build -> Tab 2 ---
    if typ == "tab4-build-btn":
        build_blob = comp_id.get("build")
        codes = tab4_split_codes(build_blob, 4)
        if len(codes) != 4:
            return (
                no_update,
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update,
            )

        return (
            {"tab": "tab-combo-detail"},
            no_update, no_update, no_update,
            class_code, codes[0], codes[1], codes[2], codes[3],
            no_update, no_update,
        )

    return (
        no_update,
        no_update, no_update, no_update,
        no_update, no_update, no_update, no_update, no_update,
        no_update, no_update,
    )
    
@app.callback(
    Output("tab4-sort-mode", "data"),
    Input("tab4-sort-class-tier-btn", "n_clicks"),
    Input("tab4-sort-rating-btn", "n_clicks"),
    Input("tab4-sort-class-alpha-btn", "n_clicks"),
    prevent_initial_call=False,
)
def tab4_set_sort_mode(n_class_tier, n_rating, n_class_alpha):
    ctx = callback_context

    if not ctx.triggered:
        return "class_tier"

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "tab4-sort-rating-btn":
        return "rating_desc"
    if trigger_id == "tab4-sort-class-alpha-btn":
        return "class_alpha"

    return "class_tier"
    

@app.callback(
    Output("tab4-table-wrap", "children"),
    Input("tab4-sort-mode", "data"),
)
def tab4_update_table(sort_mode):
    return tab4_class_summary_table(sort_mode)
    
if __name__ == "__main__":
    app.run(debug=True)
