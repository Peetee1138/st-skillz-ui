# app4.py - v0.4-deploy
#   0.2  - Implement 2-Skill Explorer and Skill Combo Detail locally for G2-G4 (20251207)
#        - initial deployment to GitHub / OnRender
#        - transition to .npz for 
#   0.3  - Implement Tab 3 and enhancements
#   0.4  - Back out efforts to highlight the heatmap on Tab 1

import os
from pathlib import Path

import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import json

from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update, callback_context
from dash.dependencies import ALL

# ---------- CONFIG ----------

DATA_DIR  = Path(".")  # adjust if you keep CSVs elsewhere
CLASS_DIR = DATA_DIR / Path("classes")

HERO_CODES_CSV = DATA_DIR / "db_hero_codes.csv"
SKILL_CODES_CSV = DATA_DIR / "db_skill_codes.csv"
SKILL_SORT_ORDER_CSV = DATA_DIR / "db_skill_sort_order.csv"
SKILL_INCOMPAT_CSV = DATA_DIR / "db_skill_codes_incompat.csv"
DETAIL_ICON_PATH = "/assets/detail_icons"

CLASS_FINAL_DATA_FILES = {
    "G2": CLASS_DIR / "G2_final_data.csv",
    "G3": CLASS_DIR / "G3_final_data.csv",
    "G4": CLASS_DIR / "G4_final_data.csv",
}

CLASS_FINAL_DATA_NPZ_FILES = {
    "G2": CLASS_DIR / "G2_final_data.npz",
    "G3": CLASS_DIR / "G3_final_data.npz",
    "G4": CLASS_DIR / "G4_final_data.npz",
}

CLASS_ALL_DATA_FILES = {
    "G2": CLASS_DIR / "G2_all_data_pass4_unique.csv",
    "G3": CLASS_DIR / "G3_all_data_pass4_unique.csv",
    "G4": CLASS_DIR / "G4_all_data_pass4_unique.csv",
}

# NEW: NPZ versions (same folder, same base name)
CLASS_ALL_DATA_NPZ_FILES = {
    "G2": CLASS_DIR / "G2_all_data_pass4_unique.npz",
    "G3": CLASS_DIR / "G3_all_data_pass4_unique.npz",
    "G4": CLASS_DIR / "G4_all_data_pass4_unique.npz",
}

CLASS_SKILL_ASSESS_FILES = {
    "G2": CLASS_DIR / "G2_data_assess.csv",
    "G3": CLASS_DIR / "G3_data_assess.csv",
    "G4": CLASS_DIR / "G4_data_assess.csv",
}

_single_skill_assess_cache = {}

# ---------- SHARED UI STYLES ----------

LEFT_FRAME_STYLE = {
    "width": "350px",
    "flex": "0 0 350px",
}

LEFT_FRAME_DROPDOWN_STYLE = {
    "width": "250px",
    "flex": "0 0 250px",
}

TITLE_BANNER_STYLE = {
    "textAlign": "center",
    "fontWeight": "bold",
    "backgroundColor": "#444444",
    "color": "white",
    "padding": "6px",
    "borderRadius": "4px",  # can be 2px on charts if you prefer
    "marginBottom": "4px",
}

ROW_CLASS_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "marginTop": "2px",
}

ROW_SKILL_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "marginTop": "2px",
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



# ---------- DATA LOAD HELPERS ----------

def load_hero_codes():
    df = pd.read_csv(HERO_CODES_CSV)
    return df

def load_skill_metadata():
    df = pd.read_csv(SKILL_CODES_CSV)
    print("Loaded skill codes with columns:", list(df.columns))

    # Try to be robust to column naming
    name_col = "full_name" if "full_name" in df.columns else df.columns[0]
    code_col = "short_name" if "short_name" in df.columns else df.columns[1]
    rarity_col = "rarity" if "rarity" in df.columns else None

    name_map = df.set_index(code_col)[name_col].to_dict()

    if rarity_col:
        rarity_map = df.set_index(code_col)[rarity_col].to_dict()
    else:
        rarity_map = {}

    return name_map, rarity_map

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


_all_data_cache = {}

def get_all_data_for_class(class_code: str):
    """
    Lazy loader for *_all_data_pass4_unique, using NPZ (mr_*) ONLY.

    - NPZ layout (from Macro_Ref):
        mr_<colname>  → 1D array for each DataFrame column
        mr_histogram  → 2D histogram array (optional, not needed here)

    - After load, we strip 'mr_' and rebuild:
        class_code, skill_list, and skill_key
    """
    if not class_code:
        return None

    if class_code in _all_data_cache:
        return _all_data_cache[class_code]

    npz_path = CLASS_ALL_DATA_NPZ_FILES.get(class_code)
    if not npz_path or not npz_path.exists():
        print(f"[get_all_data_for_class] NPZ not found for {class_code}: {npz_path}")
        _all_data_cache[class_code] = None
        return None

    try:
        print(f"[get_all_data_for_class] Loading NPZ for {class_code}: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            cols = {}
            for key in data.files:
                # only mr_* columns → strip prefix
                if not key.startswith("mr_"):
                    continue
                if key == "mr_histogram":
                    # optional 2D histogram; the app doesn't need it directly
                    continue
                col_name = key[3:]  # drop 'mr_'
                cols[col_name] = data[key]

        if not cols:
            print(f"[get_all_data_for_class] NPZ had no mr_* data for {class_code}")
            _all_data_cache[class_code] = None
            return None

        df = pd.DataFrame(cols)

    except Exception as e:
        print(f"[get_all_data_for_class] ERROR loading {class_code} NPZ: {e}")
        _all_data_cache[class_code] = None
        return None

    # --- parse skill codes, build skill_list & skill_key (same as before) ---
    skill_col = "skill_code"
    if skill_col not in df.columns:
        # older versions sometimes used 'skill_name'
        skill_col = "skill_name"

    parsed = df[skill_col].apply(parse_skill_codes)
    df["class_code"] = parsed.apply(lambda x: x[0])
    df["skill_list"] = parsed.apply(lambda x: x[1])
    df["skill_key"]  = df["skill_list"].apply(lambda lst: "".join(sorted(lst)))

    _all_data_cache[class_code] = df
    return df


def load_final_data_for_class(code: str) -> pd.DataFrame:
    """
    Load final_data for a class from NPZ ONLY.

    Expected NPZ layout:
        mr_<colname> → 1D array per column

    We strip 'mr_' and return a DataFrame. If anything goes wrong,
    we return an empty DataFrame (class will be skipped in precompute).
    """
    npz_path = CLASS_FINAL_DATA_NPZ_FILES.get(code)
    if not npz_path or not npz_path.exists():
        print(f"[final_data] NPZ not found for {code}: {npz_path}")
        return pd.DataFrame()

    try:
        data = np.load(npz_path, allow_pickle=True)
        cols = {}
        for key in data.files:
            if not key.startswith("mr_"):
                continue
            col = key[3:]  # strip 'mr_'
            cols[col] = data[key]

        if not cols:
            print(f"[final_data] No mr_* arrays in NPZ for {code}: {npz_path}")
            return pd.DataFrame()

        df = pd.DataFrame(cols)

        if "skill_name" not in df.columns:
            print(f"[final_data] WARNING: 'skill_name' missing in {code} final_data.")
        return df

    except Exception as e:
        print(f"[final_data] ERROR loading NPZ for {code}: {e}")
        return pd.DataFrame()


def precompute_class_data():
    class_df = {}
    class_skills = {}

    # keys are still "G2", "G3", "G4"
    for code in CLASS_FINAL_DATA_NPZ_FILES.keys():
        df = load_final_data_for_class(code)
        if df.empty:
            continue

        parsed = df["skill_name"].apply(parse_skill_codes)
        df["class_code"] = parsed.apply(lambda x: x[0])
        df["skill_list"] = parsed.apply(lambda x: x[1])

        skills = sorted({s for skills in df["skill_list"] for s in skills})
        class_skills[code] = skills
        class_df[code] = df

    return class_df, class_skills


hero_codes_df = load_hero_codes()
class_df, class_skills = precompute_class_data()
skill_name_map, skill_rarity_map = load_skill_metadata()
skill_sort_order = load_skill_sort_order()

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
      1) class_skills (from final_data)
      2) skill_sort_order (from db_skill_sort_order.csv)
      3) all known skill codes (from db_skill_codes.csv)
    Always returns a sorted list.
    """
    if not class_code:
        return []

    # 1) Prefer class_skills from final_data
    base = class_skills.get(class_code)
    if base:
        return sorted(set(base))

    # 2) Fallback to sort-order list
    order = skill_sort_order.get(class_code, [])
    if order:
        return sorted(set(order))

    # 3) Last resort: all known skill codes
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

def find_combo_row(df_all: pd.DataFrame, class_code: str, skills: list[str]):
    """
    Find the row in df_all for a given class + 4-skill combo.

    The data uses a canonical skill_key = ''.join(sorted(codes)),
    so we must also sort the selected codes before matching.
    """
    if df_all is None or not class_code or any(s is None for s in skills):
        return None

    # ✅ canonical key: alphabetized 4-skill string
    key = "".join(sorted(skills))

    matches = df_all[
        (df_all["class_code"] == class_code) &
        (df_all["skill_key"] == key)
    ]

    if matches.empty:
        print(f"[find_combo_row] No matching row for {class_code} + {skills} (key={key})")
        return None

    # In theory there should be exactly one; take the first
    return matches.iloc[0]

def detail_icon(filename, class_name="detail-icon", title=None):
    """
    Small helper to render an <img> from assets/detail_icons.
    Used for quality & special icons in the detail view.
    """
    return html.Img(
        src=f"{DETAIL_ICON_PATH}/{filename}",
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


def get_special_icons(row):
    """
    Returns a list of components for special icons, with a leading ' | '
    if any special rules apply.
    """
    specials = []

    # Example: R3 APEX
    if str(row.get("r3_zone", "")).upper() == "APEX":
        specials.append(
            detail_icon("icon_global_upgrade_perk.png",
                        class_name="special-icon",
                        title="APEX Build")
        )

    if not specials:
        return []

    return [html.Span(" | ", className="headline-separator")] + specials

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

        return html.Span(
            [
                html.Button(
                    html.Img(
                        src=icon_src,
                        className="skill-icon",
                        title=full_name,
                    ),
                    style={
                        "border": "1px solid #0645AD",   # thin blue "link" border
                        "padding": "0",
                        "background": "white",
                        "cursor": "pointer",
                    },
                ),
                html.Span(" ", className="skill-label-icon-space"),
                format_skill_name_with_info(full_name),
            ],
            className="headline-skill-chunk",
        )

    # First two skills on line 1, next two on line 2
    line1_children = []
    line2_children = []

    if len(skill_codes) >= 1:
        line1_children.append(make_skill_chunk(skill_codes[0]))
    if len(skill_codes) >= 2:
        line1_children.append(html.Span(" | ", className="skill-separator"))
        line1_children.append(make_skill_chunk(skill_codes[1]))

    if len(skill_codes) >= 3:
        line2_children.append(make_skill_chunk(skill_codes[2]))
    if len(skill_codes) >= 4:
        line2_children.append(html.Span(" | ", className="skill-separator"))
        line2_children.append(make_skill_chunk(skill_codes[3]))

    skills_lines = [html.Div(line1_children, className="skills-line")]
    if line2_children:
        skills_lines.append(html.Div(line2_children, className="skills-line"))

    return html.Div(
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
                    html.Span(": ", className="class-skill-colon"),
                ],
                className="class-label-block",
            ),
            html.Div(skills_lines, className="class-skills-block"),
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
            html.Span(
                f"{rating_pctile_pct:.2f}",
                className="headline-rating-number",
            ),
            html.Span(" ", className="headline-space"),
            html.Span("%ile", className="headline-sub-label"),
        ],
        className="headline-pct-block",
    )

    # Quality + special icons: now driven by percentile, not raw_rating
    quality_icons = get_quality_icons(rating_pctile_raw, net_rating)
    special_icons = get_special_icons(row)

    return html.Div(
        [
            rating_block,
            html.Span(" | ", className="headline-separator"),
            pct_block,
            html.Span(" | ", className="headline-separator"),
            html.Span(
                quality_icons + special_icons,
                className="headline-icons-block",
            ),
        ],
        className="build-headline-row",
    )

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
                    # LEFT: Build Attributes + Rating Components stacked
                    html.Div(
                        [
                            build_build_substats_table(row),
                            html.Br(style={"lineHeight": "6px"}),
                            build_rating_components_table(row),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "8px",
                        },
                    ),
                    # RIGHT: Per-skill table
                    build_single_skill_table(row),
                ],
                className="build-subtables-row",
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "16px",
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
      - c_r1_p3 → Speed (R1)
      - c_r2_p3 → Survival (R2)
      - c_r3_p3 → Efficiency (R3)
    """

    def fmt_score(v):
        if v is None or pd.isna(v):
            return "—"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    r1 = row.get("c_r1_p3", np.nan)
    r2 = row.get("c_r2_p3", np.nan)
    r3 = row.get("c_r3_p3", np.nan)

    rows = [
        {
            "title": "Speed (R1)",
            "line2": "How fast the build wins fights.",
            "line3a": (
                "R1 scores the build by comparing its Rounds-to-Win results "
                "(full histogram + refined R95) against all other builds of its class."
            ),
            "line3b": "Faster, more consistent clears \u2192 higher R1.",
            "val": fmt_score(r1),
        },
        {
            "title": "Survival (R2)",
            "line2": "How safely the build survives to the end of combat.",
            "line3a": (
                "R2 evaluates minimum HP across all heroes (min_h1h2h3) and compares "
                "survival margins across the class."
            ),
            "line3b": "Builds that stay stable under pressure \u2192 higher R2.",
            "val": fmt_score(r2),
        },
        {
            "title": "Efficiency (R3)",
            "line2": "How balanced and optimal the build is for its class.",
            "line3a": (
                "R3 measures how close the build is to the class\u2019s APEX profile, using "
                "distance-from-APEX survival margin, defensive efficiency, and timing efficiency."
            ),
            "line3b": (
                "Builds in the class\u2019s Goldilocks zone\u2014neither too fragile "
                "nor too slow\u2014score highest."
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
        "fontWeight": "600",
        "fontSize": "13px",
    }

    header = html.Tr(
        [
            html.Th("Rating Component", style=header_cell_style),
            html.Th("Rating (max 1.0)",           style=header_cell_style),
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
                    html.Td(label_children, style=label_cell_style),
                    html.Td(r["val"],       style=value_cell_style),
                ]
            )
        )

    return html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        className="build-rating-components-table",
        style={
            "borderCollapse": "collapse",
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
        "textAlign": "center",  # center the skill icons
    }

    rating_cell_style = {
        "textAlign": "right",
        "padding": "2px 6px",
        "whiteSpace": "nowrap",
    }

    # Header row (with new sparkline title)
    header = html.Tr(
        [
            html.Th("", style=header_cell_style),
            html.Th("Skill Name", style=header_cell_style),
            html.Th("Rating & Tier", style=header_cell_style),
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
        )

        skill_label_str = skill_label(sc)  # "ABC - Name"

        # Default values
        r_max = None
        tier = None
        nq_pct = sub80_pct = pct80_95 = pct95 = 0.0

        srow = assess_index.get(sc)
        if srow is not None:
            r_max = srow.get("r_max", None)
            tier = srow.get("skill_tier", None)
            nq_pct = srow.get("nq_pct", 0.0)
            sub80_pct = srow.get("sub80_pct", 0.0)
            pct80_95 = srow.get("80_95_pct", 0.0)
            pct95 = srow.get("95ile_pct", 0.0)

        # Rating + tier icon (rating text 20% larger)
        rating_children = []
        if r_max is not None and not pd.isna(r_max):
            rating_children.append(
                html.Span(
                    f"{float(r_max):.1f} ",
                    className="single-skill-rating",
                    style={"fontSize": "1.6em"},  
                )
            )
        else:
            rating_children.append(
                html.Span(
                    "—",
                    className="single-skill-rating",
                    style={"fontSize": "1.2em"},
                )
            )

        tier_icon = single_skill_tier_icon(tier)
        if tier_icon is not None:
            rating_children.append(tier_icon)

        spark = single_skill_sparkline(nq_pct, sub80_pct, pct80_95, pct95)

        body_rows.append(
            html.Tr(
                [
                    html.Td(icon_cell, style=icon_cell_style),
                    html.Td(
                        skill_label_str,
                        style={**body_cell_style, "fontSize": "16px"},  # roughly 2× base
                    ),
                    html.Td(rating_children, style=rating_cell_style),
                    html.Td(spark, style=body_cell_style),
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
      4 -> icon_shop_face_C
      6 -> icon_shop_face_D
    """
    mapping = {
        1: "icon_shop_face_SSS.png",
        2: "icon_shop_face_S.png",
        3: "icon_shop_face_A.png",
        4: "icon_shop_face_C.png",
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
                    html.Span(": ", className="class-skill-colon"),
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
                    ),
                    html.Span(" ", className="skill-label-icon-space"),
                    format_skill_name_with_info(full_name_raw),
                ],
                className="single-skill-class-skill-block",
            ),
        ],
        className="single-skill-class-line",
    )

    # Roland tag info
    roland_val = row.get("roland_tag", 0) or 0

    if roland_val == 0:
        roland_text = "No"
        roland_color = "#c0392b"   # red
        roland_sub  = "Roland Priority"
    elif roland_val == 5:
        roland_text = "Maybe"
        roland_color = "#f1c40f"   # yellow
        roland_sub  = "Roland Priority"
    else:
        roland_text = "Yes"
        roland_color = "#27ae60"   # green
        roland_sub  = f"{roland_val}ᵗʰ Priority"

    roland_block = html.Span(
        [
            html.Img(
                src="/assets/detail_icons/veteran_head.png",
                className="roland-icon",
                style={
                    "width": "28px",
                    "height": "28px",
                    "verticalAlign": "middle",
                    "marginRight": "6px",
                },
            ),
            html.Span(
                roland_text,
                className="headline-rating-number",
                style={"color": roland_color},
            ),
            html.Span(" ", className="headline-space"),
            html.Span(
                roland_sub,
                className="headline-sub-label",
            ),
        ],
        className="headline-pct-block roland-block",
        style={"display": "inline-flex", "alignItems": "center"},
    )

    roland_block = html.Span(
        [
            html.Img(
                src="/assets/detail_icons/veteran_head.png",
                className="roland-icon",
                style={
                    "width": "28px",
                    "height": "28px",
                    "verticalAlign": "middle",
                    "marginRight": "6px",
                },
            ),
            html.Span(
                roland_text,
                className="headline-rating-number",
                style={"color": roland_color},
            ),
            html.Span(" ", className="headline-space"),
            html.Span(
                roland_sub,
                className="headline-sub-label",
            ),
        ],
        className="headline-pct-block roland-block",
        style={"display": "inline-flex", "alignItems": "center"},
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
            html.Span(" | ", className="headline-separator"),
            html.Span(
                [
                    html.Span(f"{r90_raw:.1f}%", className="headline-rating-number"),
                    html.Span(" ", className="headline-space"),
                    html.Span("Good Builds (>90 Rating)", className="headline-sub-label"),
                ],
                className="headline-pct-block",
            ),
            html.Span(" | ", className="headline-separator"),
            html.Span(
                [
                    html.Span(f"{r95builds_raw:.1f}%", className="headline-rating-number"),
                    html.Span(" ", className="headline-space"),
                    html.Span("Builds in 95+ %ile for Class", className="headline-sub-label"),
                ],
                className="headline-pct-block",
            ),
            html.Span(" | ", className="headline-separator"),
            tier_icon if tier_icon is not None else html.Span(),
            html.Span(" | ", className="headline-separator"),
            roland_block,
        ],
        className="single-skill-headline-row",
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
                    html.Div(bar_block, style={"flex": "3 1 0"}),
                    html.Div(
                        stats_table,
                        style={
                            "flex": "2 1 0",
                            "paddingLeft": "16px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "12px",
                },
            ),
        ],
        className="single-skill-summary-block",
    )




# ---------- DASH APP ----------

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Skills UI — 2-Skill PoC"

def _is_available(val):
    return str(val).strip().lower() in ("true", "yes", "1", "y")

def make_layout_tab1():
    return html.Div(
        style={"margin": "10px 40px 10px 10px", "fontFamily": "Arial"},
        children=[
            html.H2("Skills UI — 2-Skill Explorer (Proof of Concept)"),

            # -------------------------------------------------------
            # ROW 1: Frame 1 (left) + Frame 2 (right)
            # -------------------------------------------------------
            html.Div(
                style={
                    "display": "flex",
                    "flexWrap": "nowrap",
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
                                    {"label": "All",          "value": "all"},
                                    {"label": "Top 10",       "value": "top10"},
                                    {"label": "Epic & Rare",  "value": "epic_rare"},
                                ],
                                value="all",
                                clearable=False,
                                style={"width": "180px", "marginTop": "4px"},
                                persistence=True,
                                persistence_type="session",
                            ),
                        ],
                    ),

                    # ---------- FRAME 2 (Right Column) ----------
                    html.Div(
                        style={"flex": "1", "minWidth": "450px"},
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
                                hidden_columns=["_s3_code", "_s4_code"],
                                columns=[
                                    {"name": "Rank", "id": "rank"},
                                    {"name": "Skill 3", "id": "s3_full"},
                                    {"name": "Skill 4", "id": "s4_full"},
                                    {"name": "Raw Rating", "id": "raw_rating"},
                                    {"name": "Net Rating", "id": "net_rating"},
                                    {"name": "_s3_code", "id": "_s3_code"},
                                    {"name": "_s4_code", "id": "_s4_code"},
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
        id="combo-detail-container",
        style={"padding": "10px", "fontFamily": "Arial"},
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
                            ],
                        ],
                    ),

                    # ---------- FRAME 2 (Right Column) ----------
                    html.Div(
                        style={"flex": "1", "minWidth": "400px"},
                        children=[
                            html.H4(
                                "Skill Combo Detail",
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
                "Rounds-to-Win Distribution",
                style={**TITLE_BANNER_STYLE, "borderRadius": "2px"},
            ),
            html.Div(
                [
                    html.Span("Tips: ", style={"fontWeight": "bold"}),
                    html.Span("(1) Solid line = this build, dashed line = APEX build. "),
                    html.Span("(2) Faint dotted lines = per-skill average across all builds. "),
                    html.Span("(3) Vertical lines show r95 for each build."),
                    html.Span("(4) Click a trace in the legend to hide/show."),
                ],
                style={"fontSize": "11px", "marginTop": "4px"},
            ),
            dcc.Graph(
                id="combo-detail-histogram",
                figure=go.Figure(),
                config={"displayModeBar": False},
                style={"height": "400px", "width": "75%"},
            ),
        ],
    )

# ========== Tab 3: Single Skill Info layout ==========

def make_layout_tab3():
    return html.Div(
        id="single-skill-tab",
        style={"padding": "10px", "fontFamily": "Arial"},
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
                "Class Skill Ranking",
                style={**TITLE_BANNER_STYLE, "borderRadius": "2px"},
            ),

            dash_table.DataTable(
                id="single-skill-ranking-table",
                style_table={
                    "maxHeight": "400px",
                    "overflowY": "auto",
                    "overflowX": "auto",
                    "width": "100%",
                },
                style_cell={
                    "fontSize": 12,
                    "textAlign": "left",
                    "color": "black",
                },
                style_header={
                    "fontFamily": "Arial",
                    "fontSize": 13,
                    "fontWeight": "bold",
                    "textAlign": "center",
                    "backgroundColor": "#2f4f4f",  # dark-ish green/teal
                    "color": "white",
                },
                columns=[
                    {"name": "Rank", "id": "skill_rank"},
                    {"name": "Skill", "id": "skill_label"},
                    {"name": "Tier", "id": "skill_tier"},
                    {"name": "Max Rating", "id": "r_max"},
                    {"name": "% N/Q", "id": "nq_pct"},
                    {"name": "<80 %ile", "id": "sub80_pct"},
                    {"name": "80–95 %ile", "id": "pct_80_95"},
                    {"name": "95+ %ile", "id": "pct_95"},
                ],
                data=[],
            ),
        ],
    )

# ========== Overall App Layout ==========

app.layout = html.Div(
    children=[
        dcc.Store(id="selected-combo"),
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
        return [], [], [], []

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
    Output("step3-title", "children"),
    Output("heatmap-title", "children"),
    Output("selection-summary", "children"),
    Output("combo-table", "data", allow_duplicate=True),
    Output("combo-heatmap", "figure", allow_duplicate=True),
    Input("hero-class", "value"),
    Input("skill1", "value"),
    Input("skill2", "value"),
    Input("heatmap-skill-filter", "value"),  # NEW
    prevent_initial_call="initial_duplicate",
)
def update_outputs(class_code, skill1, skill2, skill_filter):
    empty_fig = go.Figure()

    # Helper for dynamic titles (yellow skill names in bar, not in table)
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

    # Guard: need a class and two different skills
    if not class_code or not skill1 or not skill2 or skill1 == skill2:
        step3_title = "3rd & 4th Skill Options"
        heatmap_title = "Heat Map of 3rd/4th Skill Pairs"
        msg = "Pick a hero class plus two different core skills."
        return step3_title, heatmap_title, msg, [], empty_fig

    df = class_df.get(class_code)
    if df is None or df.empty:
        step3_title, heatmap_title = build_titles()
        msg = f"No data available for class {class_code}."
        return step3_title, heatmap_title, msg, [], empty_fig

    # Filter rows that contain both core skills
    mask = df["skill_list"].apply(
        lambda skills: (skill1 in skills) and (skill2 in skills)
    )
    sub = df[mask].copy()          # <-- sub is always defined before we touch it

    step3_title, heatmap_title = build_titles()

    if sub.empty:
        msg = (
            f"No builds found containing skills {skill1} and {skill2} "
            f"for class {class_code}."
        )
        return step3_title, heatmap_title, msg, [], empty_fig

    # ----- Build S3/S4 table -----
    s3_list = []
    s4_list = []
    for skills in sub["skill_list"]:
        others = [s for s in skills if s not in (skill1, skill2)]
        if len(others) != 2:
            continue
        a, b = sorted(others)
        s3_list.append(a)
        s4_list.append(b)

    sub = sub.loc[sub.index[: len(s3_list)]].copy()
    sub["s3"] = s3_list
    sub["s4"] = s4_list

    table_df = sub[["s3", "s4", "raw_rating", "net_rating"]].copy()
    table_df = table_df.sort_values("raw_rating", ascending=False)

    table_df["rank"] = range(1, len(table_df) + 1)
    table_df["s3_full"] = table_df["s3"].map(get_full_skill_name).fillna(table_df["s3"])
    table_df["s4_full"] = table_df["s4"].map(get_full_skill_name).fillna(table_df["s4"])

    table_df_ui = table_df[["rank", "s3_full", "s4_full", "raw_rating", "net_rating"]].copy()
    table_df_ui["_s3_code"] = table_df["s3"].values
    table_df_ui["_s4_code"] = table_df["s4"].values

    # ----- Heat map prep -----
    lookup = {}
    for _, row in table_df.iterrows():
        key = frozenset({row["s3"], row["s4"]})
        lookup[key] = row["net_rating"]

    skills_present = (
        {row["s3"] for _, row in table_df.iterrows()}
        | {row["s4"] for _, row in table_df.iterrows()}
    )

    order = skill_sort_order.get(class_code, [])
    axis_skills = [s for s in order if s in skills_present]
    axis_skills += sorted(skills_present - set(axis_skills))

    # ---------------------------------------------------------
    # NEW: Apply dynamic filter to axis_skills (Top 10 / Epic+Rare)
    # ---------------------------------------------------------
    filter_mode = skill_filter or "all"

    # Helper: build per-skill "best net_rating" for skills in axis_skills
    if filter_mode == "top10":
        best_net = {}
        for _, row in table_df.iterrows():
            s3 = row["s3"]
            s4 = row["s4"]
            val = row["net_rating"]
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue

            # Only consider skills that are actually on this axis
            for s in (s3, s4):
                if s not in axis_skills:
                    continue
                prev = best_net.get(s)
                if prev is None or val_f > prev:
                    best_net[s] = val_f

        if best_net:
            # Sort the existing axis_skills by their best net rating (desc),
            # keep only the top 10 in the same general order.
            axis_skills_sorted = sorted(
                axis_skills,
                key=lambda s: best_net.get(s, float("-inf")),
                reverse=True,
            )
            axis_skills = axis_skills_sorted[:10]

    elif filter_mode == "epic_rare":
        try:
            df_codes = load_hero_codes()
            # We assume columns: 'sk_name' for code, 'rarity' with values like 'Epic', 'Rare', etc.
            mask_er = (
                df_codes["rarity"].isin(["Epic", "Rare"])
                & df_codes["sk_name"].isin(axis_skills)
            )
            er_skills = df_codes.loc[mask_er, "sk_name"].tolist()
            # Keep original axis order, just filter down
            filtered_axis = [s for s in axis_skills if s in er_skills]
            # Only apply if we actually got some matches, otherwise fall back to All
            if filtered_axis:
                axis_skills = filtered_axis
        except Exception:
            # Fail-safe: if anything goes wrong (missing cols, etc.), fall back to All
            pass

    # If after filtering we have < 2 skills, don't try to build a weird 1x1 heatmap
    if len(axis_skills) < 2:
        msg = (
            f"Filter '{filter_mode}' left fewer than 2 skills on the heat map. "
            "Try switching back to 'All'."
        )
        empty_fig = go.Figure()
        return step3_title, heatmap_title, msg, table_df_ui.to_dict("records"), empty_fig

    # After axis_skills is finalized (and after filtering)
    n_skills = max(1, len(axis_skills))

    # Dynamic tick font: smaller when crowded, but not microscopic
    tick_font_size = 12
    if n_skills > 12:
        tick_font_size = max(8, int(180 / n_skills))

    # Dynamic icon size
    # Fewer skills → bigger icons; more skills → smaller
    icon_size_y = 0.12
    icon_size_x = 0.80
    icon_y = 1.06

    if n_skills > 10:
        icon_size_y = 0.09
        icon_size_x = 0.70
        icon_y = 1.05
    if n_skills > 16:
        icon_size_y = 0.07
        icon_size_x = 0.60
        icon_y = 1.04

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
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    text[i][j] = "n/q"
                else:
                    text[i][j] = f"{num:.1f}"
                    num_clamped = max(50.0, min(100.0, num))
                    z_numeric[i, j] = num_clamped

    # ----- Heat map layers: gray base + numeric overlay -----
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
    # Tag base trace so we don't accidentally strip it
    heat_base.meta = "base-heatmap"

    colorscale = [
        [0.0, "rgb(253,141,60)"],   # orange
        [0.5, "rgb(255,255,178)"],  # yellow
        [1.0, "rgb(35,132,67)"],    # green
    ]

    heat_num = go.Heatmap(
        z=z_numeric,
        x=axis_skills,
        y=axis_skills,
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmin=50,
        zmax=100,
        hovertemplate="S3=%{y}<br>S4=%{x}<br>Rating=%{text}<extra></extra>",
        colorbar=dict(
            title="Rating",
            tickvals=[50, 75, 100],
            ticktext=["50", "75", "100"],
            thickness=10,      # thinner bar
            len=0.4,           # 40% of plot width
            orientation="h",   # horizontal
            x=0.0,
            xanchor="left",
            y=-0.01,
            yanchor="top",
        ),
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
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=120, b=20),  # slightly larger top margin
        clickmode="event+select",
    )

    # ----- Simple icon Row -----
    images = []
    for s in axis_skills:
        images.append(
            dict(
                source=f"/assets/skill_icons/{s}.png",
                xref="x",
                yref="paper",
                x=s,
                y=icon_y,
                sizex=icon_size_x,
                sizey=icon_size_y,
                xanchor="center",
                yanchor="bottom",
                layer="above",
            )
        )

    fig.update_layout(images=images)

    summary = (
        f"Class {class_code}, core skills: {skill1} + {skill2}.  "
        f"Found {len(table_df_ui)} unique 3rd/4th combos."
    )

    return step3_title, heatmap_title, summary, table_df_ui.to_dict("records"), fig



@app.callback(
    Output("detail-hero-class", "value"),
    Output("detail-skill1", "value"),
    Output("detail-skill2", "value"),
    Output("detail-skill3", "value"),
    Output("detail-skill4", "value"),
    Input("main-tabs", "value"),        # tab changes
    Input("selected-combo", "data"),    # heatmap click → store
    Input("combo-table", "active_cell"),# rank click
    State("combo-table", "data"),
    State("hero-class", "value"),
    State("skill1", "value"),
    State("skill2", "value"),
    prevent_initial_call=True,
)
def drive_detail_selection(
    active_tab,
    sel_combo,
    active_cell,
    rows,
    class_code,
    s1,
    s2,
):
    """
    One callback to control the 4-skill selection on the detail tab.

    Priority logic:
      - If the trigger was a Rank click in the table, use that row.
      - Else, if the trigger was a heatmap click (via selected-combo), use that.
      - Only apply when 'Skill Combo Detail' tab is active.
    """
    from dash import callback_context

    # Only change detail selectors when we're on the detail tab
    if active_tab != "tab-combo-detail":
        return no_update, no_update, no_update, no_update, no_update

    ctx = callback_context
    trigger_id = None
    if ctx.triggered:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # --- 1️⃣ Rank click in the table (explicit user selection) ---
    if trigger_id == "combo-table" and active_cell:
        if active_cell.get("column_id") != "rank":
            # Clicked some other column → ignore for detail selection
            return no_update, no_update, no_update, no_update, no_update

        # Defensive checks
        if not rows or class_code is None or s1 is None or s2 is None:
            return no_update, no_update, no_update, no_update, no_update

        row_idx = active_cell.get("row")
        if row_idx is None or row_idx < 0 or row_idx >= len(rows):
            return no_update, no_update, no_update, no_update, no_update

        row = rows[row_idx]
        s3 = row.get("_s3_code")
        s4 = row.get("_s4_code")
        if not s3 or not s4:
            return no_update, no_update, no_update, no_update, no_update

        # Use current class + core from tab 1, extras from table row
        return class_code, s1, s2, s3, s4

    # --- 2️⃣ Heatmap click → selected-combo store ---
    # Trigger can be either 'selected-combo' (store updated) or 'main-tabs'
    # (user switched to detail tab after a store update).
    if sel_combo and trigger_id in ("selected-combo", "main-tabs"):
        class_code2 = sel_combo.get("class_code")
        core = sel_combo.get("core", [])
        extra = sel_combo.get("extra", [])

        s1_2 = core[0] if len(core) > 0 else None
        s2_2 = core[1] if len(core) > 1 else None
        s3_2 = extra[0] if len(extra) > 0 else None
        s4_2 = extra[1] if len(extra) > 1 else None

        # Require all four to be present; otherwise do nothing
        if not (class_code2 and s1_2 and s2_2 and s3_2 and s4_2):
            return no_update, no_update, no_update, no_update, no_update

        return class_code2, s1_2, s2_2, s3_2, s4_2

    # --- 3️⃣ Nothing meaningful to do ---
    return no_update, no_update, no_update, no_update, no_update



# @app.callback(
#     Output("detail-hero-class", "value"),
#     Output("detail-skill1", "value"),
#     Output("detail-skill2", "value"),
#     Output("detail-skill3", "value"),
#     Output("detail-skill4", "value"),
#     Input("selected-combo", "data"),        # from heatmap click
#     Input("combo-table", "active_cell"),    # from Rank click
#     State("combo-table", "data"),
#     State("hero-class", "value"),
#     State("skill1", "value"),
#     State("skill2", "value"),
#     prevent_initial_call=True,
# )
# def update_detail_selection(sel_combo, active_cell, rows, class_code, s1, s2):
#     """
#     Unifies:
#       - Heatmap click → selected-combo → detail dropdowns
#       - Rank click in S3/S4 table → detail dropdowns

#     Priority: if the user clicked on Rank, use that; otherwise, if we have
#     a selected-combo from the heatmap, use that.
#     """
#     # 1️⃣ Rank click in the table (Section 3)
#     if active_cell and active_cell.get("column_id") == "rank":
#         if not rows or class_code is None or s1 is None or s2 is None:
#             return no_update, no_update, no_update, no_update, no_update

#         row_idx = active_cell.get("row")
#         if row_idx is None or row_idx < 0 or row_idx >= len(rows):
#             return no_update, no_update, no_update, no_update, no_update

#         row = rows[row_idx]
#         s3 = row.get("_s3_code")
#         s4 = row.get("_s4_code")

#         if not s3 or not s4:
#             return no_update, no_update, no_update, no_update, no_update

#         print("Clicked combo (rank):", class_code, s1, s2, s3, s4)
#         return class_code, s1, s2, s3, s4

#     # 2️⃣ Heatmap click → selected-combo
#     if sel_combo:
#         class_code2 = sel_combo.get("class_code")
#         core = sel_combo.get("core", [])
#         extra = sel_combo.get("extra", [])

#         s1_2 = core[0] if len(core) > 0 else None
#         s2_2 = core[1] if len(core) > 1 else None
#         s3_2 = extra[0] if len(extra) > 0 else None
#         s4_2 = extra[1] if len(extra) > 1 else None

#         if not class_code2 or not s1_2 or not s2_2 or not s3_2 or not s4_2:
#             return no_update, no_update, no_update, no_update, no_update

#         print("Clicked combo (heatmap):", class_code2, s1_2, s2_2, s3_2, s4_2)
#         return class_code2, s1_2, s2_2, s3_2, s4_2

#     # 3️⃣ Nothing meaningful to do
#     return no_update, no_update, no_update, no_update, no_update

# @app.callback(
#     Output("detail-hero-class", "value"),
#     Output("detail-skill1", "value"),
#     Output("detail-skill2", "value"),
#     Output("detail-skill3", "value"),
#     Output("detail-skill4", "value"),
#     Input("combo-table", "active_cell"),
#     State("combo-table", "data"),
#     State("hero-class", "value"),
#     State("skill1", "value"),
#     State("skill2", "value"),
#     prevent_initial_call=True,
# )
# def on_rank_click(active_cell, rows, class_code, s1, s2):
#     """
#     When the user clicks a cell in the 'Rank' column of the S3/S4 table,
#     populate the detail tab's hero + 4-skill selectors with that combo.
#     """
#     # Nothing selected yet
#     if not active_cell or active_cell.get("column_id") != "rank":
#         return no_update, no_update, no_update, no_update, no_update

#     # Defensive checks
#     if not rows or class_code is None or s1 is None or s2 is None:
#         return no_update, no_update, no_update, no_update, no_update

#     row_idx = active_cell.get("row")
#     if row_idx is None or row_idx < 0 or row_idx >= len(rows):
#         return no_update, no_update, no_update, no_update, no_update

#     row = rows[row_idx]
#     s3 = row.get("_s3_code")
#     s4 = row.get("_s4_code")

#     if not s3 or not s4:
#         return no_update, no_update, no_update, no_update, no_update

#     print("Clicked combo (rank):", class_code, s1, s2, s3, s4)

#     # Set detail tab hero + 4 skills directly
#     return class_code, s1, s2, s3, s4


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
    Output("main-tabs", "value"),
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
    State("combo-table", "data"),             # table rows for s3/s4 click
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
    rows,
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
        return current_tab, no_update, no_update

    trigger_info = ctx.triggered[0]
    trigger_prop = trigger_info["prop_id"]
    trigger_val  = trigger_info["value"]
    trigger_id_str, _ = trigger_prop.split(".", 1)

    # Helper: ignore "fake" clicks where n_clicks is 0
    def is_real_click():
        try:
            return trigger_val is not None and int(trigger_val) > 0
        except Exception:
            return False

    # ---------- 0) Tab 2 Frame 1 skill icons → Single Skill Info ----------
    if trigger_id_str in (
        "detail-skill1-btn",
        "detail-skill2-btn",
        "detail-skill3-btn",
        "detail-skill4-btn",
    ):
        if not is_real_click():
            return current_tab, no_update, no_update

        if not detail_class:
            return current_tab, no_update, no_update

        btn_to_skill = {
            "detail-skill1-btn": d_s1,
            "detail-skill2-btn": d_s2,
            "detail-skill3-btn": d_s3,
            "detail-skill4-btn": d_s4,
        }
        skill_code = btn_to_skill.get(trigger_id_str)

        if not skill_code:
            return current_tab, no_update, no_update

        return "tab-single-skill", detail_class, skill_code

    # ---------- 1) Tab 1 icons → Single Skill Info ----------
    if trigger_id_str == "skill1-icon-btn":
        if not is_real_click():
            return current_tab, no_update, no_update
        if not hero_class or not skill1_code:
            return current_tab, no_update, no_update
        return "tab-single-skill", hero_class, skill1_code

    if trigger_id_str == "skill2-icon-btn":
        if not is_real_click():
            return current_tab, no_update, no_update
        if not hero_class or not skill2_code:
            return current_tab, no_update, no_update
        return "tab-single-skill", hero_class, skill2_code

    # ---------- 2) Clicks in S3/S4 table ----------
    if trigger_id_str == "combo-table":
        # table events aren't n_clicks, so don't check is_real_click here
        if not active_cell or not active_cell.get("column_id"):
            return current_tab, no_update, no_update

        col_id = active_cell.get("column_id")
        row_idx = active_cell.get("row")

        if rows is None or row_idx is None or row_idx < 0 or row_idx >= len(rows):
            return current_tab, no_update, no_update

        row = rows[row_idx]

        # 2a) Rank click → go to Combo Detail tab
        if col_id == "rank":
            return "tab-combo-detail", no_update, no_update

        # 2b) Skill 3 / Skill 4 click → go to Single Skill Info
        if col_id == "s3_full":
            skill_code = row.get("_s3_code")
        elif col_id == "s4_full":
            skill_code = row.get("_s4_code")
        else:
            return current_tab, no_update, no_update

        if not hero_class or not skill_code:
            return current_tab, no_update, no_update

        return "tab-single-skill", hero_class, skill_code

    # ---------- 3) Heatmap click → Tab 2 ----------
    if trigger_id_str == "selected-combo":
        if sel_combo:
            return "tab-combo-detail", no_update, no_update
        return current_tab, no_update, no_update

    # ---------- 4) Tab 2/3 headline + table icons (pattern IDs) ----------
    try:
        comp_id = json.loads(trigger_id_str)
    except json.JSONDecodeError:
        return current_tab, no_update, no_update

    if (
        isinstance(comp_id, dict)
        and comp_id.get("type") == "detail-skill-icon-btn"
    ):
        if not is_real_click():
            return current_tab, no_update, no_update

        skill_code = comp_id.get("skill")
        if not detail_class or not skill_code:
            return current_tab, no_update, no_update

        return "tab-single-skill", detail_class, skill_code

    # ---------- 5) Fallback ----------
    return current_tab, no_update, no_update


# @app.callback(
#     Output("main-tabs", "value"),
#     Input("selected-combo", "data"),     # heatmap click → store
#     Input("combo-table", "active_cell"), # rank click in table
#     State("main-tabs", "value"),         # current tab
#     prevent_initial_call=True,
# )
# def switch_tabs(sel_combo, active_cell, current_tab):
#     """
#     Switch to the Skill Combo Detail tab when:
#       - user clicks a heatmap cell (selected-combo updated), OR
#       - user clicks a Rank cell in the S3/S4 combo table.

#     Otherwise, leave the current tab alone.
#     """
#     ctx = callback_context
#     if not ctx.triggered:
#         # Nothing meaningful triggered this callback
#         return no_update

#     trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

#     # 1️⃣ Rank click in the S3/S4 table
#     if trigger_id == "combo-table" and active_cell:
#         if active_cell.get("column_id") == "rank":
#             return "tab-combo-detail"
#         # clicked some other column → don’t change tabs
#         return current_tab

#     # 2️⃣ Heatmap click → selected-combo updated
#     if trigger_id == "selected-combo" and sel_combo:
#         return "tab-combo-detail"

#     # 3️⃣ Fallback: keep whatever tab we’re already on
#     return current_tab

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
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def update_detail_skill_options(class_code, s1, s2, s3, s4):
    """
    Apply incompatibilities to the 4-skill selectors on the detail tab.

    Each dropdown:
      - draws from the class skill pool
      - excludes any skills already chosen in the *other* boxes
      - excludes skills incompatible with any of the *other* chosen skills
      - always returns alphabetically sorted options
    """
    if not class_code:
        return [], [], [], []

    base_skills = get_base_skills_for_class(class_code)
    if not base_skills:
        return [], [], None, None

    selected = [s1, s2, s3, s4]
    options_out = []

    for idx in range(4):
        # all other selected skills are “fixed” when building this pool
        fixed_others = [s for i, s in enumerate(selected) if i != idx and s]

        # start from base skills and apply incompatibilities
        pool = filtered_skill_pool(base_skills, fixed_others)

        # don’t show skills that are already chosen in other boxes
        used_elsewhere = set(fixed_others)
        pool = [s for s in pool if s not in used_elsewhere]

        # always sort alphabetically
        pool = sorted(pool)

        # use the shared label helper if you added it; otherwise inline
        def make_label(code: str) -> str:
            full = strip_parens(get_full_skill_name(code))
            return f"{code} - {full}"

        opts = [{"label": make_label(s), "value": s} for s in pool]
        options_out.append(opts)

    return tuple(options_out)


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
    Output("combo-detail-text", "children"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def show_combo_detail(class_code, s1, s2, s3, s4):
    """
    Frame 2 (Tab 2): Build-level summary for the selected 4-skill combo.

    Uses XX_all_data_pass4_unique via get_all_data_for_class, then feeds
    the row into build_tab2_frame2_section1 to render:
      - Class + skills line
      - Headline (rating, percentile, icons)
    """
    # Basic guard: need a full 4-skill selection
    if not class_code or not s1 or not s2 or not s3 or not s4:
        return "Select a hero and four skills to see details."

    df_all = get_all_data_for_class(class_code)
    if df_all is None or df_all.empty:
        return f"No all-data found for class {class_code}."

    # Build canonical skill_code = class_code + alphabetized 4-skill string,
    # same pattern as used in update_detail_histogram.
    skill_label_str = canonical_skill_string(s1, s2, s3, s4)
    skill_code = f"{class_code}{skill_label_str}"

    try:
        row = df_all.loc[df_all["skill_code"] == skill_code].iloc[0]
    except (KeyError, IndexError):
        return f"No matching row in all-data for skill_code={skill_code}"

    # At this point, row has:
    #   - class_code
    #   - skill_list (4 codes, in entry order)
    #   - raw_rating, net_rating, rating_pctile, r3_zone, etc.
    return build_tab2_frame2_section1(row, class_meta, skill_lookup)

# ---------- DETAIL HISTOGRAM CALLBACK ----------


@app.callback(
    Output("combo-detail-histogram", "figure"),
    Input("detail-hero-class", "value"),
    Input("detail-skill1", "value"),
    Input("detail-skill2", "value"),
    Input("detail-skill3", "value"),
    Input("detail-skill4", "value"),
)
def update_detail_histogram(class_code, s1, s2, s3, s4):
    """
    Show rounds-to-win histogram for the selected 4-skill combo vs the APEX build.

    - Uses CLASS_ALL_DATA_FILES[{class_code}] (e.g. G2_all_data_pass4_unique.csv)
    - Identifies target row by skill_code = class_code + s1 + s2 + s3 + s4
    - Identifies APEX row by r3_zone == 'APEX'
    - Uses rds_hist_* columns mapped to x-axis centers:
        rds_hist_1_3     -> 2
        rds_hist_2..20   -> 2..20
        rds_hist_21_25   -> 23
        rds_hist_25_plus -> 26
    - Draws two lines (target vs APEX) with vertical lines at each r95
    """

    fig = go.Figure()

    # --- Basic guards ---
    if not class_code or not s1 or not s2 or not s3 or not s4:
        fig.update_layout(
            title="Select a hero and four skills to see the rounds histogram",
            xaxis_title="Rounds to Win",
            yaxis_title="Count of Wins",
        )
        return fig

    df_all = get_all_data_for_class(class_code)
    if df_all is None or df_all.empty:
        fig.update_layout(
            title=f"No all-data histogram found for class {class_code}.",
            xaxis_title="Rounds to Win",
            yaxis_title="Count of Wins",
        )
        return fig

    # --- Identify target + APEX rows ---

    # skill_code is the same encoding as skill_name in final_data:
    #   e.g. G2AcrAllDesWhi
    # Canonical 4-skill label (alphabetized)
    skill_label = canonical_skill_string(s1, s2, s3, s4)
    skill_code = f"{class_code}{skill_label}"

    try:
        target_row = df_all.loc[df_all["skill_code"] == skill_code].iloc[0]
    except (KeyError, IndexError):
        fig.update_layout(
            title=f"No matching row in all-data for skill_code={skill_code}",
            xaxis_title="Rounds to Win",
            yaxis_title="Count of Wins",
        )
        return fig

    if "r3_zone" not in df_all.columns:
        fig.update_layout(
            title="APEX row not available (missing r3_zone column).",
            xaxis_title="Rounds to Win",
            yaxis_title="Count of Wins",
        )
        return fig

    apex_df = df_all.loc[df_all["r3_zone"] == "APEX"]
    if apex_df.empty:
        fig.update_layout(
            title="No APEX row found for this class.",
            xaxis_title="Rounds to Win",
            yaxis_title="Count of Wins",
        )
        return fig

    apex_row = apex_df.iloc[0]

    # --- Histogram columns & bin centers ---

    # Explicit list in desired order
    hist_cols_ordered = [
        "rds_hist_1_3",
        "rds_hist_2",
        "rds_hist_3",
        "rds_hist_4",
        "rds_hist_5",
        "rds_hist_6",
        "rds_hist_7",
        "rds_hist_8",
        "rds_hist_9",
        "rds_hist_10",
        "rds_hist_11",
        "rds_hist_12",
        "rds_hist_13",
        "rds_hist_14",
        "rds_hist_15",
        "rds_hist_16",
        "rds_hist_17",
        "rds_hist_18",
        "rds_hist_19",
        "rds_hist_20",
        "rds_hist_21_25",
        "rds_hist_25_plus",
    ]

    # Keep only those that actually exist in the dataframe
    hist_cols = [c for c in hist_cols_ordered if c in df_all.columns]
    if not hist_cols:
        fig.update_layout(
            title="No rds_hist_* columns found in all-data.",
            xaxis_title="Rounds to Win",
            yaxis_title="Count of Wins",
        )
        return fig

    # Bin centers based on your spec
    bin_centers = {}
    for c in hist_cols:
        if c == "rds_hist_1_3":
            bin_centers[c] = 2.0
        elif c == "rds_hist_21_25":
            bin_centers[c] = 23.0
        elif c == "rds_hist_25_plus":
            bin_centers[c] = 26.0
        else:
            # Expect simple 'rds_hist_N'
            m = re.search(r"rds_hist_(\d+)$", c)
            if m:
                bin_centers[c] = float(m.group(1))
            else:
                # Fallback: ignore any weird column name
                continue

    # Remove any columns we couldn't map safely
    hist_cols = [c for c in hist_cols if c in bin_centers]

    def extract_xy(row: pd.Series):
        xs = []
        ys = []
        for col in hist_cols:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            xs.append(bin_centers[col])
            ys.append(float(val))
        return xs, ys

    # Now actually build the series; tx/ty/ax/ay are always defined here
    tx, ty = extract_xy(target_row)
    ax, ay = extract_xy(apex_row)


    # --- Add target / apex curves ---

    fig.add_trace(
        go.Scatter(
            x=tx,
            y=ty,
            mode="lines+markers",
            name="This build",
            line=dict(width=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ax,
            y=ay,
            mode="lines+markers",
            name="APEX build",
            line=dict(width=5, dash="dash"),
        )
    )

    # --- Extra traces: per-skill "avg of all builds" curves from data_assess ---

    skill_df = get_single_skill_assess_df(class_code)
    if skill_df is not None and not skill_df.empty:
        # Columns like net_rds_hist_1_3, net_rds_hist_4, ...
        net_cols = [c for c in skill_df.columns if c.startswith("net_rds_hist_")]

        # Map net_* columns to the same bin centers used above via their base name
        net_bin_centers = {}
        for col in net_cols:
            base_col = col.replace("net_", "", 1)  # net_rds_hist_4 -> rds_hist_4
            if base_col in bin_centers:
                net_bin_centers[col] = bin_centers[base_col]

        # Ensure we have at least some usable columns
        net_cols = [c for c in net_cols if c in net_bin_centers]

        def extract_skill_xy(skill_code: str):
            try:
                row_s = skill_df.loc[skill_df["sk_name"] == skill_code].iloc[0]
            except (KeyError, IndexError):
                return None, None
            xs_s = []
            ys_s = []
            for col in net_cols:
                val = row_s.get(col, 0)
                if pd.isna(val):
                    val = 0
                xs_s.append(net_bin_centers[col])
                ys_s.append(float(val))
            return xs_s, ys_s

        for sc in [s1, s2, s3, s4]:
            if not sc:
                continue
            xs_s, ys_s = extract_skill_xy(sc)
            if not xs_s or not ys_s:
                continue

            fig.add_trace(
                go.Scatter(
                    x=xs_s,
                    y=ys_s,
                    mode="lines",
                    name=f"<b>{sc}</b>: avg of builds",
                    line=dict(width=3, dash="dot"),
                    opacity=0.6,  # lighter than main curves
                )
            )

    
    # --- Vertical r95 markers (one per curve) ---

    r95_target = float(target_row.get("r95", np.nan))
    r95_apex = float(apex_row.get("r95", np.nan))
    ymax = max(ty + ay) if (ty and ay) else (max(ty) if ty else (max(ay) if ay else 0)) or 0
    yline_t   = ymax * 1.05 if ymax > 0 else 1
    yline_a   = ymax * 0.95 if ymax > 0 else 1

    shapes = []
    annotations = []

    if not np.isnan(r95_target):
        shapes.append(
            dict(
                type="line",
                x0=r95_target,
                x1=r95_target,
                y0=0,
                y1=yline_t,
                line=dict(color="rgba(0, 0, 150, 0.7)", width=2),
            )
        )
        annotations.append(
            dict(
                x=r95_target,
                y=yline_t,
                xanchor="center",
                yanchor="bottom",
                text=f"r95 (this: {r95_target:.1f})",
                showarrow=False,
                font=dict(size=10, color="rgba(0, 0, 150, 0.9)"),
            )
        )

    if not np.isnan(r95_apex):
        shapes.append(
            dict(
                type="line",
                x0=r95_apex,
                x1=r95_apex,
                y0=0,
                y1=yline_a,
                line=dict(color="rgba(150, 0, 0, 0.7)", width=2),
            )
        )
        annotations.append(
            dict(
                x=r95_apex,
                y=yline_a,
                xanchor="center",
                yanchor="bottom",
                text=f"r95 (APEX: {r95_apex:.1f})",
                showarrow=False,
                font=dict(size=10, color="rgba(150, 0, 0, 0.9)"),
            )
        )

    fig.update_layout(
        title=f"Rounds Histogram: {skill_code}",
        xaxis_title="Rounds to Win",
        yaxis_title="Count of Wins",
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

# ========== Tab 3 Helpers =========

@app.callback(
    Output("single-skill-select", "options"),
    Input("single-skill-class", "value"),
)
def update_single_skill_options(class_code):
    if not class_code:
        return []

    skills = get_base_skills_for_class(class_code)
    if not skills:
        return []

    skills = sorted(skills)  # always alphabetical
    opts = [{"label": skill_label(s), "value": s} for s in skills]
    return opts

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
    Output("single-skill-ranking-table", "data"),
    Input("single-skill-class", "value"),
)
def update_single_skill_ranking(class_code):
    df = get_single_skill_assess_df(class_code)
    if df is None or df.empty:
        return []

    # 🔹 1) Remove any duplicate columns (fixes "columns are not unique" warning)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df2 = df.copy()

    # Normalise column names so we can always refer to pct_80_95 / pct_95
    rename_map = {}
    if "80_95_pct" in df2.columns and "pct_80_95" not in df2.columns:
        rename_map["80_95_pct"] = "pct_80_95"
    if "95ile_pct" in df2.columns and "pct_95" not in df2.columns:
        rename_map["95ile_pct"] = "pct_95"
    if rename_map:
        df2.rename(columns=rename_map, inplace=True)

    # 🔹 2) Normalize core columns: sk_name, skill_tier, r_max, etc.
    if "sk_name" not in df2.columns:
        # Try to guess a skill-code column if sk_name was somehow renamed
        name_col = next(
            (c for c in df2.columns if c.lower().startswith("sk_")), 
            None
        )
        if name_col is None:
            return []
        df2.rename(columns={name_col: "sk_name"}, inplace=True)

    # Friendly display label: "Acr - Acrobat"
    df2["skill_label"] = df2["sk_name"].apply(lambda c: skill_label(str(c)))

    # 🔹 3) Build a ranking based on r_max (ties share same rank)
    if "r_max" in df2.columns:
        # Sort by best rating first
        df2 = df2.sort_values(["r_max", "sk_name"], ascending=[False, True])

        # Rank: higher r_max → lower rank number; ties share the same rank
        df2["skill_rank"] = (
            df2["r_max"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
    else:
        # Fallback if no r_max: just sort alphabetically and rank that
        df2 = df2.sort_values("sk_name")
        df2["skill_rank"] = (
            df2["sk_name"]
            .rank(method="min", ascending=True)
            .astype(int)
        )

    # 🔹 4) Final column selection for the table
    #   - skill_tier is the 1/2/3/4/6 tier (for icons)
    #   - skill_rank is the positional rank we just defined
    cols = [
        "skill_rank",   # new ranking index
        "skill_label",  # "ABC - Full Name"
        "skill_tier",   # 1,2,3,4,6 → face icon
        "r_max",
        "nq_pct",
        "sub80_pct",
        "pct_80_95",
        "pct_95",
    ]

    cols = [c for c in cols if c in df2.columns]

    return df2[cols].to_dict("records")




if __name__ == "__main__":
    app.run(debug=True)
