# app2.py - v0.2
#   0.2  - implemented both 2-Skill Explorer and Skill Combo Detail locally for G2-G4 (20251207)

import os
from pathlib import Path

import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update, callback_context

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

CLASS_ALL_DATA_FILES = {
    "G2": CLASS_DIR / "G2_all_data_pass4_unique.csv",
    "G3": CLASS_DIR / "G3_all_data_pass4_unique.csv",
    "G4": CLASS_DIR / "G4_all_data_pass4_unique.csv",
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

# def load_all_data_by_class():
#     """
#     Load the per-class *_all_data_pass4_unique.csv files and
#     precompute a canonical skill_key = sorted 4-skill code string,
#     e.g. ['Acr','All','Des','Whi'] -> 'AcrAllDesWhi'.
#     """
#     all_data = {}
#     for code, path in CLASS_ALL_DATA_FILES.items():
#         if not path.exists():
#             continue

#         df = pd.read_csv(path)

#         # Be robust: some files use 'skill_code', some reuse 'skill_name'
#         skill_col = "skill_code"
#         if skill_col not in df.columns:
#             skill_col = "skill_name"

#         parsed = df[skill_col].apply(parse_skill_codes)
#         df["class_code"] = parsed.apply(lambda x: x[0])
#         df["skill_list"] = parsed.apply(lambda x: x[1])

#         # Canonical key for lookups
#         df["skill_key"] = df["skill_list"].apply(
#             lambda lst: "".join(sorted(lst))
#         )

#         all_data[code] = df

#     return all_data

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
    Lazy loader for *_all_data_pass4_unique.csv.

    - First call for a class: load, parse, create skill_key, cache.
    - Subsequent calls: reuse cached DataFrame.
    """
    if not class_code:
        return None

    if class_code in _all_data_cache:
        return _all_data_cache[class_code]

    path = CLASS_ALL_DATA_FILES.get(class_code)
    if not path or not path.exists():
        return None

    df = pd.read_csv(path)

    # Be robust: skill_code vs skill_name
    skill_col = "skill_code"
    if skill_col not in df.columns:
        skill_col = "skill_name"

    parsed = df[skill_col].apply(parse_skill_codes)
    df["class_code"] = parsed.apply(lambda x: x[0])
    df["skill_list"] = parsed.apply(lambda x: x[1])
    df["skill_key"] = df["skill_list"].apply(lambda lst: "".join(sorted(lst)))

    _all_data_cache[class_code] = df
    return df
    
# all_data_by_class = get_all_data_for_class()

def precompute_class_data():
    class_df = {}
    class_skills = {}

    for code, path in CLASS_FINAL_DATA_FILES.items():
        df = pd.read_csv(path)

        # parse skills and store as list in a new column
        parsed = df["skill_name"].apply(parse_skill_codes)
        df["class_code"] = parsed.apply(lambda x: x[0])
        df["skill_list"] = parsed.apply(lambda x: x[1])

        # Unique skills used by this class
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

    Skills remain in ENTERED order (not sorted).
    """
    class_code = row["class_code"]
    class_info = class_meta.get(class_code, {})
    class_name = class_info.get("name", class_code)
    class_icon_src = class_info.get("icon_src", f"/assets/hero_classes/{class_code}.png")

    # Skill codes from row (order entered)
    skill_codes = row.get("skill_list", [])
    skill_codes = [c for c in skill_codes if c]

    # Build skill chunks
    skill_chunks = []
    for idx, sc in enumerate(skill_codes):
        meta = skill_lookup.get(sc, {})
        full_name = meta.get("full_name", sc)
        skill_icon_src = meta.get("icon_src", f"/assets/skill_icons/{sc}.png")

        skill_label = format_skill_name_with_info(full_name)

        parts = []

        # Separator before skills 2 and 4
        if idx in (1, 3):
            parts.append(html.Span(" | ", className="skill-separator"))

        # Icon first
        parts.append(
            html.Img(
                src=skill_icon_src,
                className="skill-icon",
                title=full_name,
            )
        )

        parts.append(html.Span(" ", className="skill-label-icon-space"))
        parts.append(skill_label)

        skill_chunks.append(parts)

    # Split into two lines
    line1_chunks = []
    line2_chunks = []

    for i, parts in enumerate(skill_chunks):
        if i < 2:
            line1_chunks.extend(parts)
        else:
            line2_chunks.extend(parts)

    skills_lines = [html.Div(line1_chunks, className="skills-line")]
    if line2_chunks:
        skills_lines.append(html.Div(line2_chunks, className="skills-line"))

    return html.Div(
        [
            # Left side: icon + class label
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

            # Right side: skills block (indented)
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
                    build_build_substats_table(row),  # (2) left
                    build_single_skill_table(row),    # (3) right
                ],
                className="build-subtables-row",
                style={
                    "display": "flex",
                    "alignItems": "flex-start",  # top align
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

        icon_cell = html.Img(
            src=skill_icon_src,
            className="single-skill-icon",
            title=full_name,
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
                    style={"fontSize": "1.2em"},  # +20%
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
                    html.Td(skill_label_str, style=body_cell_style),
                    html.Td(rating_children, style=body_cell_style),
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
                                    html.Img(
                                        id="skill1-icon",
                                        style={"height": "32px"},
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
                                    html.Img(
                                        id="skill2-icon",
                                        style={"height": "32px"},
                                    ),
                                ],
                            ),
                            html.Br(),
                            html.Div(id="selection-summary"),
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
                                        "width": "34%",
                                    },
                                    {
                                        "if": {"column_id": "s4_full"},
                                        "width": "34%",
                                    },
                                    {
                                        "if": {"column_id": "raw_rating"},
                                        "width": "13%",
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
            html.H4(id="heatmap-title", style={**TITLE_BANNER_STYLE, "borderRadius": "2px"}),

            dcc.Graph(
                id="combo-heatmap",
                figure=go.Figure(),
                config={"displayModeBar": False},
                style={"height": "800px", "width": "100%"},
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
                                                html.Img(
                                                    id=f"detail-skill{i}-icon",
                                                    style={"height": "32px"},
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

            dcc.Graph(
                id="combo-detail-histogram",
                figure=go.Figure(),
                config={"displayModeBar": False},
                style={"height": "400px", "width": "75%"},
            ),
        ],
    )


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

    base_skills = class_skills.get(class_code, [])
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
    Output("step3-title", "children"),
    Output("heatmap-title", "children"),
    Output("selection-summary", "children"),
    Output("combo-table", "data"),
    Output("combo-heatmap", "figure"),
    Input("hero-class", "value"),
    Input("skill1", "value"),
    Input("skill2", "value"),
)
def update_outputs(class_code, skill1, skill2):
    import plotly.graph_objects as go

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
            ticktext=["50", 75, 100],
        ),
        xgap=1,
        ygap=1,
    )

    fig = go.Figure(data=[heat_base, heat_num])

    fig.update_layout(
        xaxis=dict(
            title="",
            side="top",
            tickmode="array",
            tickvals=axis_skills,
            ticktext=axis_skills,
            tickfont=dict(color="#000000", size=12),
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
            tickfont=dict(color="#000000", size=12),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=80, r=20, t=100, b=80),
        clickmode="event+select",
    )

    # ----- Simple icon PoC (Acr.png everywhere for now) -----
    # Assumes you have:  assets/skill_icons/Acr.png
    images = []
    for s in axis_skills:
        images.append(
            dict(
                source=f"/assets/skill_icons/{s}.png",
                xref="x",
                yref="paper",
                x=s,
                y=1.06,      # higher above the grid
                sizex=0.80,   # wider icon
                sizey=0.12,  # taller band
                xanchor="center",
                yanchor="bottom",
                layer="above",
            )
        )

    fig.update_layout(images=images)
    
    # # Left column of icons (beside rows)
    # for s in axis_skills:
    #     images.append(
    #         dict(
    #             source=f"/assets/skill_icons/{s}.png",
    #             xref="paper",
    #             yref="y",
    #             x=-0.08,
    #             y=s,
    #             sizex=0.08,
    #             sizey=0.5,
    #             xanchor="right",
    #             yanchor="middle",
    #             layer="above",
    #         )
    #     )

    # fig.update_layout(images=images)

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
    Input("selected-combo", "data"),     # heatmap click → store
    Input("combo-table", "active_cell"), # rank click in table
    State("main-tabs", "value"),         # current tab
    prevent_initial_call=True,
)
def switch_tabs(sel_combo, active_cell, current_tab):
    """
    Switch to the Skill Combo Detail tab when:
      - user clicks a heatmap cell (selected-combo updated), OR
      - user clicks a Rank cell in the S3/S4 combo table.

    Otherwise, leave the current tab alone.
    """
    ctx = callback_context
    if not ctx.triggered:
        # Nothing meaningful triggered this callback
        return no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # 1️⃣ Rank click in the S3/S4 table
    if trigger_id == "combo-table" and active_cell:
        if active_cell.get("column_id") == "rank":
            return "tab-combo-detail"
        # clicked some other column → don’t change tabs
        return current_tab

    # 2️⃣ Heatmap click → selected-combo updated
    if trigger_id == "selected-combo" and sel_combo:
        return "tab-combo-detail"

    # 3️⃣ Fallback: keep whatever tab we’re already on
    return current_tab

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

# @app.callback(
#     Output("detail-hero-class", "value"),
#     Output("detail-skill1", "value"),
#     Output("detail-skill2", "value"),
#     Output("detail-skill3", "value"),
#     Output("detail-skill4", "value"),
#     Input("selected-combo", "data"),
#     prevent_initial_call=True,
# )
# def populate_detail_from_store(data):
#     """
#     When user clicks a heatmap cell, `selected-combo` is updated.
#     Use that to pre-fill the detail tab dropdowns.
#     """
#     if not data:
#         # keep whatever is already there
#         return no_update, no_update, no_update, no_update, no_update

#     class_code = data.get("class_code")
#     core = data.get("core", [])
#     extra = data.get("extra", [])

#     s1 = core[0] if len(core) > 0 else None
#     s2 = core[1] if len(core) > 1 else None
#     s3 = extra[0] if len(extra) > 0 else None
#     s4 = extra[1] if len(extra) > 1 else None

#     return class_code, s1, s2, s3, s4

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

    base_skills = class_skills.get(class_code, [])
    if not base_skills:
        return [], [], [], []

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
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ax,
            y=ay,
            mode="lines+markers",
            name="APEX build",
            line=dict(width=3, dash="dash"),
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


if __name__ == "__main__":
    app.run(debug=True)
