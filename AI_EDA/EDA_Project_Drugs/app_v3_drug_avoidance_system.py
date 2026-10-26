"""Drug Avoidance System — updated EDA version.

Run from a terminal:
    streamlit run app_v3_drug_avoidance_system.py

Place cleaned_drugs_v2.csv beside this script or inside a DataSet folder.
The original drugs_side_effects_drugs_com.csv is optional but enables
original-versus-cleaned comparisons.

Educational EDA only. Project-defined keyword scores and avoid_drug labels are
not clinically validated and must not be used for medication decisions.
"""
from pathlib import Path
import html
import math
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration and styling (matching the existing teal dashboard design)
# -----------------------------------------------------------------------------
TEAL = "#0d9488"
ORANGE = "#d97706"
RED = "#b91c1c"
GRAY = "#94a3b8"

st.set_page_config(page_title="Drug Avoidance System", layout="wide")
st.markdown(
    f"""
    <style>
    .insight {{
        border-left: 5px solid {TEAL};
        background: rgba(13, 148, 136, 0.10);
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0 1.4rem 0;
    }}
    .insight small {{ opacity: 0.88; }}
    .warning-box {{
        border-left: 5px solid {RED};
        background: rgba(185, 28, 28, 0.08);
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1.2rem 0;
    }}
    [data-testid="stMetric"] {{
        background: rgba(13, 148, 136, 0.08);
        padding: 0.8rem 1rem;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent


def first_existing(paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)


ORIGINAL_PATH = first_existing([
    BASE_DIR / "drugs_side_effects_drugs_com.csv",
    BASE_DIR / "DataSet" / "drugs_side_effects_drugs_com.csv",
    BASE_DIR / "drugs_side_effects_drugs_com(1).csv",
    BASE_DIR / "DataSet" / "drugs_side_effects_drugs_com(1).csv",
])
CLEANED_PATH = first_existing([
    BASE_DIR / "cleaned_drugs_v2.csv",
    BASE_DIR / "DataSet" / "cleaned_drugs_v2.csv",
    BASE_DIR / "cleaned_drugs_v2(1).csv",
    BASE_DIR / "DataSet" / "cleaned_drugs_v2(1).csv",
    BASE_DIR / "cleaned_drugs_side_effects_drugs_com.csv",
    BASE_DIR / "DataSet" / "cleaned_drugs_side_effects_drugs_com.csv",
    BASE_DIR / "cleaned_drugs_side_effects_drugs_com(2).csv",
    BASE_DIR / "DataSet" / "cleaned_drugs_side_effects_drugs_com(2).csv",
    BASE_DIR / "cleaned_drugs.csv",
    BASE_DIR / "DataSet" / "cleaned_drugs.csv",
])


def insight(message, caveat=None):
    """Display an insight and optional interpretation limitation."""
    safe_message = html.escape(str(message))
    extra = f"<br><small>{html.escape(str(caveat))}</small>" if caveat else ""
    st.markdown(
        f'<div class="insight"><b>Insight</b><br>{safe_message}{extra}</div>',
        unsafe_allow_html=True,
    )


def warning_note(message):
    st.markdown(
        f'<div class="warning-box"><b>Important limitation</b><br>{html.escape(str(message))}</div>',
        unsafe_allow_html=True,
    )


def add_numeric_helpers(data):
    """Return analysis copy with numeric ratings/review counts."""
    result = data.copy()
    if "new_rating" not in result.columns:
        result["new_rating"] = pd.to_numeric(result.get("rating"), errors="coerce")
    else:
        result["new_rating"] = pd.to_numeric(result["new_rating"], errors="coerce")
    if "new_no_of_reviews" not in result.columns:
        result["new_no_of_reviews"] = pd.to_numeric(result.get("no_of_reviews"), errors="coerce")
    else:
        result["new_no_of_reviews"] = pd.to_numeric(result["new_no_of_reviews"], errors="coerce")
    if "rating" in result:
        result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    if "no_of_reviews" in result:
        result["no_of_reviews"] = pd.to_numeric(result["no_of_reviews"], errors="coerce")
    return result


def missing_percent(data):
    return data.isna().mean().mul(100) if len(data.columns) else pd.Series(dtype=float)


def outlier_table(data):
    rows = []
    for column, label in [("new_rating", "Rating"), ("new_no_of_reviews", "Number of reviews")]:
        if column not in data:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "Variable": label,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower bound": lower,
            "Upper bound": upper,
            "Potential outliers": int(((values < lower) | (values > upper)).sum()),
            "Observed minimum": values.min(),
            "Observed maximum": values.max(),
        })
    return pd.DataFrame(rows)


def drugs_per_condition(data):
    required = {"medical_condition", "drug_name"}
    if not required.issubset(data.columns):
        return pd.Series(dtype=int)
    return (
        data.dropna(subset=["medical_condition", "drug_name"])
        .groupby("medical_condition")["drug_name"]
        .nunique()
        .sort_values(ascending=False)
    )


def build_side_effects_long(data):
    """Split the long free-text field at commas and semicolons, as in the notebook."""
    required = {"drug_name", "medical_condition", "side_effects"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=["drug_name", "medical_condition", "side_effects", "record_id", "side_effect"])
    long_data = data[["drug_name", "medical_condition", "side_effects"]].copy()
    long_data = long_data.dropna(subset=["drug_name", "side_effects"]).copy()
    long_data["record_id"] = long_data.index
    long_data["side_effect"] = (
        long_data["side_effects"].astype(str).str.lower().str.strip(" ()[]{}")
        .str.split(r"\s*[;,]\s*")
    )
    long_data = long_data.explode("side_effect")
    long_data["side_effect"] = long_data["side_effect"].astype(str).str.strip(" \t\n.,;:()[]{}")
    return long_data[long_data["side_effect"].ne("")].copy()


def side_effect_term_counts(long_data):
    if long_data.empty:
        return pd.Series(dtype=int)
    return long_data.drop_duplicates(["record_id", "side_effect"])["side_effect"].value_counts()


def drug_effect_summary(long_data):
    if long_data.empty:
        return pd.Series(dtype=int)
    return (
        long_data.drop_duplicates(["drug_name", "side_effect"])
        .groupby("drug_name")["side_effect"].nunique()
        .sort_values(ascending=True)
    )


WARNING_PATTERNS = {
    "Breathing difficulties": r"shortness of breath|difficulty breathing|difficult breathing|trouble breathing",
    "Severe allergic reaction": r"anaphylaxis|severe allergic reaction",
    "Swelling of face or throat": r"swelling of (the )?(face|lips|tongue|throat)",
    "Chest pain": r"chest pain",
    "Seizures": r"\bseizures?\b",
    "Suicidal thoughts": r"suicidal thoughts?|thoughts of suicide",
    "Abnormal heartbeat": r"irregular heartbeat|abnormal heart rhythm",
    "Unusual bleeding": r"unusual bleeding|bleeding that won't stop|bleeding that will not stop",
    "Fainting": r"\bfainting\b|\bfaint\b",
}


def warning_summary(data, categories, min_desc=2):
    if not {"drug_name", "side_effects"}.issubset(data.columns):
        return pd.DataFrame(columns=["descriptions_reviewed", "descriptions_with_warning", "warning_description_percentage"])
    warning_data = data[["drug_name", "side_effects"]].dropna().drop_duplicates().copy()
    if warning_data.empty:
        return pd.DataFrame(columns=["descriptions_reviewed", "descriptions_with_warning", "warning_description_percentage"])
    text = warning_data["side_effects"].astype(str)
    for category in categories:
        warning_data[category] = text.str.contains(WARNING_PATTERNS[category], case=False, regex=True, na=False)
    warning_data["has_potential_warning"] = warning_data[list(categories)].any(axis=1) if categories else False
    summary = warning_data.groupby("drug_name").agg(
        descriptions_reviewed=("side_effects", "nunique"),
        descriptions_with_warning=("has_potential_warning", "sum"),
    )
    summary["warning_description_percentage"] = (
        summary["descriptions_with_warning"] / summary["descriptions_reviewed"] * 100
    )
    summary = summary[summary["descriptions_reviewed"] >= min_desc]
    return summary.sort_values(["warning_description_percentage", "descriptions_reviewed"], ascending=[False, False])


def rating_data(data):
    return data.dropna(subset=["new_rating", "medical_condition", "drug_name"]).copy()


def condition_rating_summary(data):
    rated = rating_data(data)
    if rated.empty:
        return pd.DataFrame(columns=["mean_rating", "median_rating", "rating_count"])
    return rated.groupby("medical_condition").agg(
        mean_rating=("new_rating", "mean"),
        median_rating=("new_rating", "median"),
        rating_count=("new_rating", "count"),
    ).sort_values("mean_rating", ascending=False)


# -----------------------------------------------------------------------------
# Load datasets
# -----------------------------------------------------------------------------
if CLEANED_PATH is None:
    st.error(
        "Could not find a cleaned CSV. Place cleaned_drugs_v2.csv beside this script "
        "or in a DataSet folder."
    )
    st.stop()

try:
    cleaned_raw = pd.read_csv(CLEANED_PATH)
    original_raw = pd.read_csv(ORIGINAL_PATH) if ORIGINAL_PATH else None
except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
    st.error(f"Could not read the selected CSV file: {error}")
    st.stop()

required = {"drug_name", "medical_condition", "side_effects", "rx_otc"}
missing = required.difference(cleaned_raw.columns)
if missing:
    st.error("The cleaned dataset is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()
if original_raw is not None and not required.issubset(original_raw.columns):
    original_raw = None

cleaned_df = add_numeric_helpers(cleaned_raw)
original_df = add_numeric_helpers(original_raw) if original_raw is not None else None

# -----------------------------------------------------------------------------
# Dashboard header and sidebar
# -----------------------------------------------------------------------------
st.title("Drug Avoidance System")
st.caption("EDA of drug side effects and medical conditions — not a clinical recommendation")

st.sidebar.header("Dataset description")
st.sidebar.markdown(
    "Explore recorded drug names, medical conditions, side-effect descriptions, prescription status, ratings, and reviews."
)
st.sidebar.header("Filters")
versions = ["Cleaned"] + (["Original"] if original_df is not None else [])
version = st.sidebar.selectbox("Dataset version", versions, index=0)
base_df = cleaned_df if version == "Cleaned" else original_df
raw_base_df = cleaned_raw if version == "Cleaned" else original_raw

rx_values = sorted(base_df["rx_otc"].dropna().astype(str).unique().tolist())
rx_choice = st.sidebar.selectbox("Prescription status (Rx / OTC)", ["All"] + rx_values)
condition_values = sorted(base_df["medical_condition"].dropna().astype(str).unique().tolist())
selected_conditions = st.sidebar.multiselect(
    "Medical condition", condition_values,
    help="Leave empty to include every condition."
)
drug_search = st.sidebar.text_input("Search drug name", "")
rating_range = st.sidebar.slider("Rating range", 0.0, 10.0, (0.0, 10.0), step=0.5)
max_review_value = int(max(100, math.ceil(float(base_df["new_no_of_reviews"].max() or 100) / 100) * 100))
max_review_value = min(max_review_value, 10000)
min_reviews = st.sidebar.slider("Minimum number of reviews", 0, max_review_value, 0, step=10 if max_review_value >= 100 else 1)
keep_missing = st.sidebar.checkbox("Keep records with missing rating or review count", value=True)
top_n = st.sidebar.slider("Items shown in charts", 5, 20, 10)

mask = pd.Series(True, index=base_df.index)
if selected_conditions:
    mask &= base_df["medical_condition"].isin(selected_conditions)
if rx_choice != "All":
    mask &= base_df["rx_otc"].astype(str).eq(rx_choice)
if drug_search.strip():
    mask &= base_df["drug_name"].astype(str).str.contains(drug_search.strip(), case=False, na=False)
rating_ok = base_df["new_rating"].between(*rating_range, inclusive="both")
reviews_ok = base_df["new_no_of_reviews"].ge(min_reviews)
if keep_missing:
    rating_ok |= base_df["new_rating"].isna()
    reviews_ok |= base_df["new_no_of_reviews"].isna()
filtered_df = base_df.loc[mask & rating_ok & reviews_ok].copy()
raw_filtered = raw_base_df.loc[filtered_df.index].copy()

if filtered_df.empty:
    st.warning("No records match these filters. Widen the filters in the sidebar.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(filtered_df):,}")
k2.metric("Distinct drug names", f"{filtered_df['drug_name'].nunique():,}")
k3.metric("Medical conditions", f"{filtered_df['medical_condition'].nunique():,}")
mean_rating = filtered_df["new_rating"].mean()
k4.metric("Mean recorded rating", f"{mean_rating:.2f}" if pd.notna(mean_rating) else "N/A")
st.caption(
    f"Dataset: {version.lower()} | Showing {len(filtered_df):,} of {len(base_df):,} records after filters. "
    f"File used: {CLEANED_PATH.name if version == 'Cleaned' else ORIGINAL_PATH.name}."
)

# -----------------------------------------------------------------------------
# Main content tabs
# -----------------------------------------------------------------------------
tab_data, tab_cond, tab_side, tab_warn, tab_rate, tab_dist, tab_rx_target = st.tabs([
    "Data & Cleaning", "Q1 Conditions", "Q2 Side Effects", "Q3 Potential Warnings",
    "Q4 Ratings", "Distributions & Outliers", "Rx/OTC & Target"
])

# TAB 1: Data and cleaning
with tab_data:
    st.subheader("Data preview")
    preview_rows = st.slider("Rows to preview", 5, 100, 10)
    st.dataframe(raw_filtered.head(preview_rows), use_container_width=True)
    st.subheader("Dataset dimensions")
    dim1, dim2 = st.columns(2)
    dim1.metric("Selected dataset", f"{raw_base_df.shape[0]:,} rows x {raw_base_df.shape[1]} columns")
    if original_raw is not None:
        dim2.metric("Original dataset", f"{original_raw.shape[0]:,} rows x {original_raw.shape[1]} columns")
        common_cols = [c for c in original_raw.columns if c in cleaned_raw.columns]
        miss = pd.DataFrame({
            "Original": missing_percent(original_raw[common_cols]),
            "Cleaned": missing_percent(cleaned_raw[common_cols]),
        })
        miss = miss[(miss["Original"] > 0) | (miss["Cleaned"] > 0)]
        if not miss.empty:
            miss_plot = miss.rename_axis("Column").reset_index().melt(
                id_vars="Column", var_name="Dataset", value_name="Missing percentage"
            )
            st.plotly_chart(
                px.bar(miss_plot, x="Column", y="Missing percentage", color="Dataset", barmode="group",
                       color_discrete_map={"Original": GRAY, "Cleaned": TEAL}, title="Missing values in shared columns (%)"),
                use_container_width=True,
            )
    else:
        dim2.metric("Original comparison", "Not available")
        st.info("Place the original CSV beside this app to enable original-versus-cleaned comparisons.")

    st.subheader("Potential outliers (IQR method)")
    st.dataframe(outlier_table(filtered_df).round(2), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.box(filtered_df, x="new_rating", points="outliers", title="Recorded rating", color_discrete_sequence=[TEAL]), use_container_width=True)
    with right:
        st.plotly_chart(px.box(filtered_df, x="new_no_of_reviews", points="outliers", title="Number of reviews", color_discrete_sequence=[TEAL]), use_container_width=True)
    min_rating_seen = filtered_df["new_rating"].min()
    min_reviews_seen = filtered_df["new_no_of_reviews"].min()
    no_negative = ((pd.isna(min_rating_seen) or min_rating_seen >= 0) and (pd.isna(min_reviews_seen) or min_reviews_seen >= 0))
    insight(
        "As seen from the plot, the outliers are true values since ratings and reviews can differ between each drug and each person. That is why some values reach 3,000 reviews or a rating of 2. From observing the plot, we can also tell that there are no invalid values such as negative numbers."
    )
    st.download_button(
        "Download filtered dataset (CSV)",
        data=raw_filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_drug_eda.csv", mime="text/csv"
    )

# TAB 2: Conditions
with tab_cond:
    st.subheader("Q1: Which medical conditions have the most available listed drugs?")
    condition_counts = drugs_per_condition(filtered_df)
    if condition_counts.empty:
        st.info("No condition/drug pairs are available under the selected filters.")
    else:
        shown = condition_counts.head(top_n).sort_values()
        fig = px.bar(x=shown.values, y=shown.index, orientation="h",
                     labels={"x": "Distinct drug names", "y": "Medical condition"},
                     title=f"Top {len(shown)} Medical Conditions by Distinct Drug Names",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "The medical condition that has the largest amount of drugs would be Colds & Flu, due to it being a common symptom and being what medicinal companies target the most in terms of business."
        )

# TAB 3: Side effect term exploration
with tab_side:
    st.subheader("Q2: Which side-effect terms are mentioned most often?")
    long_data = build_side_effects_long(filtered_df)
    term_counts = side_effect_term_counts(long_data)
    if term_counts.empty:
        st.info("No side-effect descriptions are available under these filters.")
    else:
        top_terms = term_counts.head(top_n).sort_values()
        st.plotly_chart(
            px.bar(x=top_terms.values, y=top_terms.index, orientation="h",
                   labels={"x": "Records mentioning the term", "y": "Extracted text term"},
                   title=f"Top {len(top_terms)} Most Frequently Mentioned Side-Effect Terms",
                   color_discrete_sequence=[TEAL]), use_container_width=True
        )
        insight(
            "As we can see, the most common side-effect terms across all the drugs are lips, then tongue, then hives. Whether these side effects are dangerous cannot be determined without reviewing them with a real doctor."
        )
        effects_by_drug = drug_effect_summary(long_data)
        if not effects_by_drug.empty:
            fewest = effects_by_drug.head(top_n).sort_values(ascending=False)
            st.plotly_chart(
                px.bar(x=fewest.values, y=fewest.index, orientation="h",
                       labels={"x": "Distinct extracted terms", "y": "Drug name"},
                       title=f"{len(fewest)} Drugs with the Fewest Recorded Side-Effect Terms",
                       color_discrete_sequence=[ORANGE]), use_container_width=True
            )
            insight(
                "We can see that guselkumab and ubrogepant are the drugs with the fewest recorded side-effect terms. However, we cannot conclude that using them means you will not be affected at all."
            )
            with st.expander("Inspect descriptions for the drugs with the fewest terms"):
                st.dataframe(
                    filtered_df[filtered_df["drug_name"].isin(effects_by_drug.head(top_n).index)]
                    [["drug_name", "medical_condition", "side_effects"]].drop_duplicates(),
                    use_container_width=True
                )

# TAB 4: Potential warning terms
with tab_warn:
    st.subheader("Q3: Which drug descriptions mention selected potential warning terms?")
    c1, c2 = st.columns([3, 1])
    categories = c1.multiselect("Warning categories", list(WARNING_PATTERNS), default=list(WARNING_PATTERNS))
    min_desc = c2.slider("Minimum distinct descriptions per drug", 1, 10, 2)
    if not categories:
        st.info("Select at least one warning category.")
    else:
        summary = warning_summary(filtered_df, categories, min_desc)
        if summary.empty:
            st.info("No drugs match the selected minimum-description requirement. Lower the minimum or widen the filters.")
        else:
            shown = summary.head(top_n).sort_values("warning_description_percentage")
            st.plotly_chart(
                px.bar(shown, x="warning_description_percentage", y=shown.index, orientation="h", range_x=[0, 100],
                       hover_data=["descriptions_with_warning", "descriptions_reviewed"],
                       labels={"warning_description_percentage": "Descriptions mentioning warning terms (%)", "y": "Drug name"},
                       title="Drugs Whose Descriptions Frequently Mention Potential Warnings",
                       color_discrete_sequence=[RED]), use_container_width=True
            )
            insight(
                "Anything that causes discomfort to the patient is considered a problem. Though this is not a measure of how often patients experience these effects, doctors should still take them into consideration when prescribing medication."
            )
            with st.expander("View the warning summary table"):
                st.dataframe(summary.round(1), use_container_width=True)

# TAB 5: Rating comparisons and correlation
with tab_rate:
    st.subheader("Q4: How are recorded ratings distributed across medical conditions?")
    cond_rating = condition_rating_summary(filtered_df)
    if cond_rating.empty:
        st.info("No ratings are available under the selected filters.")
    else:
        n_conditions = min(top_n, 15, len(cond_rating))
        chosen_conditions = cond_rating.nlargest(n_conditions, "rating_count").index
        rated = rating_data(filtered_df)
        rating_plot = rated[rated["medical_condition"].isin(chosen_conditions)]
        st.plotly_chart(
            px.box(rating_plot, x="new_rating", y="medical_condition", points="outliers",
                   labels={"new_rating": "Recorded user rating", "medical_condition": "Medical condition"},
                   title=f"Rating Distribution for the {len(chosen_conditions)} Conditions with the Most Ratings",
                   color_discrete_sequence=[TEAL]), use_container_width=True
        )
        insight(
            "As you can see, the ratings of drugs related to these conditions are mostly distributed between 6 and 9, and have not been affected by the reported side effects."
        )
        with st.expander("View average and median ratings by condition"):
            st.dataframe(cond_rating.round(2), use_container_width=True)

    st.subheader("Relationship between rating and number of reviews")
    numeric_pairs = filtered_df[["new_rating", "new_no_of_reviews"]].dropna()
    if len(numeric_pairs) >= 2:
        corr = numeric_pairs["new_rating"].corr(numeric_pairs["new_no_of_reviews"])
        left, right = st.columns([1, 2])
        with left:
            corr_matrix = numeric_pairs[["new_no_of_reviews", "new_rating"]].corr()
            st.plotly_chart(px.imshow(corr_matrix, text_auto=".3f", zmin=-1, zmax=1,
                                      color_continuous_scale="RdBu_r", title="Correlation Heatmap"), use_container_width=True)
        with right:
            use_log = st.checkbox("Use logarithmic scale for review counts", value=False)
            st.plotly_chart(
                px.scatter(filtered_df.dropna(subset=["new_rating", "new_no_of_reviews"]),
                           x="new_rating", y="new_no_of_reviews", opacity=0.6, log_y=use_log,
                           hover_data=["drug_name", "medical_condition"],
                           labels={"new_rating": "Rating", "new_no_of_reviews": "Number of reviews"},
                           title="Rating vs Number of Reviews", color_discrete_sequence=[TEAL]), use_container_width=True
            )
        insight(
            "Most records have relatively few reviews, while a small number have many reviews across different rating levels. The correlation between rating and number of reviews is -0.043, indicating a very weak negative linear relationship."
        )

# TAB 6: Distributions and averages
with tab_dist:
    st.subheader("Rating distribution")
    rated_values = filtered_df["new_rating"].dropna()
    review_values = filtered_df["new_no_of_reviews"].dropna()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(filtered_df, x="new_rating", nbins=30, marginal="rug",
                                     title="Distribution of Recorded Ratings", color_discrete_sequence=[TEAL]), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(filtered_df, x="new_no_of_reviews", nbins=30,
                                     title="Distribution of Number of Reviews", color_discrete_sequence=[TEAL]), use_container_width=True)
    rating_skew = rated_values.skew() if len(rated_values) > 2 else float("nan")
    review_skew = review_values.skew() if len(review_values) > 2 else float("nan")
    s1, s2 = st.columns(2)
    s1.metric("Rating skewness", f"{rating_skew:.2f}" if pd.notna(rating_skew) else "N/A")
    s2.metric("Review-count skewness", f"{review_skew:.2f}" if pd.notna(review_skew) else "N/A")
    rating_direction = "left" if pd.notna(rating_skew) and rating_skew < 0 else "right"
    review_direction = "right" if pd.notna(review_skew) and review_skew > 0 else "left"
    insight(
        f"Ratings are imbalanced with a skew of {rating_skew:.2f}, leaning to the left."
    )
    insight(
        f"The number of reviews is imbalanced, with a right skew of {review_skew:.2f}."
    )

    st.subheader("Drugs with the highest average review counts")
    review_by_drug = (
        filtered_df.dropna(subset=["drug_name", "new_no_of_reviews"])
        .groupby("drug_name")["new_no_of_reviews"].mean().sort_values(ascending=False)
    )
    if not review_by_drug.empty:
        top_reviews = review_by_drug.head(top_n).sort_values()
        st.plotly_chart(
            px.bar(x=top_reviews.values, y=top_reviews.index, orientation="h",
                   labels={"x": "Average number of reviews", "y": "Drug name"},
                   title=f"Top {len(top_reviews)} Drugs by Average Number of Reviews",
                   color_discrete_sequence=[TEAL]), use_container_width=True
        )
        insight(
            "As seen from the plot, each drug has its own reviews and ratings. Based on the type of drug, it can have multiple reviews, which is why some values reach around 3,000."
        )

# TAB 7: Rx/OTC and project-defined labels
with tab_rx_target:
    st.subheader("Prescription status distribution")
    rx_counts = filtered_df["rx_otc"].fillna("Unknown").astype(str).value_counts()
    rx_plot = rx_counts.rename_axis("Prescription status").reset_index(name="Record count")
    st.plotly_chart(
        px.bar(rx_plot, x="Prescription status", y="Record count", text="Record count",
               title="Recorded Prescription Status (Rx / Rx/OTC / OTC)", color_discrete_sequence=[TEAL]),
        use_container_width=True
    )
    rx_pct = rx_counts / rx_counts.sum() * 100
    insight(
        "The data is imbalanced, which makes sense because most patients require a prescription to get their drugs."
    )

    st.subheader("Project-defined target and scores")
    target_available = "avoid_drug" in filtered_df.columns
    if target_available:
        left, right = st.columns(2)
        avoid_counts = filtered_df["avoid_drug"].value_counts(dropna=False).sort_index()
        target_plot = pd.DataFrame({
            "Target label": ["1 — flagged" if x == 1 else "0 — not flagged" if x == 0 else "Missing" for x in avoid_counts.index],
            "Record count": avoid_counts.values,
        })
        left.plotly_chart(px.bar(target_plot, x="Target label", y="Record count", text="Record count",
                                 title="Project-Defined avoid_drug Target", color_discrete_sequence=[ORANGE]), use_container_width=True)
        score_columns = [c for c in ["side_effect_score", "condition_score"] if c in filtered_df]
        if score_columns:
            score_counts = filtered_df[score_columns].melt(var_name="Score type", value_name="Score")
            right.plotly_chart(px.histogram(score_counts.dropna(), x="Score", color="Score type", barmode="group",
                                            title="Project-Defined Score Values", color_discrete_map={"side_effect_score": TEAL, "condition_score": ORANGE}), use_container_width=True)
        # No insight box is displayed here because the user has not supplied
        insight(
            "The class is imbalanced.meaning avoiding accuracy since it wont rely any good info."
            " it also tells that some of drugs are flaged as avoid "
            "due to there side effects which could lead to potential harm due to their medical conditions however that isnt 100% true since drugs could have or have no effects on the patients"
            "that is why patients  must reconsile with their doctors on this subject "
        )
        # an insight for the target/score chart.
    else:
        st.info("The selected dataset has no avoid_drug target. Choose the cleaned v2 dataset to inspect the project-defined scores.")

st.sidebar.download_button(
    "Download filtered CSV",
    data=raw_filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_drug_eda.csv",
    mime="text/csv",
)
