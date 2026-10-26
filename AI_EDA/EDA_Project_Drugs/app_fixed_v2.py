"""Drug side-effects and ratings dashboard.

Run this app from the PyCharm terminal with:
    D:\\ai310\\Scripts\\python.exe -m streamlit run "C:\\Users\\abdal\\PycharmProjects\\AI_EDA\\EDA_Project_Drugs\\app.py"

The dashboard reads the original and cleaned CSV files directly with pandas.
"""

import html

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TEAL = "#0d9488"

ORIGINAL_CSV_PATH = r"C:\Users\abdal\PycharmProjects\AI_EDA\EDA_Project_Drugs\DataSet\drugs_side_effects_drugs_com.csv"
CLEANED_CSV_PATH = r"C:\Users\abdal\PycharmProjects\AI_EDA\EDA_Project_Drugs\DataSet\cleaned_drugs_side_effects_drugs_com.csv"

st.set_page_config(page_title="Drug Insights Dashboard", layout="wide")

st.markdown(
    f"""
    <style>
    .insight {{
        border-left: 5px solid {TEAL};
        background: rgba(13, 148, 136, 0.10);
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0 1.6rem 0;
    }}
    .insight small {{ opacity: 0.85; }}
    [data-testid="stMetric"] {{
        background: rgba(13, 148, 136, 0.08);
        padding: 0.8rem 1rem;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def insight(text, caveat=None):
    """Display an insight and an optional limitation below a chart."""
    extra = f"<br><small>{caveat}</small>" if caveat else ""
    st.markdown(
        f'<div class="insight"><b>Insight</b><br>{text}{extra}</div>',
        unsafe_allow_html=True,
    )


def esc(value):
    """Escape dataset text before placing it into an HTML insight."""
    return html.escape(str(value))


def add_numeric_helpers(data):
    """Return an analysis copy with numeric rating/review columns.

    The original CSV DataFrame is not modified, so its displayed column count
    remains the true number of columns stored in that file.
    """
    result = data.copy()
    if "new_rating" not in result.columns:
        result["new_rating"] = pd.to_numeric(result["rating"], errors="coerce")
    else:
        result["new_rating"] = pd.to_numeric(result["new_rating"], errors="coerce")

    if "new_no_of_reviews" not in result.columns:
        result["new_no_of_reviews"] = pd.to_numeric(
            result["no_of_reviews"], errors="coerce"
        )
    else:
        result["new_no_of_reviews"] = pd.to_numeric(
            result["new_no_of_reviews"], errors="coerce"
        )
    return result


def missing_percent(data):
    """Return missing-value percentages for the supplied DataFrame."""
    if data.empty and len(data.columns) == 0:
        return pd.Series(dtype=float)
    return data.isna().mean().mul(100)


def outlier_table(data):
    """Summarize potential IQR outliers in the numeric rating/review columns."""
    rows = []
    for column in ("new_rating", "new_no_of_reviews"):
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "feature": column,
                    "Q1": float("nan"),
                    "Q3": float("nan"),
                    "IQR": float("nan"),
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                    "potential_outliers": 0,
                }
            )
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "feature": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "potential_outliers": int(((values < lower) | (values > upper)).sum()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Q1", "Q3", "IQR", "lower_bound", "upper_bound", "potential_outliers"]
        )
    return pd.DataFrame(rows).set_index("feature")


def drugs_per_condition(data):
    """Count distinct listed drug names for each medical condition."""
    return (
        data.dropna(subset=["medical_condition", "drug_name"])
        .groupby("medical_condition")["drug_name"]
        .nunique()
        .sort_values(ascending=False)
    )


def build_side_effects_long(data):
    """Split side-effect descriptions on commas/semicolons for exploratory counts.

    These extracted strings are only a first-pass text analysis; they may be
    sentence fragments and should not be treated as clinically standardized terms.
    """
    required = {"drug_name", "side_effects"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=["drug_name", "side_effects", "record_id", "side_effect"])

    long_data = (
        data[["drug_name", "side_effects"]]
        .dropna(subset=["drug_name", "side_effects"])
        .drop_duplicates()
        .copy()
        .reset_index(drop=True)
    )
    long_data["record_id"] = long_data.index
    long_data["side_effect"] = (
        long_data["side_effects"]
        .astype(str)
        .str.lower()
        .str.strip(" ()[]{}")
        .str.split(r"\s*[;,]\s*")
    )
    long_data = long_data.explode("side_effect")
    long_data["side_effect"] = long_data["side_effect"].astype(str).str.strip(" \t\n.,;:()[]{}")
    long_data = long_data[long_data["side_effect"].ne("")].copy()
    return long_data


def side_effect_term_counts(long_data):
    """Count distinct record mentions of each extracted term."""
    if long_data.empty:
        return pd.Series(dtype=int)
    return (
        long_data.drop_duplicates(["record_id", "side_effect"])["side_effect"]
        .value_counts()
    )


def drug_effect_summary(long_data):
    """Count distinct extracted terms for each drug, ascending."""
    if long_data.empty:
        return pd.Series(dtype=int)
    return (
        long_data.drop_duplicates(["drug_name", "side_effect"])
        .groupby("drug_name")["side_effect"]
        .nunique()
        .sort_values(ascending=True)
    )


WARNING_PATTERNS = {
    "Breathing difficulties": r"shortness of breath|difficulty breathing|difficult breathing|trouble breathing",
    "Severe allergic reaction": r"anaphylaxis|severe allergic reaction",
    "Swelling of face or throat": r"swelling of (?:the )?(?:face|lips|tongue|throat)",
    "Chest pain": r"chest pain",
    "Seizures": r"\bseizures?\b",
    "Suicidal thoughts": r"suicidal thoughts?|thoughts of suicide",
    "Abnormal heartbeat": r"irregular heartbeat|abnormal heart rhythm",
    "Unusual bleeding": r"unusual bleeding|bleeding that won't stop|bleeding that will not stop",
    "Fainting": r"\bfainting\b|\bfaint\b",
}


def warning_summary(data, categories, min_desc=2):
    """Summarize how often descriptions mention selected warning keywords."""
    if not {"drug_name", "side_effects"}.issubset(data.columns):
        return pd.DataFrame(
            columns=["descriptions_reviewed", "descriptions_with_warning", "warning_description_percentage"]
        )
    warning_data = (
        data[["drug_name", "side_effects"]]
        .dropna(subset=["drug_name", "side_effects"])
        .drop_duplicates()
        .copy()
    )
    if warning_data.empty:
        return pd.DataFrame(
            columns=["descriptions_reviewed", "descriptions_with_warning", "warning_description_percentage"]
        )
    for category in categories:
        pattern = WARNING_PATTERNS[category]
        warning_data[category] = warning_data["side_effects"].str.contains(
            pattern, case=False, regex=True, na=False
        )
    selected_categories = list(categories)
    if selected_categories:
        warning_data["has_potential_warning"] = warning_data[selected_categories].any(axis=1)
    else:
        warning_data["has_potential_warning"] = False

    summary = warning_data.groupby("drug_name").agg(
        descriptions_reviewed=("side_effects", "nunique"),
        descriptions_with_warning=("has_potential_warning", "sum"),
    )
    summary["warning_description_percentage"] = (
        summary["descriptions_with_warning"] / summary["descriptions_reviewed"] * 100
    )
    summary = summary[summary["descriptions_reviewed"] >= min_desc]
    return summary.sort_values(
        ["warning_description_percentage", "descriptions_reviewed"], ascending=[False, False]
    )


def rating_data(data):
    """Return rows with recorded ratings and a medical condition/drug name."""
    return data.dropna(subset=["new_rating", "medical_condition", "drug_name"]).copy()


def condition_rating_summary(data):
    """Summarize the average, median, and number of ratings per condition."""
    rated = rating_data(data)
    if rated.empty:
        return pd.DataFrame(columns=["mean_rating", "median_rating", "rating_count"])
    return (
        rated.groupby("medical_condition")
        .agg(
            mean_rating=("new_rating", "mean"),
            median_rating=("new_rating", "median"),
            rating_count=("new_rating", "count"),
        )
        .sort_values("mean_rating", ascending=False)
    )


# -----------------------------------------------------------------------------
# Read both CSV files directly with pandas (no file bytes / upload required).
# -----------------------------------------------------------------------------
st.title("Drug Insights Dashboard")
st.caption("Drug side effects, potential warnings, and user ratings")

try:
    # These are the exact paths requested for the user's PyCharm project.
    original_df = pd.read_csv(
        r"C:\Users\abdal\PycharmProjects\AI_EDA\EDA_Project_Drugs\DataSet\drugs_side_effects_drugs_com.csv"
    )
    cleaned_df = pd.read_csv(
        r"C:\Users\abdal\PycharmProjects\AI_EDA\EDA_Project_Drugs\DataSet\cleaned_drugs_side_effects_drugs_com.csv"
    )
except FileNotFoundError as error:
    st.error(
        "Could not find one of the CSV files. Check that both filenames and paths are correct. "
        f"Details: {error}"
    )
    st.stop()
except pd.errors.EmptyDataError:
    st.error("One of the CSV files is empty. Re-save the original and cleaned CSV files from the notebook.")
    st.stop()
except pd.errors.ParserError as error:
    st.error(f"Pandas could not parse one of the CSV files: {error}")
    st.stop()

required_columns = {"drug_name", "medical_condition", "side_effects", "rating", "no_of_reviews", "rx_otc"}
for label, data in (("original", original_df), ("cleaned", cleaned_df)):
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        st.error(f"The {label} dataset is missing required columns: {', '.join(sorted(missing_columns))}.")
        st.stop()
    if data.empty:
        st.error(f"The {label} CSV has column names but contains no data rows.")
        st.stop()

# Keep the raw DataFrames unchanged for accurate row/column counts and previews.
# Add numeric helper columns only to internal analysis copies.
original_analysis_df = add_numeric_helpers(original_df)
cleaned_analysis_df = add_numeric_helpers(cleaned_df)

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Dataset description")
st.sidebar.markdown(
    "**Drugs, side effects, and medical conditions**\n\n"
    "Each record contains drug information associated with a medical condition, including side-effect text, "
    "drug class, prescription status, pregnancy category, user rating, and number of reviews.\n\n"
    "**Purpose:** explore recorded medication information and identify descriptions that mention potential warnings. "
    "This dashboard does not determine which medication is appropriate or safest for a patient."
)
st.sidebar.header("Filters")
version = st.sidebar.selectbox("Dataset version", ["Cleaned", "Original"])
if version == "Cleaned":
    base_df = cleaned_analysis_df
    display_base_df = cleaned_df
else:
    base_df = original_analysis_df
    display_base_df = original_df

rx_values = sorted(base_df["rx_otc"].dropna().astype(str).unique().tolist())
rx_options = ["All"] + rx_values
rx_choice = st.sidebar.selectbox("Prescription status (Rx / OTC)", rx_options)
conditions = st.sidebar.multiselect(
    "Medical condition",
    sorted(base_df["medical_condition"].dropna().astype(str).unique().tolist()),
    help="Leave empty to include all conditions.",
)
rating_range = st.sidebar.slider("Rating range", 0.0, 10.0, (0.0, 10.0), step=0.5)
min_reviews = st.sidebar.slider("Minimum number of reviews", 0, 200, 0)
keep_unrated = st.sidebar.checkbox("Keep records without ratings or reviews", value=True)
top_n = st.sidebar.slider("Items shown in charts", 5, 25, 10)

mask = pd.Series(True, index=base_df.index)
if conditions:
    mask &= base_df["medical_condition"].isin(conditions)
if rx_choice != "All":
    mask &= base_df["rx_otc"].astype(str) == rx_choice
rating_ok = base_df["new_rating"].between(*rating_range)
reviews_ok = base_df["new_no_of_reviews"].ge(min_reviews)
if keep_unrated:
    rating_ok |= base_df["new_rating"].isna()
    reviews_ok |= base_df["new_no_of_reviews"].isna()
filtered_df = base_df[mask & rating_ok & reviews_ok]
display_df = display_base_df.loc[filtered_df.index]

if filtered_df.empty:
    st.warning("No records match the selected filters. Try widening the filters in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(filtered_df):,}")
k2.metric("Distinct drugs", f"{filtered_df['drug_name'].nunique():,}")
k3.metric("Medical conditions", f"{filtered_df['medical_condition'].nunique():,}")
rating_mean = filtered_df["new_rating"].mean()
k4.metric("Average recorded rating", f"{rating_mean:.2f}" if pd.notna(rating_mean) else "N/A")
st.caption(
    f"Showing the {version.lower()} dataset: {len(filtered_df):,} of {len(base_df):,} records after filters. "
    "Numeric helper columns are used internally and are not counted in the raw dataset dimensions."
)

# Dataset tabs
tab_data, tab_cond, tab_side, tab_warn, tab_rate = st.tabs(
    ["Data", "Conditions", "Side effects", "Potential warnings", "Ratings"]
)

# -----------------------------------------------------------------------------
# TAB 1 - data preview and quality
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Data preview")
    preview_rows = st.slider("Rows to preview", 5, 100, 10, key="preview_rows")
    st.dataframe(display_df.head(preview_rows), use_container_width=True)
    st.caption(
        "The displayed table uses the columns saved in the selected CSV. "
        "Numeric helper columns are generated only in a separate analysis copy."
    )

    st.subheader("Summary statistics")
    st.dataframe(filtered_df[["new_rating", "new_no_of_reviews"]].describe().T, use_container_width=True)
    with st.expander("Text and categorical column summary"):
        text_columns = display_df.select_dtypes(exclude="number")
        if not text_columns.empty:
            st.dataframe(text_columns.describe().T, use_container_width=True)
        else:
            st.info("There are no text or categorical columns in the filtered data.")

    st.subheader("Original vs cleaned dataset")
    c1, c2 = st.columns(2)
    c1.metric("Original dimensions", f"{original_df.shape[0]:,} rows x {original_df.shape[1]} columns")
    c2.metric("Cleaned dimensions", f"{cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]} columns")

    common_columns = [
        column for column in original_df.columns
        if column in cleaned_df.columns and not column.startswith("new_")
    ]
    if common_columns:
        miss = pd.DataFrame(
            {
                "Original": missing_percent(original_df[common_columns]),
                "Cleaned": missing_percent(cleaned_df[common_columns]),
            }
        )
        miss = miss[(miss["Original"] > 0) | (miss["Cleaned"] > 0)]
        if not miss.empty:
            miss_long = miss.rename_axis("Column").reset_index().melt(
                id_vars="Column", var_name="Dataset", value_name="Missing percentage"
            )
            fig = px.bar(
                miss_long,
                x="Column",
                y="Missing percentage",
                color="Dataset",
                barmode="group",
                color_discrete_map={"Original": "#94a3b8", "Cleaned": TEAL},
                title="Missing values per column (%)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The columns shared by both datasets have no missing values.")
    st.caption("Only columns present in both files are compared. The removed alcohol column is therefore omitted.")
    insight(
        "The cleaned dataset removes records without side-effect descriptions and drops the alcohol column. "
        "Selected categorical gaps were filled with labels such as Unknown or NA, while missing ratings and review counts were retained.",
        "Missing ratings and review counts were not replaced with zero, because zero would imply an actual recorded value."
    )

    st.subheader("Potential outliers (IQR method)")
    st.dataframe(outlier_table(filtered_df).round(2), use_container_width=True)
    o1, o2 = st.columns(2)
    o1.plotly_chart(
        px.box(filtered_df, x="new_rating", title="Recorded rating", color_discrete_sequence=[TEAL]),
        use_container_width=True,
    )
    o2.plotly_chart(
        px.box(
            filtered_df,
            x="new_no_of_reviews",
            title="Number of reviews",
            range_x=[0, 3000],
            color_discrete_sequence=[TEAL],
        ),
        use_container_width=True,
    )
    insight(
        "The IQR method identifies potential outliers in ratings and review counts. Some records have substantially more reviews than others, and these values are retained for investigation rather than removed automatically.",
        "A high number of reviews does not show that a medication is effective or safe."
    )

    st.download_button(
        "Download cleaned dataset (CSV)",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="DataSet/cleaned_drugs_side_effects_drugs_com.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# TAB 2 - Q1: conditions with the most distinct drugs
# -----------------------------------------------------------------------------
with tab_cond:
    st.subheader("Q1: Which medical conditions have the most available drugs?")
    condition_counts = drugs_per_condition(filtered_df)
    top_counts = condition_counts.head(top_n).sort_values()
    fig = px.bar(
        x=top_counts.values,
        y=top_counts.index,
        orientation="h",
        labels={"x": "Distinct drug names", "y": "Medical condition"},
        title=f"Top {len(top_counts)} conditions by number of distinct drugs",
        color_discrete_sequence=[TEAL],
    )
    st.plotly_chart(fig, use_container_width=True)
    top3 = ", ".join(f"<b>{esc(condition)}</b> ({count})" for condition, count in condition_counts.head(3).items())
    insight(
        f"The conditions with the most distinct listed drug names are {top3}.",
        "These counts describe the records in this dataset. They do not explain why more drugs are listed for one condition or how common the condition is in real life."
    )

# -----------------------------------------------------------------------------
# TAB 3 - Q2: side-effect descriptions
# -----------------------------------------------------------------------------
with tab_side:
    st.subheader("Q2: Most common recorded side-effect terms")
    long_data = build_side_effects_long(filtered_df)
    term_counts = side_effect_term_counts(long_data)
    if term_counts.empty:
        st.info("No side-effect descriptions are available under the current filters.")
    else:
        top_terms = term_counts.head(top_n).sort_values()
        fig = px.bar(
            x=top_terms.values,
            y=top_terms.index,
            orientation="h",
            labels={"x": "Records mentioning the extracted term", "y": "Extracted term"},
            title=f"Top {len(top_terms)} most frequently extracted side-effect terms",
            color_discrete_sequence=[TEAL],
        )
        st.plotly_chart(fig, use_container_width=True)
        frequent_terms = ", ".join(
            f"<b>{esc(term)}</b> ({count:,})" for term, count in term_counts.head(4).items()
        )
        insight(
            f"The most frequently extracted terms are {frequent_terms}.",
            "Descriptions are split at commas and semicolons, so some extracted terms may be sentence fragments. These counts are not patient-level incidence rates. This dataset has no gender column, so gender-based comparisons are not possible."
        )

        effects_by_drug = drug_effect_summary(long_data)
        if not effects_by_drug.empty:
            fewest_terms = effects_by_drug.head(top_n).sort_values(ascending=False)
            fig = px.bar(
                x=fewest_terms.values,
                y=fewest_terms.index,
                orientation="h",
                labels={"x": "Distinct extracted terms", "y": "Drug"},
                title=f"{len(fewest_terms)} drugs with the fewest recorded terms",
                color_discrete_sequence=["#d97706"],
            )
            st.plotly_chart(fig, use_container_width=True)
            first_drugs = ", ".join(
                f"<b>{esc(drug)}</b> ({count})" for drug, count in effects_by_drug.head(3).items()
            )
            insight(
                f"The drugs with the fewest distinct extracted terms are {first_drugs}.",
                "Fewer extracted terms do not mean fewer actual side effects. This analysis does not identify the safest medication."
            )
            with st.expander("Inspect original descriptions for these drugs"):
                st.dataframe(
                    filtered_df[filtered_df["drug_name"].isin(effects_by_drug.head(top_n).index)]
                    [["drug_name", "medical_condition", "side_effects"]]
                    .drop_duplicates(),
                    use_container_width=True,
                )

# -----------------------------------------------------------------------------
# TAB 4 - Q3: descriptions containing potential warning terms
# -----------------------------------------------------------------------------
with tab_warn:
    st.subheader("Q3: Which drugs' descriptions mention potential serious warnings?")
    w1, w2 = st.columns([3, 1])
    selected_categories = w1.multiselect(
        "Warning categories",
        list(WARNING_PATTERNS),
        default=list(WARNING_PATTERNS),
    )
    min_descriptions = w2.slider("Minimum distinct descriptions per drug", 1, 10, 2)

    if not selected_categories:
        st.info("Select at least one warning category.")
    else:
        warning_results = warning_summary(filtered_df, selected_categories, min_descriptions)
        if warning_results.empty:
            st.info("No drugs meet the selected minimum number of descriptions. Lower the minimum or widen the filters.")
        else:
            top_warnings = warning_results.head(top_n).sort_values("warning_description_percentage")
            fig = px.bar(
                top_warnings,
                x="warning_description_percentage",
                y=top_warnings.index,
                orientation="h",
                range_x=[0, 100],
                hover_data=["descriptions_with_warning", "descriptions_reviewed"],
                labels={
                    "warning_description_percentage": "Descriptions mentioning selected terms (%)",
                    "y": "Drug",
                },
                title="Drugs whose descriptions most often mention potential warnings",
                color_discrete_sequence=["#b91c1c"],
            )
            st.plotly_chart(fig, use_container_width=True)
            first_drug = warning_results.index[0]
            first_row = warning_results.iloc[0]
            all_description_pct = int((top_warnings["warning_description_percentage"] == 100).sum())
            insight(
                f"{all_description_pct} of the {len(top_warnings)} displayed drugs have selected warning terms in all their reviewed descriptions. "
                f"For example, {esc(first_drug)} has warning terms in {int(first_row['descriptions_with_warning'])} of {int(first_row['descriptions_reviewed'])} descriptions.",
                "This is keyword searching in drug descriptions, not a measure of how often patients experience these effects. It does not mean a drug should be avoided, and the dataset contains no gender column."
            )
            with st.expander("Full warning summary"):
                st.dataframe(warning_results.round(1), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5 - Q4: rating comparisons and rating/review relationship
# -----------------------------------------------------------------------------
with tab_rate:
    st.subheader("Q4: How do recorded user ratings vary across medical conditions?")
    condition_ratings = condition_rating_summary(filtered_df)
    if condition_ratings.empty:
        st.info("No recorded ratings are available under the current filters.")
    else:
        number_conditions = min(top_n, 15, len(condition_ratings))
        top_conditions = condition_ratings.nlargest(number_conditions, "rating_count").index
        rating_rows = rating_data(filtered_df)
        rating_plot_data = rating_rows[rating_rows["medical_condition"].isin(top_conditions)]
        fig = px.box(
            rating_plot_data,
            x="new_rating",
            y="medical_condition",
            labels={"new_rating": "Recorded user rating", "medical_condition": "Medical condition"},
            title=f"Rating distribution for the {len(top_conditions)} conditions with the most ratings",
            color_discrete_sequence=[TEAL],
        )
        st.plotly_chart(fig, use_container_width=True)

        min_ratings = st.slider("Minimum ratings per condition for mean comparison", 1, 50, 5)
        eligible_ratings = condition_ratings[
            condition_ratings["rating_count"] >= min_ratings
        ].sort_values("mean_rating", ascending=False)
        if len(eligible_ratings) >= 2:
            highest = eligible_ratings.iloc[0]
            lowest = eligible_ratings.iloc[-1]
            insight(
                f"Among conditions with at least {min_ratings} recorded ratings, {esc(eligible_ratings.index[0])} has the highest mean rating "
                f"({highest['mean_rating']:.2f}, {int(highest['rating_count'])} ratings), while {esc(eligible_ratings.index[-1])} has the lowest "
                f"({lowest['mean_rating']:.2f}, {int(lowest['rating_count'])} ratings).",
                "Review counts and rating distributions should be considered when comparing conditions. User ratings are opinions, not verified measures of clinical effectiveness or safety."
            )
            t1, t2 = st.columns(2)
            t1.markdown("**Highest mean ratings**")
            t1.dataframe(eligible_ratings.head(5).round(2), use_container_width=True)
            t2.markdown("**Lowest mean ratings**")
            t2.dataframe(eligible_ratings.tail(5).iloc[::-1].round(2), use_container_width=True)
        else:
            st.info("Fewer than two conditions meet this minimum. Lower the minimum to compare their means.")

    st.subheader("Relationship between rating and number of reviews")
    numeric_pairs = filtered_df[["new_no_of_reviews", "new_rating"]].dropna()
    if len(numeric_pairs) >= 2:
        correlation_matrix = numeric_pairs.corr()
        correlation = correlation_matrix.loc["new_rating", "new_no_of_reviews"]
        h1, h2 = st.columns([1, 2])
        with h1:
            st.plotly_chart(
                px.imshow(
                    correlation_matrix,
                    text_auto=".3f",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="Correlation heatmap",
                ),
                use_container_width=True,
            )
        with h2:
            log_scale = st.checkbox("Use logarithmic scale for review counts", value=False)
            fig = px.scatter(
                filtered_df,
                x="new_rating",
                y="new_no_of_reviews",
                opacity=0.6,
                log_y=log_scale,
                hover_data=["drug_name", "medical_condition"],
                labels={"new_rating": "Rating", "new_no_of_reviews": "Number of reviews"},
                title="Rating vs number of reviews",
                color_discrete_sequence=[TEAL],
            )
            st.plotly_chart(fig, use_container_width=True)
        magnitude = abs(correlation)
        strength = "very weak" if magnitude < 0.2 else "weak" if magnitude < 0.4 else "moderate" if magnitude < 0.6 else "strong"
        direction = "negative" if correlation < 0 else "positive"
        insight(
            f"The correlation between rating and number of reviews is {correlation:.3f}, indicating a {strength} {direction} linear relationship. "
            "Most records have relatively few reviews, while a small number have many reviews across different rating levels.",
            "Correlation does not establish cause and effect. Unusually high review counts should be investigated, not automatically removed."
        )
    else:
        st.info("At least two rows with both a rating and review count are required to calculate a correlation.")
