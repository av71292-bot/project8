import difflib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Job Recommendation & Analytics Dashboard",
    page_icon="💼",
    layout="wide",
)

# Custom CSS for Vibrant & Colorful UI Design
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff2222 0%, #ff4b4b 100%);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e2f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header with Vibrant Gradient Styling
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2.5rem;">💼 Job Recommendation & Analytics Dashboard</h1>
        <p style="font-size: 1.2rem; margin-top: 10px; opacity: 0.9;">Discover your ideal career opportunities with smart matching and market insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# Caching and Loading Data/Matrices
@st.cache_resource
def load_assets():
  try:
    h_df = joblib.load("hourly_df.joblib")
    h_mat = joblib.load("hourly_matrix.joblib")
    f_df = joblib.load("fixed_df.joblib")
    f_mat = joblib.load("fixed_matrix.joblib")
    return h_df, h_mat, f_df, f_mat
  except Exception:
    return None, None, None, None


def get_work_mode(row):
  text = str(row.get("title", "")).lower()
  if "link" in row and pd.notna(row["link"]):
    text += " " + str(row["link"]).lower()
  if "hybrid" in text:
    return "Hybrid"
  elif any(
      k in text
      for k in [
          "remote",
          "work from home",
          "wfh",
          "work-from-home",
          "telecommute",
          "anywhere",
      ]
  ):
    return "Remote"
  else:
    return "Onsite / Office"


hourly_df, hourly_matrix, fixed_df, fixed_matrix = load_assets()

# ---------------- SIDEBAR CONFIGURATION ----------------
st.sidebar.markdown(
    """
    <div style="text-align: left; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white;">
        <h2 style="margin:0; font-size: 1.5rem;">🧭 Menu & Info</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

sidebar_option = st.sidebar.radio(
    "Navigation",
    ["Job Recommendation System", "Market Overview & Quick Statistics"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.info(
    "This application analyzes job market trends, identifies emerging job"
    " categories, and provides intelligent job recommendations using advanced"
    " data science techniques."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developed By")
st.sidebar.success("Ankita Vaghela\n\nData Science & AI Practitioner")

# Main Application Logic
if hourly_df is None or fixed_df is None:
  st.error(
      "⚠️ Model files (.joblib) not found! Please make sure they are in the"
      " same directory."
  )
else:
  # ---------------- TAB 1: RECOMMENDATION SYSTEM ----------------
  if sidebar_option == "Job Recommendation System":
    st.markdown("### 🔍 Smart Job Recommendation Engine")
    st.markdown(
        "Enter a job title below to find the most relevant career matches based"
        " on your profile preference."
    )

    # Job Category Selection
    job_type = st.radio(
        "Select Job Category",
        ["Hourly Jobs", "Fixed-Price Jobs"],
        horizontal=True,
    )

    if job_type == "Hourly Jobs":
      selected_df = hourly_df
      selected_matrix = hourly_matrix
    else:
      selected_df = fixed_df
      selected_matrix = fixed_matrix

    # Work Mode Filter (Remote, Hybrid, Onsite / Office)
    work_mode_filter = st.radio(
        "Select Work Mode",
        ["All", "Remote", "Hybrid", "Onsite / Office"],
        horizontal=True,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
      job_title_input = st.text_input(
          "Enter a Job Title (e.g., Full Stack Developer, Python Developer):",
          "Full Stack Developer",
      )

    with col2:
      top_n = st.slider(
          "Top Results", min_value=3, max_value=15, value=5
      )

    if st.button("Get Recommendations"):
      with st.spinner("Analyzing job trends and finding best matches..."):
        try:
          # 1. Exact match check
          match_idx = selected_df[
              selected_df["title"].str.lower() == job_title_input.lower()
          ].index

          # 2. Fuzzy/Smart matching if exact match not found
          if len(match_idx) == 0:
            all_titles = selected_df["title"].dropna().unique().tolist()
            close_matches = difflib.get_close_matches(
                job_title_input, all_titles, n=1, cutoff=0.3
            )

            if close_matches:
              matched_title = close_matches[0]
              st.info(
                  f'ℹ️ Exact match for "{job_title_input}" not found. Showing'
                  f' results based on the closest match: "{matched_title}"'
              )
              match_idx = selected_df[
                  selected_df["title"].str.lower() == matched_title.lower()
              ].index

          if len(match_idx) == 0:
            st.warning(
                "⚠️ No matching jobs found. Please choose from the popular"
                " titles below or check your spelling."
            )
            sample_titles = selected_df["title"].dropna().unique()[:10]
            for t in sample_titles:
              st.text(f"- {t}")
          else:
            idx = match_idx[0]
            actual_matrix_idx = selected_df.index.get_loc(idx)

            sim_scores = list(enumerate(selected_matrix[actual_matrix_idx]))
            sim_scores = sorted(
                sim_scores, key=lambda x: x[1], reverse=True
            )
            sim_scores = sim_scores[1 : top_n + 25]

            job_indices = [i[0] for i in sim_scores]

            # Category-wise columns selection including Rate/Budget, Work Mode, and Link
            if job_type == "Hourly Jobs":
              cols_to_show = [
                  "title",
                  "country",
                  "Work_Mode",
                  "avg_hourly_rate",
                  "link",
              ]
            else:
              cols_to_show = [
                  "title",
                  "country",
                  "Work_Mode",
                  "budget",
                  "link",
              ]

            raw_recs = selected_df.iloc[job_indices].copy()
            raw_recs["Work_Mode"] = raw_recs.apply(get_work_mode, axis=1)

            available_cols = [c for c in cols_to_show if c in raw_recs.columns]
            recommendations = raw_recs[available_cols].drop_duplicates(
                subset="title"
            )

            if work_mode_filter != "All":
              recommendations = recommendations[
                  recommendations["Work_Mode"] == work_mode_filter
              ]

            recommendations = recommendations.head(top_n)

            # Make Table Look Better (1-based index)
            recommendations.index = np.arange(1, len(recommendations) + 1)

            actual_count = len(recommendations)
            if actual_count == 0:
              st.info(
                  f'ℹ️ No "{work_mode_filter}" jobs found for this match.'
                  ' Try selecting "All" for Work Mode.'
              )
            else:
              mode_label = (
                  ""
                  if work_mode_filter == "All"
                  else f" ({work_mode_filter})"
              )
              st.success(
                  f"✅ Showing top {actual_count} recommended job"
                  f" opportunities{mode_label} for you:"
              )

              col_cfg = {}
              if "link" in recommendations.columns:
                col_cfg["link"] = st.column_config.LinkColumn(
                    "Job Link", help="Click to open job posting"
                )
              if "avg_hourly_rate" in recommendations.columns:
                col_cfg["avg_hourly_rate"] = st.column_config.NumberColumn(
                    "Hourly Rate", format="%.2f"
                )
              if "budget" in recommendations.columns:
                col_cfg["budget"] = st.column_config.NumberColumn(
                    "Salary", format="%.2f"
                )

              st.dataframe(
                  recommendations,
                  use_container_width=True,
                  column_config=col_cfg,
              )

        except Exception as e:
          st.error(f"An error occurred during processing: {e}")

  # ---------------- TAB 2: QUICK STATISTICS ----------------
  elif sidebar_option == "Market Overview & Quick Statistics":
    st.markdown("### 📊 Dataset Overview & Quick Statistics")
    st.markdown(
        "Here is a comprehensive snapshot of the job market data loaded into"
        " the system:"
    )

    # 4 Metric Cards for Stronger Impact
    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="Hourly Jobs", value=f"{len(hourly_df):,}")
    with col2:
      st.metric(label="Fixed Jobs", value=f"{len(fixed_df):,}")
    with col3:
      st.metric(label="Countries", value=hourly_df["country"].nunique())
    with col4:
      st.metric(
          label="Unique Job Titles", value=hourly_df["title"].nunique()
      )

    st.markdown("---")
    st.subheader("✨ Sample Available Job Titles")

    # Clean DataFrame for sample titles with sliced long names
    sample_df = pd.DataFrame({
        "Sample Job Titles": (
            hourly_df["title"]
            .dropna()
            .drop_duplicates()
            .str.slice(0, 60)
            .head(10)
            .values
        )
    })
    sample_df.index = np.arange(1, len(sample_df) + 1)
    st.dataframe(sample_df, use_container_width=True)

# Footer Professional
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #555; font-weight: 500;">Job Market'
    " Analysis & Recommendation System | Data Science Portfolio Project</p>",
    unsafe_allow_html=True,
)