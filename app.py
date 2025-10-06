# ============================================================
# 📊 Open Source Data Analyzer
# Author: Eram Sohab Kandhal
# College: M.H. Saboo Siddik College of Engineering
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
from io import BytesIO
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Open Source Data Analyzer", layout="wide")

# -----------------------------
# Navbar
# -----------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Upload Data", "Data Cleaning", "EDA", "Visualization", "Dashboard", "Download"],
    icons=["house", "cloud-upload", "brush", "search", "bar-chart", "grid", "download"],
    default_index=0,
    orientation="horizontal",
)

# -----------------------------
# Home (Landing Page)
# -----------------------------
if selected == "Home":
    st.markdown(
        """
        <div style="text-align:center; padding:40px;">
            <h1>📊 Open Source Data Analyzer</h1>
            <h3 style="color:gray;">A beginner-friendly tool for cleaning, analyzing & visualizing datasets 🚀</h3>
            <p style="font-size:18px;">Upload → Clean → Explore → Visualize → Dashboard → Download</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⚡ **Automated Cleaning**\n\nFix missing values, duplicates & inconsistencies.")
    with col2:
        st.success("📈 **Smart EDA**\n\nGet dataset insights, summaries & correlations in one click.")
    with col3:
        st.warning("🎨 **Interactive Visuals**\n\nBuild charts & dashboards without coding.")

    st.divider()
    st.markdown("### 🔄 Workflow")
    steps = ["Upload Data", "Clean Data", "Run EDA", "Generate Visualizations", "Dashboard Summary", "Download Report"]
    for i, step in enumerate(steps, 1):
        st.markdown(f"**{i}. {step}** ✅")

    st.divider()
    st.success("👉 Start by uploading your dataset using the **Upload Data** tab.")
    st.markdown("<p style='text-align:center; color:gray;'>💻 Open Source Project - M.H. Saboo Siddik College of Engineering</p>", unsafe_allow_html=True)

# -----------------------------
# -----------------------------
# Upload Data
# -----------------------------
elif selected == "Upload Data":
    st.title("📂 Upload Your Dataset")
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if file:
        try:
            if file.name.endswith("csv"):
                file.seek(0)  # Reset file pointer
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except UnicodeDecodeError:
                    file.seek(0)
                    try:
                        df = pd.read_csv(file, encoding="ISO-8859-1")
                    except Exception:
                        file.seek(0)
                        df = pd.read_csv(file, encoding="latin1")
            else:
                file.seek(0)
                df = pd.read_excel(file)

            st.session_state.df = df
            st.success("✅ Dataset uploaded successfully!")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error uploading file: {e}")



# -----------------------------
# Data Cleaning
# -----------------------------
elif selected == "Data Cleaning":
    st.title("🧹 Data Cleaning")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        st.subheader("🔍 Handle Missing Values")
        missing_option = st.radio("Choose method:", ["None", "Fill with Mean", "Fill with Median", "Fill with Mode"])
        if missing_option != "None":
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if missing_option == "Fill with Mean" and df[col].dtype != "object":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif missing_option == "Fill with Median" and df[col].dtype != "object":
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            st.success(f"✅ Missing values filled using {missing_option}")

        if st.checkbox("Remove Duplicates"):
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]
            st.success(f"✅ Removed {before - after} duplicate rows")

        st.dataframe(df.head())
        st.session_state.df = df

# -----------------------------
# EDA
# -----------------------------
elif selected == "EDA":
    st.title("🔎 Exploratory Data Analysis (EDA)")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        import plotly.express as px
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        df = st.session_state.df.copy()

        # -----------------------------
        # Top metrics
        # -----------------------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing (%)", round(df.isnull().mean().mean() * 100, 2))
        col4.metric("Duplicates", int(df.duplicated().sum()))

        st.markdown("---")

        # -----------------------------
        # Data types + summary
        # -----------------------------
        left, right = st.columns([1, 2])

        with left:
            st.subheader("📌 Data Types")
            dtypes_df = df.dtypes.apply(lambda x: x.name).to_frame("dtype")
            st.dataframe(dtypes_df)

            st.markdown("### 🔢 Categorical quick view")
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                for c in cat_cols[:3]:
                    top = df[c].value_counts(dropna=False).head(6)
                    st.write(f"**{c}** (top values)")
                    st.dataframe(top.to_frame(name="count"))
            else:
                st.info("No categorical columns detected.")

        with right:
            st.subheader("📊 Numerical Summary")
            num = df.select_dtypes(include=np.number)
            if not num.empty:
                desc_styled = num.describe().T.style.background_gradient(cmap="Blues")
                st.dataframe(desc_styled)
            else:
                st.info("No numeric columns to summarize.")

        st.markdown("---")

        # -----------------------------
        # Missing Data Visuals
        # -----------------------------
        st.subheader("❗ Missing Data Overview")
        mcol1, mcol2 = st.columns(2)

        with mcol1:
            missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
            if missing_pct.max() == 0:
                st.success("No missing values found in the dataset ✅")
            else:
                miss_df = missing_pct.reset_index()
                miss_df.columns = ["column", "missing_pct"]
                fig_miss = px.bar(
                    miss_df, x="missing_pct", y="column", orientation="h",
                    title="Missing % by Column", text="missing_pct",
                    width=700, height=400
                )
                fig_miss.update_layout(xaxis_title="Missing (%)", yaxis_title="")
                st.plotly_chart(fig_miss, use_container_width=True)

        with mcol2:
            max_rows_for_heatmap = 200
            isnull_int = df.isnull().astype(int)
            if isnull_int.shape[0] > max_rows_for_heatmap:
                heat_df = isnull_int.sample(n=max_rows_for_heatmap, random_state=42)
                st.caption(f"Showing a random sample of {max_rows_for_heatmap} rows for the heatmap.")
            else:
                heat_df = isnull_int
            fig, ax = plt.subplots(figsize=(10, max(3, len(heat_df.columns) * 0.25)))
            sns.heatmap(heat_df.T, cbar=True, cmap="YlGnBu", linewidths=0.2, ax=ax)
            ax.set_xlabel("Sample rows")
            ax.set_ylabel("Columns (transposed)")
            ax.set_title("Missingness heatmap")
            st.pyplot(fig)

        st.markdown("---")

        # -----------------------------
        # Correlation Heatmap
        # -----------------------------
        st.subheader("🔗 Correlation Heatmap (numeric only)")
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            st.info("Need at least two numeric columns for correlation heatmap.")
            corr = pd.DataFrame()  # define empty to avoid NameError later
        else:
            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(min(14, 1.1 * corr.shape[0]), min(10, 1.1 * corr.shape[1])))
            sns.heatmap(
                corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 9}
            )
            ax.set_title("Correlation matrix (upper triangle masked)")
            st.pyplot(fig)

            corr_pairs = corr.abs().where(~mask).stack().reset_index()
            corr_pairs.columns = ["feature_1", "feature_2", "corr_abs"]
            corr_pairs = corr_pairs.sort_values(by="corr_abs", ascending=False)

            st.subheader("🔎 Top strong correlations")
            if corr_pairs.empty:
                st.info("No correlation pairs found.")
            else:
                st.dataframe(corr_pairs.head(10).style.format({"corr_abs": "{:.2f}"}))

        st.markdown("---")

        # -----------------------------
        # Distribution & Boxplots
        # -----------------------------
        st.subheader("📈 Distribution & Boxplots (first 4 numeric columns)")
        if numeric_df.empty:
            st.info("No numeric columns for distribution plots.")
        else:
            plot_cols = numeric_df.columns[:4].tolist()
            cols = st.columns(2)
            for i, col_name in enumerate(plot_cols):
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                sns.histplot(numeric_df[col_name].dropna(), kde=True, ax=axes[0])
                axes[0].set_title(f"Distribution: {col_name}")
                sns.boxplot(x=numeric_df[col_name].dropna(), ax=axes[1])
                axes[1].set_title(f"Boxplot: {col_name}")
                target_col = cols[i % 2]
                target_col.pyplot(fig)

        st.markdown("---")

        # -----------------------------
        # Scatter Matrix
        # -----------------------------
        st.subheader("🔬 Scatter Matrix (interactive)")
        if numeric_df.shape[1] >= 2:
            sm_cols = numeric_df.columns[:6]
            fig_sm = px.scatter_matrix(
                numeric_df[sm_cols], dimensions=sm_cols,
                title="Scatter matrix (first up to 6 numeric columns)", height=700
            )
            fig_sm.update_traces(diagonal_visible=False)
            st.plotly_chart(fig_sm, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns to show scatter matrix.")

        st.markdown("---")

        # -----------------------------
        # Custom Visualizations
        # -----------------------------
        with st.expander("🧰 Custom Visualizations"):
            cols = df.columns.tolist()
            cx, cy = st.columns(2)
            with cx:
                x_axis = st.selectbox("Choose X-axis", cols, index=0)
            with cy:
                y_axis = st.selectbox("Choose Y-axis (optional)", [None] + cols, index=0)

            chart_type = st.selectbox("Chart Type", [
                "Auto (smart)", "Bar", "Line", "Scatter", "Histogram", "Box", "Violin"
            ])

            if st.button("Generate Custom Chart"):
                try:
                    if chart_type == "Auto (smart)":
                        if pd.api.types.is_numeric_dtype(df[x_axis]) and (
                            y_axis is None or pd.api.types.is_numeric_dtype(df[y_axis])
                        ):
                            fig = px.histogram(df, x=x_axis, nbins=30, title=f"Histogram: {x_axis}")
                        else:
                            vc = df[x_axis].value_counts().reset_index()
                            vc.columns = [x_axis, "count"]
                            fig = px.bar(vc, x=x_axis, y="count", title=f"Bar: {x_axis}")
                    elif chart_type == "Bar":
                        if y_axis is None:
                            vc = df[x_axis].value_counts().reset_index()
                            vc.columns = [x_axis, "count"]
                            fig = px.bar(vc, x=x_axis, y="count", title=f"Bar: {x_axis}")
                        else:
                            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                    elif chart_type == "Line":
                        fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter: {y_axis} vs {x_axis}")
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, nbins=30, title=f"Histogram: {x_axis}")
                    elif chart_type == "Box":
                        fig = px.box(df, x=x_axis, y=y_axis, title=f"Boxplot: {y_axis} by {x_axis}")
                    elif chart_type == "Violin":
                        fig = px.violin(df, x=x_axis, y=y_axis, box=True, points="all",
                                        title=f"Violin: {y_axis} by {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating chart: {e}")

        st.markdown("---")

        # -----------------------------
        # Key Insights
        # -----------------------------
        st.subheader("📝 Key Insights (Auto)")
        rows = df.shape[0]
        cols_n = df.shape[1]
        dup = int(df.duplicated().sum())
        miss_pct = round(df.isnull().mean().mean() * 100, 2)

        st.write(f"- Total rows: **{rows}**")
        st.write(f"- Total columns: **{cols_n}**")
        st.write(f"- Duplicate rows: **{dup}**")
        st.write(f"- Average missingness across columns: **{miss_pct}%**")

        if not numeric_df.empty and not corr.empty:
            strong = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
            strong = strong[(strong < 1) & (strong > 0.5)].head(5)
            if not strong.empty:
                st.write("- Strong numeric correlations (abs > 0.5):")
                for idx, val in strong.items():
                    st.write(f"  - {idx}: **{val:.2f}**")
            else:
                st.write("- No strong numeric correlations (abs > 0.5) found.")

# -----------------------------
# Visualization (Smooth & Interactive)
# -----------------------------
elif selected == "Visualization":
    st.title("🎨 Interactive Data Visualization Studio")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.df.copy()

        st.write("Select variables and chart type to visualize your data interactively 📊")

        # --- Column Selection ---
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis", df.columns)
        with col2:
            y_axis = st.selectbox("Select Y-axis (optional)", ["None"] + list(df.columns))

        # --- Chart Type ---
        chart_type = st.selectbox(
            "Choose Chart Type",
            ["Auto", "Bar", "Line", "Scatter", "Pie", "Histogram", "Boxplot", "Violin", "Heatmap"]
        )

        import plotly.express as px

        try:
            # Sample large datasets for performance
            df_plot = df.sample(5000, random_state=42) if df.shape[0] > 5000 else df

            fig = None  # initialize figure

            # ----- Auto Chart Selection -----
            if chart_type == "Auto":
                if pd.api.types.is_numeric_dtype(df_plot[x_axis]) and (y_axis == "None" or pd.api.types.is_numeric_dtype(df_plot[y_axis])):
                    chart_type = "Histogram"
                else:
                    chart_type = "Bar"

            # ----- Plotly Charts -----
            if chart_type == "Bar":
                if y_axis == "None":
                    vc = df_plot[x_axis].value_counts().reset_index()
                    vc.columns = [x_axis, "Count"]
                    fig = px.bar(vc, x=x_axis, y="Count", color="Count", title=f"Bar Chart of {x_axis}")
                else:
                    fig = px.bar(df_plot, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}")

            elif chart_type == "Line" and y_axis != "None":
                fig = px.line(df_plot, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")

            elif chart_type == "Scatter" and y_axis != "None":
                fig = px.scatter(df_plot, x=x_axis, y=y_axis, color=x_axis, title=f"Scatter: {y_axis} vs {x_axis}")

            elif chart_type == "Pie":
                vc = df_plot[x_axis].value_counts().reset_index()
                vc.columns = [x_axis, "Count"]
                fig = px.pie(vc, names=x_axis, values="Count", title=f"Pie Chart of {x_axis}")

            elif chart_type == "Histogram":
                fig = px.histogram(df_plot, x=x_axis, nbins=30, title=f"Histogram of {x_axis}")

            elif chart_type == "Boxplot" and y_axis != "None":
                fig = px.box(df_plot, x=x_axis, y=y_axis, color=x_axis, title=f"Boxplot: {y_axis} by {x_axis}")

            elif chart_type == "Violin" and y_axis != "None":
                fig = px.violin(df_plot, x=x_axis, y=y_axis, box=True, points="all", color=x_axis, title=f"Violin: {y_axis} by {x_axis}")

            elif chart_type == "Heatmap":
                numeric_df = df_plot.select_dtypes(include="number")
                if not numeric_df.empty:
                    fig = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
                else:
                    st.info("⚠️ No numeric columns available for heatmap.")

            # ----- Show Figure -----
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Visualization Error: {e}")



# -----------------------------
# Dashboard (Modern Layout)
# -----------------------------
elif selected == "Dashboard":
    st.title("📊 Data Overview Dashboard")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and clean a dataset first.")
    else:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=np.number)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing (%)", round(df.isnull().mean().mean() * 100, 2))
        col4.metric("Duplicate Rows", df.duplicated().sum())

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📦 Top Category Distribution")
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) > 0:
                col = cat_cols[0]
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc.head(8), x=col, y="count", color="count", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available.")
        with col2:
            st.subheader("🔗 Correlation Heatmap")
            if not numeric_df.empty:
                fig = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for correlation.")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Trend Visualization")
            if len(numeric_df.columns) >= 2:
                fig = px.line(df, x=numeric_df.columns[0], y=numeric_df.columns[1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for line chart.")
        with col2:
            st.subheader("📊 Numeric Distribution")
            if len(numeric_df.columns) > 0:
                fig = px.histogram(df, x=numeric_df.columns[0], nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for histogram.")

        st.markdown("---")
        st.subheader("🧠 Key Insights")
        st.write(f"- Dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.write(f"- Missing values: **{round(df.isnull().mean().mean() * 100, 2)}%**.")
        st.write(f"- Duplicates found: **{df.duplicated().sum()} rows**.")
        if not numeric_df.empty:
            top_corr = numeric_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates().head(3)
            st.write("🔗 Strongest correlations:")
            st.dataframe(top_corr)

# -----------------------------
# Download (Professional EDA PDF Report)
# -----------------------------
elif selected == "Download":
    st.title("📄 Generate Professional EDA Report (PDF)")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and clean a dataset first.")
    else:
        import tempfile
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from datetime import datetime

        df = st.session_state.df

        def generate_styled_pdf(df):
            pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                    rightMargin=50, leftMargin=50,
                                    topMargin=50, bottomMargin=50)

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=22, alignment=TA_CENTER, spaceAfter=20)
            subtitle_style = ParagraphStyle('Subtitle', fontName='Helvetica-Oblique', fontSize=14, alignment=TA_CENTER, textColor=colors.HexColor("#0077b6"))
            heading_style = ParagraphStyle('Heading', fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor("#023047"), spaceBefore=15, spaceAfter=10)
            body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=11, alignment=TA_JUSTIFY, leading=16)
            highlight_style = ParagraphStyle('Highlight', fontName='Helvetica-Bold', fontSize=12, textColor=colors.red)

            elements = []

            # ---------------- COVER PAGE ----------------
            elements.append(Paragraph("📊 Open Source Data Analyzer", title_style))
            elements.append(Paragraph("Automated EDA & Visualization Report", subtitle_style))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>Team:</b> Eram Kandhal, Sarah Madre, Sheza Momin, Prathamesh Patil ", body_style))
            elements.append(Paragraph("<b>College:</b> M.H. Saboo Siddik College of Engineering", body_style))
            elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}", body_style))
            elements.append(Spacer(1, 40))
            elements.append(Paragraph(
                "This report provides a comprehensive overview of your dataset, including summary statistics, data distributions, "
                "missing value patterns, and key correlations between numerical features. All analyses were generated automatically using "
                "the <b>Open Source Data Analyzer</b> tool.",
                body_style))
            elements.append(PageBreak())

            # ---------------- DATASET OVERVIEW ----------------
            elements.append(Paragraph("📋 Dataset Overview", heading_style))
            overview_data = [
                ["Total Rows", df.shape[0]],
                ["Total Columns", df.shape[1]],
                ["Missing Values (%)", round(df.isnull().mean().mean() * 100, 2)],
                ["Duplicate Rows", df.duplicated().sum()]
            ]
            table = Table(overview_data, colWidths=[200, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))

            # ---------------- DATA TYPES ----------------
            elements.append(Paragraph("📌 Data Types", heading_style))
            for col, dtype in df.dtypes.items():
                elements.append(Paragraph(f"<b>{col}</b>: {dtype}", body_style))
            elements.append(Spacer(1, 20))

            # ---------------- VISUALS SECTION ----------------
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")

            numeric_df = df.select_dtypes(include=np.number)

            # Missing Value Heatmap
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(df.isnull(), cbar=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
            ax.set_title("Missing Values Heatmap", fontsize=14)
            img1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig.savefig(img1, bbox_inches="tight", dpi=150)
            plt.close(fig)
            elements.append(Paragraph("❗ Missing Values Heatmap", heading_style))
            elements.append(Image(img1, width=400, height=200))

            # Correlation Heatmap
            if not numeric_df.empty:
             fig, ax = plt.subplots(figsize=(10, 7))
             sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax, annot_kws={"size":9})
             ax.set_title("Correlation Heatmap", fontsize=14)
             img2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
             fig.savefig(img2, bbox_inches="tight", dpi=150)
             plt.close(fig)
             elements.append(Paragraph("🔗 Correlation Heatmap", heading_style))
             elements.append(Image(img2, width=400, height=200))

                # Distribution plots
            for col in numeric_df.columns[:3]:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, color="#219ebc", ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    img3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    fig.savefig(img3, bbox_inches="tight")
                    plt.close(fig)
                    elements.append(Paragraph(f"📈 Distribution of {col}", heading_style))
                    elements.append(Image(img3, width=400, height=200))
                    elements.append(Spacer(1, 15))

            # ---------------- INSIGHTS ----------------
            elements.append(Paragraph("🧠 Key Insights", heading_style))
            elements.append(Paragraph(f"- Dataset has <b>{df.shape[0]}</b> rows and <b>{df.shape[1]}</b> columns.", body_style))
            elements.append(Paragraph(f"- Missing values: <b>{round(df.isnull().mean().mean() * 100, 2)}%</b>.", body_style))
            elements.append(Paragraph(f"- Duplicate rows: <b>{df.duplicated().sum()}</b>.", body_style))

            if not numeric_df.empty:
                top_corr = numeric_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates().head(3)
                elements.append(Paragraph("<b>Strongest correlations:</b>", highlight_style))
                for idx, val in top_corr.items():
                    elements.append(Paragraph(f"{idx}: <b>{val:.2f}</b>", body_style))

            # Build PDF
            doc.build(elements)
            return pdf_path

        if st.button("📄 Generate Professional Report"):
            pdf_path = generate_styled_pdf(df)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download Professional EDA Report (PDF)", f, file_name="EDA_Report.pdf", mime="application/pdf")
            st.success("✅ Beautiful EDA Report Generated Successfully!")
