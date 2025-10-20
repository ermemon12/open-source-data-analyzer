import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
from io import BytesIO
from datetime import datetime
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Open Source Data Analyzer", layout="wide")

from streamlit_option_menu import option_menu
import streamlit as st



# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Open Source Data Analyzer", layout="wide")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; padding:10px 0 15px 0;">
        <h1 style="color:#0077b6; margin-bottom:5px;">🧠📊 Open Source Data Analyzer</h1>
        <h3 style="color:#023047; margin-top:0;">A beginner-friendly tool for cleaning, analyzing & visualizing datasets</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom:-10px;'></div>", unsafe_allow_html=True)

# -----------------------------
# Navbar
# -----------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Upload Data", "Data Cleaning", "EDA", "Visualization", "Dashboard", "Download"],
    icons=["house", "cloud-upload", "brush", "search", "bar-chart", "grid", "download"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f8ff"},
        "icon": {"color": "#0077b6", "font-size": "20px"},
        "nav-link": {
            "font-size": "17px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#caf0f8",
        },
        "nav-link-selected": {
            "background-color": "#caf0f8",
            "color": "#0077b6",
            "font-weight": "bold",
        },
    },
)

# -----------------------------
# Home Section
# -----------------------------
if selected == "Home":
    st.markdown(
        """
        <div style="text-align:center; padding:25px 0 10px 0;">
            <h2 style="color:#0077b6;">Welcome to the Open Source Data Analyzer</h2>
            <p style="font-size:18px; color:#023047;">Easily clean, explore, visualize, and download insights from your data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ✨ Key Features")

    col1, col2, col3 = st.columns(3)
    box_style = """
        background-color: #ade8f4;
        padding: 25px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        font-size: 16px;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    """
    with col1:
        st.markdown(f'<div style="{box_style}">⚡ <b>Automated Cleaning</b><br><br>Fix missing values, duplicates & inconsistencies.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="{box_style}">📈 <b>Smart EDA</b><br><br>Get dataset insights, summaries & correlations in one click.</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="{box_style}">🎨 <b>Interactive Visuals</b><br><br>Build charts & dashboards without coding.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔄 Workflow")
    st.graphviz_chart('''
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", color="#0077b6", fillcolor="#ade8f4", fontname="Helvetica", fontsize=12];
            edge [color="#0077b6", penwidth=2];

            Upload_Data -> Clean_Data -> Run_EDA -> Generate_Visualizations -> Dashboard_Summary -> Download_Report;
        }
    ''')


    st.divider()
    st.success("  Start by uploading your dataset using the **Upload Data** tab.")
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
        import plotly.express as px

        st.write("All visualizations are interactive. Select columns to update charts dynamically 📊")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("Visualizations Dashboard")
        col1, col2 = st.columns(2)

        # ---------------- Column 1 ----------------
        with col1:
            if categorical_cols:
                st.markdown("### Bar Chart")
                bar_col = st.selectbox("Select Column for Bar Chart", categorical_cols, key="bar")
                vc = df[bar_col].value_counts().reset_index()
                vc.columns = [bar_col, "Count"]
                fig_bar = px.bar(vc, x=bar_col, y="Count", color="Count", title=f"Bar Chart of {bar_col}")
                st.plotly_chart(fig_bar, use_container_width=True)
                # Save selection
                st.session_state['viz_bar'] = bar_col

            if len(numeric_cols) > 1:
                st.markdown("### Line Chart")
                x_line = st.selectbox("X-axis for Line Chart", numeric_cols, key="line_x")
                y_line = st.selectbox("Y-axis for Line Chart", numeric_cols, key="line_y")
                fig_line = px.line(df, x=x_line, y=y_line, title=f"{y_line} vs {x_line}")
                st.plotly_chart(fig_line, use_container_width=True)
                st.session_state['viz_line'] = (x_line, y_line)

            if categorical_cols:
                st.markdown("### Pie Chart")
                pie_col = st.selectbox("Select Column for Pie Chart", categorical_cols, key="pie")
                vc = df[pie_col].value_counts().reset_index()
                vc.columns = [pie_col, "Count"]
                fig_pie = px.pie(vc, names=pie_col, values="Count", title=f"Pie Chart of {pie_col}")
                st.plotly_chart(fig_pie, use_container_width=True)
                st.session_state['viz_pie'] = pie_col
            
            if len(numeric_cols) > 0:
                st.markdown("### Histogram")
                hist_col = st.selectbox("Select Column for Histogram", numeric_cols, key="hist")
                fig_hist = px.histogram(df, x=hist_col, nbins=30, title=f"Histogram of {hist_col}")
                st.plotly_chart(fig_hist, use_container_width=True)
                # Save selection
                st.session_state['viz_hist'] = hist_col


        # ---------------- Column 2 ----------------
        with col2:
            if len(numeric_cols) > 1:
                st.markdown("### Scatter Chart")
                x_scatter = st.selectbox("X-axis for Scatter", numeric_cols, key="scatter_x")
                y_scatter = st.selectbox("Y-axis for Scatter", numeric_cols, key="scatter_y")
                fig_scatter = px.scatter(df, x=x_scatter, y=y_scatter, color=x_scatter, title=f"Scatter: {y_scatter} vs {x_scatter}")
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.session_state['viz_scatter'] = (x_scatter, y_scatter)

            if numeric_cols:
                st.markdown("### Box Plot")
                box_col = st.selectbox("Select Column for Box Plot", numeric_cols, key="box")
                fig_box = px.box(df, y=box_col, title=f"Box Plot of {box_col}")
                st.plotly_chart(fig_box, use_container_width=True)
                st.session_state['viz_box'] = box_col

            if numeric_cols and categorical_cols:
                st.markdown("### Violin Plot")
                v_num = st.selectbox("Select Numeric Column for Violin", numeric_cols, key="violin_num")
                v_cat = st.selectbox("Select Category Column for Violin", categorical_cols, key="violin_cat")
                fig_violin = px.violin(df, x=v_cat, y=v_num, box=True, points="all", color=v_cat, title=f"Violin Plot: {v_num} by {v_cat}")
                st.plotly_chart(fig_violin, use_container_width=True)
                st.session_state['viz_violin'] = (v_num, v_cat)

            if len(numeric_cols) > 1:
                st.markdown("### Correlation Heatmap")
                corr_df = df[numeric_cols].corr()
                fig_heat = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)
                st.session_state['viz_heatmap'] = True


# -----------------------------
# Dashboard (Saved Visualizations)
# -----------------------------
elif selected == "Dashboard":
    st.title("📊 Your Dashboard (Saved Visualizations)")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and clean a dataset first.")
    else:
        df = st.session_state.df.copy()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("📌 Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing (%)", round(df.isnull().mean().mean() * 100, 2))
        col4.metric("Duplicate Rows", df.duplicated().sum())

        st.markdown("---")
        st.subheader("🎨 Saved Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            # Bar Chart
            if 'viz_bar' in st.session_state:
                bar_col = st.session_state['viz_bar']
                vc = df[bar_col].value_counts().reset_index()
                vc.columns = [bar_col, "Count"]
                fig_bar = px.bar(vc, x=bar_col, y="Count", color="Count", title=f"Bar Chart of {bar_col}")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Line Chart
            if 'viz_line' in st.session_state:
                x_line, y_line = st.session_state['viz_line']
                fig_line = px.line(df, x=x_line, y=y_line, title=f"{y_line} vs {x_line}")
                st.plotly_chart(fig_line, use_container_width=True)

            # Pie Chart
            if 'viz_pie' in st.session_state:
                pie_col = st.session_state['viz_pie']
                vc = df[pie_col].value_counts().reset_index()
                vc.columns = [pie_col, "Count"]
                fig_pie = px.pie(vc, names=pie_col, values="Count", title=f"Pie Chart of {pie_col}")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # Histogram
            if 'viz_hist' in st.session_state:
                hist_col = st.session_state['viz_hist']
                fig_hist = px.histogram(df, x=hist_col, nbins=30, title=f"Histogram of {hist_col}")
                st.plotly_chart(fig_hist, use_container_width=True)



        with col2:
            # Scatter Chart
            if 'viz_scatter' in st.session_state:
                x_scatter, y_scatter = st.session_state['viz_scatter']
                fig_scatter = px.scatter(df, x=x_scatter, y=y_scatter, color=x_scatter, title=f"Scatter: {y_scatter} vs {x_scatter}")
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Box Plot
            if 'viz_box' in st.session_state:
                box_col = st.session_state['viz_box']
                fig_box = px.box(df, y=box_col, title=f"Box Plot of {box_col}")
                st.plotly_chart(fig_box, use_container_width=True)

            # Violin Plot
            if 'viz_violin' in st.session_state:
                v_num, v_cat = st.session_state['viz_violin']
                fig_violin = px.violin(df, x=v_cat, y=v_num, box=True, points="all", color=v_cat, title=f"Violin Plot: {v_num} by {v_cat}")
                st.plotly_chart(fig_violin, use_container_width=True)

            # Heatmap
            if 'viz_heatmap' in st.session_state and len(numeric_cols) > 1:
                corr_df = df[numeric_cols].corr()
                fig_heat = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)


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
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from datetime import datetime
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.io as pio

        df = st.session_state.df
        numeric_df = df.select_dtypes(include=np.number)

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
            elements.append(Paragraph("<b>Team:</b> Eram Kandhal, Sarah Madre, Sheza Momin, Prathamesh Patil", body_style))
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
            sns.set_theme(style="whitegrid")

            # Missing Value Heatmap
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(df.isnull(), cbar=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
                ax.set_title("Missing Values Heatmap", fontsize=14)
                img1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                fig.savefig(img1, bbox_inches="tight", dpi=150)
                plt.close(fig)
                elements.append(Paragraph("❗ Missing Values Heatmap", heading_style))
                elements.append(Image(img1, width=400, height=200))
            except Exception as e:
                elements.append(Paragraph(f"⚠️ Could not generate missing values heatmap: {e}", body_style))

            # Correlation Heatmap
            if not numeric_df.empty:
                try:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax, annot_kws={"size":9})
                    ax.set_title("Correlation Heatmap", fontsize=14)
                    img2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    fig.savefig(img2, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    elements.append(Paragraph("🔗 Correlation Heatmap", heading_style))
                    elements.append(Image(img2, width=400, height=200))
                except Exception as e:
                    elements.append(Paragraph(f"⚠️ Could not generate correlation heatmap: {e}", body_style))

            # Distribution plots (first 3 numeric columns)
            for col in numeric_df.columns[:3]:
                try:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, color="#219ebc", ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    img3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    fig.savefig(img3, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    elements.append(Paragraph(f"📈 Distribution of {col}", heading_style))
                    elements.append(Image(img3, width=400, height=200))
                    elements.append(Spacer(1, 15))
                except:
                    continue

            # ---------------- DASHBOARD SNAPSHOT ----------------
            elements.append(PageBreak())
            elements.append(Paragraph("📊 Dashboard Snapshot", heading_style))

            # Save dashboard Plotly figures if available
            dashboard_charts = ['viz_bar', 'viz_line', 'viz_pie', 'viz_hist', 'viz_scatter', 'viz_box', 'viz_violin', 'viz_corr']
            for chart in dashboard_charts:
                if chart in st.session_state:
                    try:
                        fig = st.session_state[chart]
                        if isinstance(fig, px.Figure):
                            img_dash = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                            pio.write_image(fig, img_dash, width=800, height=400)
                            elements.append(Image(img_dash, width=400, height=200))
                            elements.append(Spacer(1, 15))
                    except Exception as e:
                        elements.append(Paragraph(f"⚠️ Could not render {chart}: {e}", body_style))

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


            doc.build(elements)
            return pdf_path

        if st.button("📄 Generate Professional Report"):
            pdf_path = generate_styled_pdf(df)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download Professional EDA Report (PDF)", f, file_name="EDA_Report.pdf", mime="application/pdf")
            st.success("✅ EDA Report Generated Successfully!")
