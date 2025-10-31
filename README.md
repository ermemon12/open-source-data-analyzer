# 📊 Open Source Data Analyzer

**Open Source Data Analyzer** is a beginner-friendly, interactive tool built with **Streamlit** for cleaning, analyzing, and visualizing datasets. It enables users to perform end-to-end data analysis without writing any code.

---

## 🚀 Features

- **Upload & Explore**  
  Upload CSV or Excel files and preview the dataset easily.

- **Data Cleaning**  
  Handle missing values (fill with mean, median, or mode), remove duplicates, and standardize column names.

- **Exploratory Data Analysis (EDA)**  
  - Quick dataset overview: rows, columns, missing data, duplicates  
  - Numerical summaries & categorical top values  
  - Correlation heatmaps and scatter matrices  
  - Distribution & box plots for numeric columns  

- **Interactive Visualizations**  
  - Bar, Line, Scatter, Histogram, Box, Violin, Pie, and Heatmap charts  
  - Auto chart suggestion based on column types  
  - Smooth and interactive plotting with Plotly  

- **Dashboard Overview**  
  - Key metrics: row count, column count, missing values, duplicates  
  - Trend and numeric distribution visualization  
  - Strongest correlations display  

- **Generate PDF Report**  
  - Professional EDA report generation in PDF format  
  - Includes dataset overview, missing value heatmap, correlation heatmap, distributions, and key insights  

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- ReportLab (for PDF report generation)

---

## 📁 Project Structure

    open-source-data-analyzer/
    │
    ├─ app.py                 # Main Streamlit app
    ├─ requirements.txt       # Project dependencies
    ├─ .gitignore             # Ignore venv and cache files
    └─ venv/                  # Excluded from Git


---

## ⚡ Installation & Setup

### Clone the repository

       git clone https://github.com/yourusername/open-source-data-analyzer.git
       cd open-source-data-analyzer
### Create a virtual environment

    python -m venv venv


### Activate the virtual environment

#### Windows:

    venv\Scripts\activate


#### Mac/Linux:

    source venv/bin/activate


### Install dependencies

    pip install -r requirements.txt


### Run the app

    streamlit run app.py

### 🤝 Contributing

We welcome contributions from everyone!
See the CONTRIBUTING.md
 file for setup instructions and contribution guidelines.

### 🪪 License

This project is licensed under the MIT License
.

🌟 Acknowledgements

Open-source community ❤️

Contributors and testers

Data scientists and developers who support collaborative innovation
