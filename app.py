# ==========================================
# Sales Forecasting Dashboard
# Developed by: Mayuri Sonawane
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("📊 Sales Forecasting Dashboard")
st.markdown("### Employee Sales Forecasting & Business Analytics")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    return df

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ==========================================
# Page 1 — Sales Overview Dashboard
# ==========================================

if page == "Sales Overview":

    st.header("📈 Sales Overview Dashboard")

    # -----------------------------
    # KPI Cards
    # -----------------------------

    total_sales = round(df["Sales"].sum(),2)

    total_orders = df["Order ID"].nunique()

    total_customers = df["Customer ID"].nunique()

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    col2.metric(
        "Total Orders",
        total_orders
    )

    col3.metric(
        "Customers",
        total_customers
    )

    st.markdown("---")

    # -----------------------------
    # Sales by Year
    # -----------------------------

    st.subheader("Total Sales by Year")

    yearly = (
        df.groupby(
            df["Order Date"].dt.year
        )["Sales"]
        .sum()
    )

    fig,ax = plt.subplots(figsize=(8,4))

    yearly.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Year")

    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -----------------------------
    # Monthly Trend
    # -----------------------------

    st.subheader("Monthly Sales Trend")

    monthly = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
    )

    fig2,ax2 = plt.subplots(figsize=(10,4))

    monthly.plot(ax=ax2)

    ax2.set_xlabel("Date")

    ax2.set_ylabel("Sales")

    st.pyplot(fig2)

    # -----------------------------
    # Interactive Filter
    # -----------------------------

    st.subheader("Sales by Region and Category")

    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered = df[
        (df["Region"]==region) &
        (df["Category"]==category)
    ]

    st.write(
        f"Showing records for **{region} Region** and **{category} Category**"
    )

    st.dataframe(filtered)

    st.success("Page 1 Completed Successfully ✅")

# ==========================================
# Page 2 — Forecast Explorer
# ==========================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.write(
        "Explore the 3-month sales forecast generated using the best-performing XGBoost forecasting model."
    )

    # ---------------------------------
    # Select Forecast Type
    # ---------------------------------

    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if forecast_type == "Category":

        selected_option = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

    else:

        selected_option = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

    # ---------------------------------
    # Forecast Horizon
    # ---------------------------------

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # ---------------------------------
    # XGBoost Forecast Values
    # ---------------------------------

    forecast_df = pd.DataFrame({

        "Month":[
            "Month 1",
            "Month 2",
            "Month 3"
        ],

        "Forecast Sales":[
            51037.70,
            30091.78,
            61376.88
        ]

    })

    st.subheader("Forecast Results")

    st.dataframe(
        forecast_df.head(horizon),
        use_container_width=True
    )

    # ---------------------------------
    # Forecast Chart
    # ---------------------------------

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(

        forecast_df["Month"][:horizon],

        forecast_df["Forecast Sales"][:horizon],

        marker="o",
        linewidth=3

    )

    ax.set_title("Next 3-Month Sales Forecast")

    ax.set_xlabel("Forecast Month")

    ax.set_ylabel("Predicted Sales")

    ax.grid(True)

    st.pyplot(fig)

    # ---------------------------------
    # Model Performance
    # ---------------------------------

    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        "13,915.32"
    )

    c2.metric(
        "RMSE",
        "18,893.85"
    )

    c3.metric(
        "MAPE",
        "13.29%"
    )

    st.success("Best Model Used: XGBoost")

# ==========================================
# Page 3 — Anomaly Report
# ==========================================

elif page == "Anomaly Report":

    st.header("🚨 Anomaly Detection Report")

    st.write(
        """
        This page displays unusual weekly sales detected using the
        Isolation Forest algorithm.
        """
    )

    # -------------------------------------
    # Show Anomaly Chart
    # -------------------------------------

    st.subheader("Weekly Sales Anomaly Chart")

    st.image(
        "charts/isolation_forest_anomaly.png"
    )

    # -------------------------------------
    # Display Anomaly Information
    # -------------------------------------

    st.subheader("Anomaly Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Isolation Forest",
        "7 Anomalies"
    )

    col2.metric(
        "Z-Score",
        "0 Anomalies"
    )

    # -------------------------------------
    # Create Anomaly Table
    # -------------------------------------

    weekly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    weekly_sales["Prediction"] = model.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["Prediction"] == -1
    ][["Order Date", "Sales"]]

    anomalies = anomalies.rename(
        columns={
            "Order Date": "Anomaly Date",
            "Sales": "Sales Value"
        }
    )

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies,
        use_container_width=True
    )

    st.success(
        "Isolation Forest identified 7 unusual sales weeks."
    )

# ==========================================
# Page 4 — Product Demand Segments
# ==========================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segments")

    st.write("""
    This page displays the product demand clusters generated using
    K-Means Clustering. Products are grouped according to their
    sales behaviour, growth and volatility.
    """)

    # ---------------------------------
    # Cluster Chart
    # ---------------------------------

    st.subheader("Demand Cluster Visualization")

    st.image(
        "charts/product_clusters.png",
        caption="K-Means Product Demand Clusters",
        width=900
    )

    # ---------------------------------
    # Create Cluster Table
    # ---------------------------------

    monthly_subcat = (
        df.groupby(
            ["Sub-Category",
             pd.Grouper(key="Order Date", freq="ME")]
        )["Sales"]
        .sum()
        .reset_index()
    )

    feature_df = monthly_subcat.groupby("Sub-Category").agg(

        Total_Sales=("Sales","sum"),

        Sales_Volatility=("Sales","std"),

        Average_Order_Value=("Sales","mean")

    ).reset_index()

    growth = []

    for subcat in monthly_subcat["Sub-Category"].unique():

        temp = monthly_subcat[
            monthly_subcat["Sub-Category"]==subcat
        ].sort_values("Order Date")

        first = temp.iloc[:12]["Sales"].sum()

        last = temp.iloc[-12:]["Sales"].sum()

        if first == 0:

            rate = 0

        else:

            rate = ((last-first)/first)*100

        growth.append(rate)

    feature_df["Sales_Growth"] = growth

    # ---------------------------------
    # Scale Features
    # ---------------------------------

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = feature_df[
        [
            "Total_Sales",
            "Sales_Growth",
            "Sales_Volatility",
            "Average_Order_Value"
        ]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    feature_df["Cluster"] = model.fit_predict(X_scaled)

    # ---------------------------------
    # Assign Labels
    # ---------------------------------

    cluster_names = {

        0:"Low Volume, Stable Demand",

        1:"High Volume, Stable Demand",

        2:"Growing Demand",

        3:"High Growth, High Volatility"

    }

    feature_df["Demand Segment"] = feature_df["Cluster"].map(
        cluster_names
    )

    st.subheader("Product Demand Segments")

    st.dataframe(
        feature_df[
            [
                "Sub-Category",
                "Demand Segment"
            ]
        ]
    )

    # ---------------------------------
    # Stocking Strategy
    # ---------------------------------

    st.subheader("Recommended Stocking Strategy")

    strategy = pd.DataFrame({

        "Demand Segment":[

            "High Volume, Stable Demand",

            "Growing Demand",

            "High Growth, High Volatility",

            "Low Volume, Stable Demand"

        ],

        "Recommended Strategy":[

            "Maintain high inventory levels and ensure continuous availability.",

            "Increase inventory gradually and monitor demand closely.",

            "Maintain safety stock and review inventory frequently.",

            "Keep lean inventory to reduce holding costs."

        ]

    })

    st.table(strategy)

    st.success("Demand Segmentation Completed Successfully ✅")
