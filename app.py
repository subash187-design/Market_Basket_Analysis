import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from modules.preprocess import preprocess_data
from modules.apriori_model import run_apriori
from modules.fpgrowth_model import run_fpgrowth
from modules.recommender import generate_rules, recommend_products
from modules.visualizer import plot_top_items

st.set_page_config(page_title="Retail Basket Analytics Dashboard", layout="wide")

# =========================
# Sidebar Branding
# =========================
st.sidebar.title("🛒 Market Basket Analysis")
st.sidebar.caption("Retail Basket Analytics Dashboard")
st.sidebar.divider()

# =========================
# Data Source & Upload
# =========================
st.sidebar.header("📁 Transaction Data")
input_source = st.sidebar.radio(
    "Choose input method",
    ["Upload File", "Manual / Interactive Table"],
    index=0
)

if input_source == "Upload File":
    file_type = st.sidebar.radio("File format", ["CSV", "Excel"])
    uploaded_file = st.sidebar.file_uploader(
        f"Upload {file_type} file",
        type=["csv"] if file_type == "CSV" else ["xlsx", "xls"]
    )
else:
    uploaded_file = None

# =========================
# Analysis Settings
# =========================
st.sidebar.divider()
st.sidebar.header("⚙️ Analysis Settings")
min_support = st.sidebar.slider("Minimum Support", 0.05, 1.0, 0.2, 0.05)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.05, 1.0, 0.5, 0.05)
mode = st.sidebar.selectbox("Choose Mode", ["Single Algorithm", "Comparison Mode"])

if mode == "Single Algorithm":
    algorithm = st.sidebar.selectbox("Choose Algorithm", ["Apriori", "FP-Growth"])

# =========================
# Main Page Header
# =========================
st.title("🛒 Retail Basket Analytics Dashboard")
st.write("Discover product associations, compare algorithms, and generate smart recommendations.")

# =========================
# Data Acquisition (No default example transactions)
# =========================
raw_data = None

if input_source == "Upload File":
    if uploaded_file is not None:
        try:
            if file_type == "CSV":
                raw_data = pd.read_csv(uploaded_file)
            else:
                raw_data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        # Prompt the user to upload their dataset - DO NOT run any example analysis
        st.info("👈 Please upload your transaction file (CSV or Excel) in the sidebar to start the analysis.")

        with st.expander("ℹ️ How to format your transaction data", expanded=True):
            st.markdown("""
            Upload a **CSV** or **Excel** file containing retail transactions.
            
            **Example structure:**
            | TransactionID | Item1 | Item2 | Item3 |
            | :--- | :--- | :--- | :--- |
            | 1 | Milk | Bread | Butter |
            | 2 | Bread | Butter | |
            | 3 | Milk | Cookies | |
            
            - Each row represents a single customer transaction.
            - You can have multiple item columns or comma-separated items.
            """)

            sample_template = pd.DataFrame({
                "TransactionID": [1, 2, 3],
                "Item1": ["", "", ""],
                "Item2": ["", "", ""],
                "Item3": ["", "", ""]
            })
            st.download_button(
                label="⬇ Download Blank CSV Template",
                data=sample_template.to_csv(index=False),
                file_name="transaction_template.csv",
                mime="text/csv"
            )

        st.stop()
else:
    # Manual Entry / Table Editor
    if "manual_data" not in st.session_state:
        st.session_state.manual_data = pd.DataFrame({
            "TransactionID": [1, 2, 3],
            "Item1": ["Milk", "Bread", "Butter"],
            "Item2": ["Bread", "Butter", "Jam"],
            "Item3": ["Butter", "", ""]
        })
    raw_data = st.session_state.manual_data

# =========================
# Interactive Transaction Editor (User can update transactions)
# =========================
st.subheader("📝 Transaction Dataset (View & Update)")
st.caption("You can edit any cell, add new transaction rows (+), or delete rows. All analytics will update in real-time.")

data = st.data_editor(raw_data, num_rows="dynamic", use_container_width=True)

col_dl, _ = st.columns([1, 4])
with col_dl:
    st.download_button(
        label="⬇ Download Transactions CSV",
        data=data.to_csv(index=False),
        file_name="transactions.csv",
        mime="text/csv"
    )

transactions = preprocess_data(data)

if not transactions:
    st.warning("⚠️ No valid transactions found. Please add items in the table above to perform analysis.")
    st.stop()

# =========================
# Comparison Mode
# =========================
if mode == "Comparison Mode":
    st.subheader("📈 Algorithm Comparison Dashboard")

    basket_ap, freq_ap, time_ap = run_apriori(transactions, min_support)
    basket_fp, freq_fp, time_fp = run_fpgrowth(transactions, min_support)

    rules_ap = generate_rules(freq_ap, min_confidence)
    rules_fp = generate_rules(freq_fp, min_confidence)

    ap_rule_count = 0 if rules_ap is None else len(rules_ap)
    fp_rule_count = 0 if rules_fp is None else len(rules_fp)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apriori Time (sec)", f"{time_ap:.4f}")
    col2.metric("FP-Growth Time (sec)", f"{time_fp:.4f}")
    col3.metric("Apriori Itemsets", len(freq_ap))
    col4.metric("FP-Growth Itemsets", len(freq_fp))

    col5, col6 = st.columns(2)
    col5.metric("Apriori Rules", ap_rule_count)
    col6.metric("FP-Growth Rules", fp_rule_count)

    # Comparison table
    comparison_df = pd.DataFrame({
        "Metric": ["Execution Time (sec)", "Frequent Itemsets", "Association Rules"],
        "Apriori": [time_ap, len(freq_ap), ap_rule_count],
        "FP-Growth": [time_fp, len(freq_fp), fp_rule_count]
    })

    st.subheader("📋 Comparison Table")
    st.dataframe(comparison_df, use_container_width=True)

    # Comparison chart
    st.subheader("📊 Comparison Chart")
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Itemsets", "Rules"]
    apriori_values = [len(freq_ap), ap_rule_count]
    fpgrowth_values = [len(freq_fp), fp_rule_count]

    x = range(len(metrics))
    width = 0.35

    ax.bar([i - width/2 for i in x], apriori_values, width, label="Apriori")
    ax.bar([i + width/2 for i in x], fpgrowth_values, width, label="FP-Growth")

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Count")
    ax.set_title("Apriori vs FP-Growth")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Time comparison chart
    st.subheader("⏱️ Execution Time Comparison")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["Apriori", "FP-Growth"], [time_ap, time_fp])
    ax2.set_ylabel("Time (sec)")
    ax2.set_title("Execution Time Comparison")
    plt.tight_layout()
    st.pyplot(fig2)

# =========================
# Single Algorithm Mode
# =========================
else:
    if algorithm == "Apriori":
        basket, frequent_itemsets, execution_time = run_apriori(transactions, min_support)
    else:
        basket, frequent_itemsets, execution_time = run_fpgrowth(transactions, min_support)

    rules = generate_rules(frequent_itemsets, min_confidence)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", len(transactions))
    col2.metric("Unique Products", len(basket.columns))
    col3.metric("Frequent Itemsets", len(frequent_itemsets))
    col4.metric("Execution Time (sec)", f"{execution_time:.4f}")

    st.subheader("🧾 Encoded Basket Data")
    st.dataframe(basket, use_container_width=True)

    display_freq = frequent_itemsets.copy()
    if not display_freq.empty:
        display_freq["itemsets"] = display_freq["itemsets"].apply(lambda x: ", ".join(list(x)))

    st.subheader("📌 Frequent Itemsets")
    st.dataframe(display_freq, use_container_width=True)

    if not display_freq.empty:
        st.download_button(
            label="⬇ Download Frequent Itemsets CSV",
            data=display_freq.to_csv(index=False),
            file_name="frequent_itemsets.csv",
            mime="text/csv"
        )

        st.subheader("📊 Frequent Itemsets Chart")
        fig = plot_top_items(frequent_itemsets.copy())
        st.pyplot(fig)

    if rules is not None and not rules.empty:
        st.subheader("🔗 Association Rules")
        st.dataframe(rules, use_container_width=True)

        st.download_button(
            label="⬇ Download Association Rules CSV",
            data=rules.to_csv(index=False),
            file_name="association_rules.csv",
            mime="text/csv"
        )

        st.subheader("💡 Rule-Based Recommendations")
        for _, row in rules.iterrows():
            st.success(
                f"If a customer buys **{row['antecedents']}**, they may also buy **{row['consequents']}** "
                f"(Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})"
            )

        st.subheader("🔍 Product Recommendation Search")
        selected_product = st.selectbox("Select a product", sorted(basket.columns))
        recommendations = recommend_products(rules, selected_product)

        if recommendations:
            st.write(f"Recommended products for **{selected_product}**:")
            for rec in recommendations:
                st.info(
                    f"👉 **{rec['recommended_product']}** "
                    f"(Confidence: {rec['confidence']:.2f}, Lift: {rec['lift']:.2f})"
                )
        else:
            st.warning("No recommendation found for the selected product.")
    else:
        st.warning("No association rules found. Try lowering support or confidence.")