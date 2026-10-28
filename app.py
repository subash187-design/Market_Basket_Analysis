import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from modules.preprocess import preprocess_data
from modules.apriori_model import run_apriori
from modules.fpgrowth_model import run_fpgrowth
from modules.recommender import generate_rules, recommend_products
from modules.visualizer import plot_top_items

def parse_pasted_text(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    parsed_lines = []
    max_items = 0
    for line in lines:
        items = [item.strip() for item in line.split(",") if item.strip()]
        if items:
            parsed_lines.append(items)
            max_items = max(max_items, len(items))
    if not parsed_lines:
        return None
    col_names = ["TransactionID"] + [f"Item{i+1}" for i in range(max_items)]
    data_rows = []
    for idx, items in enumerate(parsed_lines, start=1):
        row = [idx] + items + [""] * (max_items - len(items))
        data_rows.append(row)
    return pd.DataFrame(data_rows, columns=col_names)

st.set_page_config(page_title="Retail Basket Analytics Dashboard", layout="wide")

# =========================
# Session State Initialization
# =========================
if "active_data" not in st.session_state:
    st.session_state.active_data = None

# =========================
# Sidebar Branding
# =========================
st.sidebar.title("🛒 Market Basket Analysis")
st.sidebar.caption("Retail Basket Analytics Dashboard")
st.sidebar.divider()

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("📁 Quick Upload")
sidebar_upload = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    key="sidebar_uploader"
)

if sidebar_upload is not None:
    try:
        if sidebar_upload.name.endswith(".xlsx") or sidebar_upload.name.endswith(".xls"):
            st.session_state.active_data = pd.read_excel(sidebar_upload)
        else:
            st.session_state.active_data = pd.read_csv(sidebar_upload)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

if st.session_state.active_data is not None:
    if st.sidebar.button("🔄 Load Different Data", use_container_width=True):
        st.session_state.active_data = None
        st.rerun()

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
# Interactive Landing Workspace (When no dataset is loaded)
# =========================
if st.session_state.active_data is None:
    # Feature overview cards
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        st.container(border=True).markdown("""
        ### 📥 1. Flexible Input
        Upload sales data in **CSV/Excel**, build transactions interactively in the live grid, or paste plain text.
        """)
    with card_col2:
        st.container(border=True).markdown("""
        ### ⚡ 2. Mining Engine
        Benchmark **Apriori** and **FP-Growth** side-by-side with execution time and frequent itemset counts.
        """)
    with card_col3:
        st.container(border=True).markdown("""
        ### 💡 3. Smart Insights
        Generate association rules with **Support**, **Confidence**, and **Lift** to drive cross-selling.
        """)

    st.write("")
    st.subheader("🚀 Get Started: Choose How to Input Your Transactions")

    tab_upload, tab_builder, tab_paste, tab_template = st.tabs([
        "📁 Upload File",
        "✍️ Build Transactions Online",
        "📋 Quick Text Paste",
        "🛠️ Interactive Template Builder"
    ])

    # Tab 1: Direct File Upload
    with tab_upload:
        st.markdown("#### Upload your transaction spreadsheet")
        c_fmt, _ = st.columns([1, 2])
        with c_fmt:
            main_file_type = st.radio("Choose file format", ["CSV", "Excel"], horizontal=True)

        main_upload = st.file_uploader(
            f"Drag and drop your {main_file_type} file here",
            type=["csv"] if main_file_type == "CSV" else ["xlsx", "xls"],
            key="main_uploader"
        )

        if main_upload is not None:
            try:
                if main_file_type == "CSV":
                    df_up = pd.read_csv(main_upload)
                else:
                    df_up = pd.read_excel(main_upload)

                st.success(f"✅ Loaded '{main_upload.name}' with {len(df_up)} rows and {len(df_up.columns)} columns.")
                st.caption("Preview of uploaded data:")
                st.dataframe(df_up.head(5), use_container_width=True)

                if st.button("🚀 Analyze Uploaded Transactions", type="primary", use_container_width=True):
                    st.session_state.active_data = df_up
                    st.rerun()
            except Exception as ex:
                st.error(f"Error reading file: {ex}")

    # Tab 2: Interactive Table Builder
    with tab_builder:
        st.markdown("#### Build or edit transactions directly in your browser")
        st.caption("Click any cell to edit, click the bottom row **(+)** to add new transactions, or select rows and press **Delete**.")

        if "live_builder_df" not in st.session_state:
            st.session_state.live_builder_df = pd.DataFrame([
                {"TransactionID": 1, "Item1": "Milk", "Item2": "Bread", "Item3": "Butter"},
                {"TransactionID": 2, "Item1": "Bread", "Item2": "Butter", "Item3": "Jam"},
                {"TransactionID": 3, "Item1": "Milk", "Item2": "Cookies", "Item3": ""},
                {"TransactionID": 4, "Item1": "Bread", "Item2": "Milk", "Item3": ""},
                {"TransactionID": 5, "Item1": "Butter", "Item2": "Jam", "Item3": ""},
            ])

        edited_builder = st.data_editor(
            st.session_state.live_builder_df,
            num_rows="dynamic",
            use_container_width=True,
            key="landing_builder_editor"
        )

        if st.button("🚀 Run Analysis on These Transactions", type="primary", use_container_width=True):
            st.session_state.active_data = edited_builder
            st.rerun()

    # Tab 3: Quick Text Paste
    with tab_paste:
        st.markdown("#### Paste comma-separated transactions")
        st.caption("Paste one transaction per line, with items separated by commas.")

        pasted_text = st.text_area(
            "Transaction Text",
            placeholder="Milk, Bread, Butter\nBread, Butter, Jam\nMilk, Cookies\nBread, Milk",
            height=160,
            label_visibility="collapsed"
        )

        if st.button("🚀 Convert & Run Analysis", type="primary", use_container_width=True):
            parsed_df = parse_pasted_text(pasted_text)
            if parsed_df is not None and not parsed_df.empty:
                st.session_state.active_data = parsed_df
                st.rerun()
            else:
                st.warning("Please enter at least one line of items separated by commas.")

    # Tab 4: Interactive Template Builder
    with tab_template:
        st.markdown("#### Customize and download a starter CSV template")
        c_rows, c_cols = st.columns(2)
        with c_rows:
            gen_rows = st.slider("Number of sample transaction rows", min_value=3, max_value=30, value=5)
        with c_cols:
            gen_items = st.slider("Number of item columns", min_value=2, max_value=8, value=4)

        template_cols = ["TransactionID"] + [f"Item{i+1}" for i in range(gen_items)]
        template_data = {
            col: [r + 1 if col == "TransactionID" else "" for r in range(gen_rows)]
            for col in template_cols
        }
        gen_template_df = pd.DataFrame(template_data)

        st.caption("Live Template Preview:")
        st.dataframe(gen_template_df, use_container_width=True)

        st.download_button(
            label=f"⬇ Download Custom CSV Template ({gen_rows} rows × {gen_items} items)",
            data=gen_template_df.to_csv(index=False),
            file_name="transactions_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.stop()

# =========================
# Interactive Transaction Editor (User can update transactions)
# =========================
raw_data = st.session_state.active_data

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