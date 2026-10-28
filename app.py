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
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Analysis Settings")
file_type = st.sidebar.radio("Choose input type", ["CSV", "Excel"])
min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.2, 0.05)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.5, 0.05)
mode = st.sidebar.selectbox("Choose Mode", ["Single Algorithm", "Comparison Mode"])

if mode == "Single Algorithm":
    algorithm = st.sidebar.selectbox("Choose Algorithm", ["Apriori", "FP-Growth"])

# =========================
# File Upload
# =========================
if file_type == "CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
else:
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

st.title("🛒 Retail Basket Analytics Dashboard")
st.write("Discover product associations, compare algorithms, and generate smart recommendations.")

# =========================
# Read Data (User-driven upload)
# =========================
if uploaded_file is not None:
    try:
        if file_type == "CSV":
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("👆 Please upload a transaction file (CSV or Excel) to begin the analysis.")
    st.stop()

# =========================
# Dynamic Transaction Table (View, Edit, Add, Delete)
# =========================
st.subheader("📝 Dynamic Transaction Table")
st.caption("💡 **Interactive features:** Click any cell to edit, click **'+'** at the bottom to add a new transaction, or select rows and press **Delete**. Sorting, searching, and fullscreen are also enabled.")

# Dynamic interactive data editor
data = st.data_editor(data, num_rows="dynamic", use_container_width=True)

col_dl_tx, _ = st.columns([1, 4])
with col_dl_tx:
    st.download_button(
        label="⬇ Download Current Transactions CSV",
        data=data.to_csv(index=False),
        file_name="transactions.csv",
        mime="text/csv"
    )

transactions = preprocess_data(data)

if not transactions:
    st.warning("⚠️ No valid transactions found in the table. Please make sure items are listed in the rows.")
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

    # Comparison Table
    comparison_df = pd.DataFrame({
        "Metric": ["Execution Time (sec)", "Frequent Itemsets", "Association Rules"],
        "Apriori": [f"{time_ap:.4f}", len(freq_ap), ap_rule_count],
        "FP-Growth": [f"{time_fp:.4f}", len(freq_fp), fp_rule_count],
        "Winner": [
            "FP-Growth 🏆" if time_fp < time_ap else "Apriori 🏆",
            "Tie (Identical)" if len(freq_ap) == len(freq_fp) else ("FP-Growth" if len(freq_fp) > len(freq_ap) else "Apriori"),
            "Tie (Identical)" if ap_rule_count == fp_rule_count else ("FP-Growth" if fp_rule_count > ap_rule_count else "Apriori")
        ]
    })

    st.subheader("📋 Dynamic Comparison Table")
    st.dataframe(comparison_df, use_container_width=True)

    # Comparison Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📊 Itemsets & Rules Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics = ["Itemsets", "Rules"]
        apriori_values = [len(freq_ap), ap_rule_count]
        fpgrowth_values = [len(freq_fp), fp_rule_count]
        x = range(len(metrics))
        width = 0.35
        ax.bar([i - width/2 for i in x], apriori_values, width, label="Apriori", color="#4C78A8")
        ax.bar([i + width/2 for i in x], fpgrowth_values, width, label="FP-Growth", color="#F58518")
        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Count")
        ax.set_title("Apriori vs FP-Growth")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    with col_chart2:
        st.subheader("⏱️ Execution Time Comparison")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(["Apriori", "FP-Growth"], [time_ap, time_fp], color=["#4C78A8", "#F58518"])
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Execution Speed (Lower is better)")
        plt.tight_layout()
        st.pyplot(fig2)

    # Detailed Interactive Tables Tab
    st.subheader("🔍 Explore Algorithm Rule Tables")
    tab_ap, tab_fp = st.tabs(["Apriori Association Rules", "FP-Growth Association Rules"])

    for tab, rules_obj, name in [(tab_ap, rules_ap, "Apriori"), (tab_fp, rules_fp, "FP-Growth")]:
        with tab:
            if rules_obj is not None and not rules_obj.empty:
                st.dataframe(
                    rules_obj,
                    column_config={
                        "support": st.column_config.ProgressColumn("Support", format="%.3f", min_value=0.0, max_value=1.0),
                        "confidence": st.column_config.ProgressColumn("Confidence", format="%.3f", min_value=0.0, max_value=1.0),
                        "lift": st.column_config.NumberColumn("Lift", format="%.3f")
                    },
                    use_container_width=True
                )
            else:
                st.info(f"No association rules found for {name} with current settings.")

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

    # =========================
    # Encoded Basket Data
    # =========================
    st.subheader("🧾 Encoded Basket Data Matrix")
    st.caption("Binary representation (True/False) of product occurrence per customer transaction.")
    st.dataframe(basket, use_container_width=True)

    # =========================
    # Dynamic Frequent Itemsets Table
    # =========================
    st.subheader("📌 Dynamic Frequent Itemsets Table")

    display_freq = frequent_itemsets.copy()
    if not display_freq.empty:
        display_freq["item_count"] = display_freq["itemsets"].apply(len)
        display_freq["itemsets"] = display_freq["itemsets"].apply(lambda x: ", ".join(list(x)))

        # Interactive filter controls
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            item_search = st.text_input("🔍 Search specific product in frequent itemsets", placeholder="e.g. Milk, Bread...")
        with col_f2:
            max_items = int(display_freq["item_count"].max()) if not display_freq.empty else 1
            selected_size = st.selectbox(
                "Filter by item count",
                ["All Sizes"] + [f"{i} item(s)" for i in range(1, max_items + 1)]
            )

        # Apply filters
        filtered_freq = display_freq.copy()
        if item_search.strip():
            filtered_freq = filtered_freq[filtered_freq["itemsets"].str.contains(item_search.strip(), case=False, na=False)]
        if selected_size != "All Sizes":
            size_num = int(selected_size.split()[0])
            filtered_freq = filtered_freq[filtered_freq["item_count"] == size_num]

        # Dynamic table with progress bar column
        st.dataframe(
            filtered_freq,
            column_config={
                "support": st.column_config.ProgressColumn(
                    "Support",
                    help="Frequency of the itemset across transactions",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "item_count": st.column_config.NumberColumn(
                    "Item Count",
                    help="Number of items in this itemset",
                    format="%d"
                ),
                "itemsets": st.column_config.TextColumn("Itemset Products")
            },
            use_container_width=True
        )

        st.download_button(
            label="⬇ Download Filtered Frequent Itemsets CSV",
            data=filtered_freq.to_csv(index=False),
            file_name="frequent_itemsets.csv",
            mime="text/csv"
        )

        st.subheader("📊 Top Frequent Itemsets Chart")
        fig = plot_top_items(frequent_itemsets.copy())
        st.pyplot(fig)
    else:
        st.warning("No frequent itemsets found with current Minimum Support. Try lowering the threshold.")

    # =========================
    # Dynamic Association Rules Table
    # =========================
    st.subheader("🔗 Dynamic Association Rules Table")

    if rules is not None and not rules.empty:
        # Dynamic filter controls
        col_r1, col_r2, col_r3 = st.columns(3)

        all_antecedents = sorted(set(rules["antecedents"].str.split(", ").explode()))
        all_consequents = sorted(set(rules["consequents"].str.split(", ").explode()))

        with col_r1:
            sel_antecedent = st.selectbox("Filter Antecedent (If buys)", ["All"] + all_antecedents)
        with col_r2:
            sel_consequent = st.selectbox("Filter Consequent (Then buys)", ["All"] + all_consequents)
        with col_r3:
            min_lift_val = float(rules["lift"].min())
            max_lift_val = float(rules["lift"].max())
            filter_lift = st.slider(
                "Filter by Minimum Lift",
                min_value=0.0,
                max_value=max(2.0, round(max_lift_val + 0.5, 1)),
                value=0.0,
                step=0.1
            )

        # Apply filters
        filtered_rules = rules.copy()
        if sel_antecedent != "All":
            filtered_rules = filtered_rules[filtered_rules["antecedents"].str.contains(sel_antecedent, case=False, regex=False)]
        if sel_consequent != "All":
            filtered_rules = filtered_rules[filtered_rules["consequents"].str.contains(sel_consequent, case=False, regex=False)]
        if filter_lift > 0.0:
            filtered_rules = filtered_rules[filtered_rules["lift"] >= filter_lift]

        # Dynamic table with interactive progress bars for support and confidence
        st.dataframe(
            filtered_rules,
            column_config={
                "antecedents": st.column_config.TextColumn("Antecedent (If bought)"),
                "consequents": st.column_config.TextColumn("Consequent (Then bought)"),
                "support": st.column_config.ProgressColumn(
                    "Support",
                    help="Joint probability of antecedent and consequent",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Likelihood of consequent given antecedent",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "lift": st.column_config.NumberColumn(
                    "Lift",
                    help="Strength of rule (>1 indicates positive association)",
                    format="%.3f"
                ),
            },
            use_container_width=True
        )

        st.download_button(
            label="⬇ Download Association Rules CSV",
            data=filtered_rules.to_csv(index=False),
            file_name="association_rules.csv",
            mime="text/csv"
        )

        # =========================
        # Rule-Based Insights
        # =========================
        st.subheader("💡 Rule-Based Insights")
        for _, row in filtered_rules.head(10).iterrows():
            st.success(
                f"If a customer buys **{row['antecedents']}**, they may also buy **{row['consequents']}** "
                f"(Confidence: **{row['confidence']:.2f}**, Lift: **{row['lift']:.2f}**)"
            )

        # =========================
        # Product Recommendation Search
        # =========================
        st.subheader("🔍 Interactive Product Recommender")
        selected_product = st.selectbox("Select a product to find recommendations for", sorted(basket.columns))
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
        st.warning("No association rules found. Try lowering minimum support or confidence in the sidebar.")