import matplotlib.pyplot as plt

def plot_top_items(frequent_itemsets):
    top = frequent_itemsets.sort_values(by="support", ascending=False).head(10).copy()
    top["itemsets"] = top["itemsets"].apply(lambda x: ", ".join(list(x)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top["itemsets"], top["support"])
    ax.set_xlabel("Itemsets")
    ax.set_ylabel("Support")
    ax.set_title("Top Frequent Itemsets")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig