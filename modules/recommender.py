from mlxtend.frequent_patterns import association_rules

def generate_rules(frequent_itemsets, min_confidence):
    if frequent_itemsets.empty or len(frequent_itemsets) < 2:
        return None

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return None

    rules = rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    return rules

def recommend_products(rules, selected_product):
    recommendations = []

    if rules is None or rules.empty:
        return recommendations

    for _, row in rules.iterrows():
        antecedent_items = [item.strip() for item in row["antecedents"].split(",")]
        if selected_product in antecedent_items:
            recommendations.append({
                "recommended_product": row["consequents"],
                "confidence": row["confidence"],
                "lift": row["lift"]
            })

    return recommendations