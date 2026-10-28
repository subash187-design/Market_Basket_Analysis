import pandas as pd

def preprocess_data(data):
    """
    Cleans and standardizes transaction data.
    Supports:
    - DataFrames with ID columns (TransactionID, TID, etc.)
    - DataFrames with only item columns
    - DataFrames with comma-separated items in cells
    """
    if data is None or data.empty:
        return []

    # Check if first column is likely an ID column
    first_col_name = str(data.columns[0]).strip().lower()
    is_id_col = (
        data.shape[1] > 1 and 
        any(k in first_col_name for k in ["transaction", "tid", "id", "receipt", "order", "invoice"])
    )

    items_df = data.iloc[:, 1:] if is_id_col else data

    cleaned_transactions = []

    for _, row in items_df.iterrows():
        cleaned = []
        seen = set()

        for item in row:
            if pd.notna(item):
                # Handle comma-separated items in a single cell if present
                item_str = str(item).strip()
                parts = item_str.split(",") if "," in item_str else [item_str]

                for part in parts:
                    clean_item = part.strip().title()
                    if clean_item and clean_item.lower() not in ["none", "nan", "null", ""] and clean_item not in seen:
                        cleaned.append(clean_item)
                        seen.add(clean_item)

        if cleaned:
            cleaned_transactions.append(cleaned)

    return cleaned_transactions