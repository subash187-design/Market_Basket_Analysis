import pandas as pd

def preprocess_data(data):
    if data is None or data.empty:
        return []

    # If first column is an ID column (e.g. TransactionID, ID, TID), skip it
    first_col = str(data.columns[0]).strip().lower()
    if data.shape[1] > 1 and any(k in first_col for k in ["transaction", "tid", "id", "receipt", "order"]):
        transactions = data.iloc[:, 1:].values.tolist()
    else:
        transactions = data.values.tolist()

    cleaned_transactions = []

    for transaction in transactions:
        cleaned = []
        seen = set()

        for item in transaction:
            if pd.notna(item):
                item = str(item).strip().title()   # standardize format
                if item != "" and item not in seen:
                    cleaned.append(item)
                    seen.add(item)

        if cleaned:
            cleaned_transactions.append(cleaned)

    return cleaned_transactions