import pandas as pd

def preprocess_data(data):
    # Take all item columns except first column (TransactionID)
    transactions = data.iloc[:, 1:].values.tolist()

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