import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def run_fpgrowth(transactions, min_support):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    start_time = time.time()
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    execution_time = time.time() - start_time

    return df, frequent_itemsets, execution_time