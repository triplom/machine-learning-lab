"""
Apriori Algorithm — Association Rule Mining
============================================
Uses the mlxtend library.

Install: pip install mlxtend

Dataset: synthetic grocery basket transactions.

Goal: Find rules like "if customer buys bread and butter, they also buy milk"
with high support, confidence, and lift.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- Synthetic grocery transaction dataset ---
transactions = [
    ["Milk", "Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Butter"],
    ["Bread", "Butter", "Eggs"],
    ["Milk", "Bread", "Butter", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Eggs"],
    ["Milk", "Bread", "Eggs"],
    ["Bread", "Eggs"],
    ["Milk", "Bread", "Butter", "Eggs"],
    ["Milk", "Butter", "Eggs"],
    ["Bread", "Butter", "Milk"],
    ["Eggs", "Butter"],
    ["Milk", "Bread"],
    ["Bread", "Butter", "Eggs", "Milk"],
]

# Encode transactions into boolean matrix
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print("Encoded transaction matrix:")
print(df)
print()

# --- Apriori: find frequent itemsets ---
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
print("Frequent Itemsets (min_support=0.4):")
print(frequent_itemsets.sort_values("support", ascending=False).to_string(index=False))
print()

# --- Generate association rules ---
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values("lift", ascending=False)
print("Association Rules (lift >= 1.0):")
print(
    rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(
        index=False
    )
)
print()

# --- Filter: high confidence rules ---
strong_rules = rules[rules["confidence"] >= 0.7]
print("Strong Rules (confidence >= 0.7):")
print(
    strong_rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].to_string(index=False)
)
