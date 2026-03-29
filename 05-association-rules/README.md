# Association Rules

Find **frequent itemsets** and association rules in transactional data.

## Algorithms

| Algorithm | Notes | File |
|-----------|-------|------|
| Apriori | Classic frequent itemset mining | apriori.py |

## Key Metrics

| Metric | Meaning |
|--------|---------|
| Support | P(A ∩ B) — how often the itemset appears |
| Confidence | P(B\|A) — how often B appears given A |
| Lift | Confidence / P(B) — lift > 1 means positive association |

## Installation

```bash
pip install mlxtend
```

## Lab

```bash
cd 05-association-rules
python apriori.py
```

## Cross-reference

- [cheatsheets/clustering.md](../cheatsheets/clustering.md)
