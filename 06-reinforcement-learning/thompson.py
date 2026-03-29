"""
Thompson Sampling
==================
Problem: 10 ads with unknown click-through rates.
Goal: Maximise total clicks over 10,000 rounds.

Thompson Sampling uses Beta distributions:
  - For each ad, maintain Beta(alpha, beta) distribution
  - alpha = 1 + successes, beta = 1 + failures
  - Sample from each distribution and pick the highest

Generally outperforms UCB because it is Bayesian.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

N = 10000
d = 10
true_ctrs = [0.1, 0.25, 0.15, 0.35, 0.05, 0.4, 0.2, 0.3, 0.12, 0.18]

random.seed(42)
dataset = np.array(
    [[1 if random.random() < true_ctrs[j] else 0 for j in range(d)] for _ in range(N)]
)

# --- Thompson Sampling ---
numbers_of_rewards_1 = [0] * d  # successes (clicks)
numbers_of_rewards_0 = [0] * d  # failures (no clicks)
ads_selected = []
total_reward = 0

for n in range(N):
    # Sample from Beta distribution for each ad
    theta = [
        np.random.beta(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        for i in range(d)
    ]
    ad = np.argmax(theta)
    ads_selected.append(ad)
    reward = dataset[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward

print(f"Total reward (Thompson): {total_reward}")
print(f"Ad selected most: Ad {np.argmax(numbers_of_rewards_1) + 1}")
print(f"True best ad: Ad {np.argmax(true_ctrs) + 1} (CTR={max(true_ctrs)})")
print(f"\nSuccesses per ad: {numbers_of_rewards_1}")
print(f"Failures per ad:  {numbers_of_rewards_0}")

# Estimated CTR from Beta distribution means
estimated_ctrs = [
    numbers_of_rewards_1[i] / max(1, numbers_of_rewards_1[i] + numbers_of_rewards_0[i])
    for i in range(d)
]
print("\nEstimated CTRs:")
for i, (est, true) in enumerate(zip(estimated_ctrs, true_ctrs)):
    print(f"  Ad {i + 1}: estimated={est:.3f}  true={true}")

# Histogram
plt.figure(figsize=(10, 5))
plt.hist(ads_selected, bins=range(d + 1), align="left", rwidth=0.8, color="darkorange")
plt.title("Thompson Sampling — Ad Selection Frequency")
plt.xlabel("Ad")
plt.ylabel("Number of times selected")
plt.xticks(range(d), [f"Ad {i + 1}" for i in range(d)])
plt.tight_layout()
plt.savefig("thompson_selections.png", dpi=100)
plt.show()
print("Plot saved: thompson_selections.png")
