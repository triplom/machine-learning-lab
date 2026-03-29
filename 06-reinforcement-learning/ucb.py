"""
Upper Confidence Bound (UCB)
=============================
Problem: 10 ads with unknown click-through rates.
Goal: Maximise total clicks over 10,000 rounds.

UCB formula:
  UCB_i = avg_reward_i + sqrt(3/2 * ln(n) / n_i)

At each step: select the ad with the highest UCB.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Simulate 10 ads with fixed (unknown to agent) CTRs
N = 10000  # total rounds
d = 10  # number of ads
true_ctrs = [0.1, 0.25, 0.15, 0.35, 0.05, 0.4, 0.2, 0.3, 0.12, 0.18]

# Simulate full dataset (what reward we'd get if we chose each ad each round)
random.seed(42)
dataset = np.array(
    [[1 if random.random() < true_ctrs[j] else 0 for j in range(d)] for _ in range(N)]
)

# --- UCB Algorithm ---
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(1, N + 1):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = np.sqrt(1.5 * np.log(n) / numbers_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = float("inf")  # ensure every ad is tried at least once
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset[n - 1, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

print(f"Total reward (UCB): {total_reward}")
print(
    f"Ad selected most: Ad {np.argmax(numbers_of_selections) + 1} "
    f"({max(numbers_of_selections)} times)"
)
print(f"True best ad: Ad {np.argmax(true_ctrs) + 1} (CTR={max(true_ctrs)})")
print(f"\nSelections per ad: {numbers_of_selections}")

# Histogram of ad selections
plt.figure(figsize=(10, 5))
plt.hist(ads_selected, bins=range(d + 1), align="left", rwidth=0.8, color="steelblue")
plt.title("UCB — Ad Selection Frequency")
plt.xlabel("Ad")
plt.ylabel("Number of times selected")
plt.xticks(range(d), [f"Ad {i + 1}" for i in range(d)])
plt.tight_layout()
plt.savefig("ucb_selections.png", dpi=100)
plt.show()
print("Plot saved: ucb_selections.png")
