import numpy as np
def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += -p * np.log2(p)
    return entropy
fair_coin_probs = [0.5, 0.5]
entropy_fair_coin = calculate_entropy(fair_coin_probs)
print(f"Entropy of a fair coin toss: {entropy_fair_coin:.3f} bits")
biased_coin_probs = [0.9, 0.1]
entropy_biased_coin = calculate_entropy(biased_coin_probs)
print(f"Entropy  {entropy_biased_coin:.3f} bits")
predictable_event_probs = [1.0, 0.0]
entropy_predictable_event = calculate_entropy(predictable_event_probs)
print(f"Entropy {entropy_predictable_event:.3f} bits")