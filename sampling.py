# core sampling techniques experiment
# synthetic dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare


RNG = np.random.default_rng(seed=42)


N = 2000
pclass = RNG.choice([1, 2, 3], size=N, p=[0.2, 0.3, 0.5])
sex = RNG.choice(["male", "female"], size=N, p=[0.65, 0.35])
# Survival probability depends on class & sex, just like the real Titanic.
base_prob = 0.15 + (pclass == 1) * 0.35 + (pclass == 2) * 0.15 + (sex == "female") * 0.35
survived = RNG.binomial(1, np.clip(base_prob, 0, 1))
age = RNG.normal(30, 12, size=N).clip(0.5, 80)

population = pd.DataFrame({
    "PassengerId": np.arange(1, N + 1),
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "Survived": survived,
})

print("=" * 70)
print("POPULATION OVERVIEW")
print("=" * 70)
print(population["Survived"].value_counts(normalize=True).rename("proportion"))
print()

def simple_random_sample(df, frac=0.2):
    return df.sample(frac=frac, random_state=42)


def systematic_sample(df, frac=0.2):
    n = len(df)
    step = int(1 / frac)
    start = RNG.integers(0, step)
    idx = np.arange(start, n, step)
    return df.iloc[idx]


def stratified_sample(df, strata_col, frac=0.2):
    parts = [group.sample(frac=frac, random_state=42)
             for _, group in df.groupby(strata_col)]
    return pd.concat(parts, ignore_index=True)


def cluster_sample(df, cluster_col, frac_clusters=0.5):
    clusters = df[cluster_col].unique()
    chosen = RNG.choice(clusters, size=int(len(clusters) * frac_clusters), replace=False)
    return df[df[cluster_col].isin(chosen)]



def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = RNG.integers(0, i + 1)
            if j < k:
                reservoir[j] = item
    return reservoir



def bootstrap_sample(df):
    return df.sample(n=len(df), replace=True, random_state=42)



def compare_to_population(sample, population, label):
    pop_counts = population["Survived"].value_counts(normalize=True).sort_index()
    samp_counts = sample["Survived"].value_counts(normalize=True).sort_index()
    samp_counts = samp_counts.reindex(pop_counts.index, fill_value=0)

  
    expected = pop_counts.values * len(sample)
    observed = samp_counts.values * len(sample)
    # Avoid zero-expected-count errors on tiny samples.
    expected = np.clip(expected, 1e-6, None)
    stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    print(f"{label:22s} | n={len(sample):5d} | "
          f"Survived%={samp_counts.get(1, 0):.3f} (pop={pop_counts.get(1, 0):.3f}) | "
          f"chi2={stat:7.3f} | p={p_value:.4f} "
          f"{'(sample looks representative)' if p_value > 0.05 else '(DISTRIBUTION SHIFTED!)'}")
    return samp_counts


if __name__ == "__main__":
    samples = {
        "Simple Random":    simple_random_sample(population),
        "Systematic":       systematic_sample(population),
        "Stratified":       stratified_sample(population, "Survived"),
        "Cluster (Pclass)": cluster_sample(population, "Pclass"),
        "Bootstrap":        bootstrap_sample(population),
    }

    print("=" * 70)
    print("DISTRIBUTION-COMPARISON TABLE (Survived proportion vs population)")
    print("=" * 70)
    results = {}
    for label, samp in samples.items():
        results[label] = compare_to_population(samp, population, label)

   
    print("\nReservoir sampling demo (k=200 from a simulated stream):")
    reservoir = reservoir_sample((row for _, row in population.iterrows()), k=200)
    reservoir_df = pd.DataFrame(reservoir)
    compare_to_population(reservoir_df, population, "Reservoir (k=200)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    pop_prop = population["Survived"].value_counts(normalize=True).sort_index()

    for ax, (label, samp) in zip(axes, samples.items()):
        samp_prop = samp["Survived"].value_counts(normalize=True).sort_index()
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, pop_prop.reindex([0, 1], fill_value=0), width, label="Population")
        ax.bar(x + width/2, samp_prop.reindex([0, 1], fill_value=0), width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(["Died", "Survived"])
        ax.set_title(label)
        ax.legend(fontsize=8)

    axes[-1].axis("off")  # spare subplot
    plt.suptitle("Sample vs Population: Survived Distribution per Sampling Method")
    plt.tight_layout()
    out_path = "outputs/day1_sampling_comparison.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved comparison chart -> {out_path}")

  
