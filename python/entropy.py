
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from scipy.stats import binom_test

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import chi2_contingency

from scipy.stats import entropy


# Function to calculate entropy
# def calculate_entropy(series):
#     counts = series.value_counts()
#     probabilities = counts / len(series)
#     entropy = -np.sum(probabilities * np.log2(probabilities))
#     return entropy

def calculate_entropy(series):
    counts = series.value_counts()
    probabilities = counts / len(series)
    return entropy(probabilities, base=2)

def reportEntropyPerTrait(fileIn):
    df = pd.read_csv(fileIn)

    def calculate_entropy(selected, total):
        if total == 0 or selected == 0 or selected == total:
            return 0  # Avoid log(0) issues
        p = selected / total


        q = 1 - p
        return -p * np.log2(p) - q * np.log2(q)

    df['entropy'] = df.apply(lambda row: calculate_entropy(row['selectedCnt'], row['totalCnt']), axis=1)


    # Group by trait and sum entropy
    trait_entropy = df.groupby('trait')['entropy'].mean().reset_index()

    print(trait_entropy)


reportEntropyPerTrait('../mturk2/mturk2formattedResults.csv')
# Old method
# def reportEntropyPerTrait():
#     # df = pd.read_csv('formattedResults3.csv')
#     # df = pd.read_csv('../csv/formattedResults2.csv')
#     df = pd.read_csv('../mturk2/mturk2formattedResults.csv')
#     grouped = df.groupby(['video.left', 'video.right'])
#
#     entropy_per_trait = {
#         'o': [],
#         'c': [],
#         'e': [],
#         'a': [],
#         'n': []
#     }
#
#     for name, group in grouped:
#         entropy_per_trait['o'].append(calculate_entropy(group['o.answer']))
#         entropy_per_trait['c'].append(calculate_entropy(group['c.answer']))
#         entropy_per_trait['e'].append(calculate_entropy(group['e.answer']))
#         entropy_per_trait['a'].append(calculate_entropy(group['a.answer']))
#         entropy_per_trait['n'].append(calculate_entropy(group['n.answer']))
#
#     # Calculate average entropy per trait
#     average_entropy_per_trait = {trait: np.mean(entropies) for trait, entropies in entropy_per_trait.items()}
#
#
#     # Display the results
#     for trait, avg_entropy in average_entropy_per_trait.items():
#         print(f"Average entropy for {trait.upper()}: {avg_entropy:.4f}")


def calculatePermutation(fileIn):
    # Function to calculate entropy
    df = pd.read_csv(fileIn)
    # df = pd.read_csv('formattedResults2.csv')
    # Calculate observed entropy for each trait
    # Sample DataFrame based on the provided structure



    # Function to calculate entropy


    # Group by video pairs and calculate entropy for each trait
    grouped = df.groupby(['video.left', 'video.right'])

    entropy_per_trait = {
        'o': [],
        'c': [],
        'e': [],
        'a': [],
        'n': []
    }

    for name, group in grouped:
        entropy_per_trait['o'].append(calculate_entropy(group['o.answer']))
        entropy_per_trait['c'].append(calculate_entropy(group['c.answer']))
        entropy_per_trait['e'].append(calculate_entropy(group['e.answer']))
        entropy_per_trait['a'].append(calculate_entropy(group['a.answer']))
        entropy_per_trait['n'].append(calculate_entropy(group['n.answer']))

    # Calculate average entropy per trait
    average_entropy_per_trait = {trait: np.mean(entropies) for trait, entropies in entropy_per_trait.items()}
    stderr_entropy_per_trait = {trait: np.std(entropies) / np.sqrt(len(entropies)) for trait, entropies in
                                entropy_per_trait.items()}

    # Perform permutation test to calculate p-values
    n_permutations = 100
    p_values = {trait: 0 for trait in average_entropy_per_trait.keys()}

    for trait in average_entropy_per_trait.keys():
        print
        permuted_entropies = []
        for _ in range(n_permutations):
            permuted_group_entropies = []
            for name, group in grouped:
                permuted_series = group[trait + '.answer'].sample(frac=0.5, replace=False).reset_index(drop=True)
                permuted_group_entropy = calculate_entropy(permuted_series)
                permuted_group_entropies.append(permuted_group_entropy)
            permuted_entropies.append(np.mean(permuted_group_entropies))

        permuted_entropies = np.array(permuted_entropies)
        p_values[trait] = np.mean(permuted_entropies >= average_entropy_per_trait[trait])


    # # Display the results
    for trait, avg_entropy in average_entropy_per_trait.items():
        print(f"Average entropy for {trait.upper()}: {avg_entropy:.4f}")
        # print(f"Average entropy for {trait.upper()}: {avg_entropy:.4f}, p-value: {p_values[trait]:.4f}")

    # Prepare data for plotting
    plt.rcParams.update({'font.size': 16})
    traits = list(average_entropy_per_trait.keys())
    avg_entropies = list(average_entropy_per_trait.values())
    stderr_entropies = list(stderr_entropy_per_trait.values())
    p_values_list = list(p_values.values())

    # Plotting the box plot with error bars and significance stars
    fig, ax = plt.subplots()

    # Create box plots
    data_to_plot = [np.array(entropy_per_trait[trait]) for trait in traits]
    box = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True)

    # Add colors and means to the box plots
    colors = ['#FF9999' for _ in traits]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # medians
    for median in box['medians']:
        median.set(color='black')
    for whisker in box['whiskers']:
        whisker.set(linestyle=":")
    # # Add significance stars
    for i, p_value in enumerate(p_values_list):
        if p_value < 0.05:
            ax.text(i + 1,  max(entropy_per_trait[trait]), '*', ha='center', va='bottom', color='black', fontsize=12)

    # Customize plot
    ax.set_xticklabels(['Open.', 'Consc.', 'Agree', 'Extro.', 'Stab.'],
                       rotation=45, ha='right')
    ax.set_xlabel('Traits')
    ax.set_ylabel('1 - Entropy')
    plt.tight_layout()
    plt.show()
# reformatFiles()
# precisionAccuracy()
# calculatePrecision(0)



# confusionMatrix()

# reportEntropyPerTrait()
# calculatePermutation('../mturk2/mturk2formattedResults.csv')

# calculatePermutation('formattedResults3.csv')

# calculatePermutation('formattedResults2.csv')
