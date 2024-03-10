import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def show_random(data, n):
    for i in np.random.randint(len(data), size=(n)):
        print(f"RU: {data[i][0]}")
        print(f"EN: {data[i][1]}\n")


def stat(data, log=False):
    # Extract sentences for each language
    ru_sentences, en_sentences = zip(*data)

    # Tokenize sentences and calculate length
    ru_lengths = [len(sentence.split()) for sentence in ru_sentences]
    en_lengths = [len(sentence.split()) for sentence in en_sentences]

    # Calculate mean and standard deviation for sentence lengths
    ru_mean, ru_sd = np.mean(ru_lengths), np.std(ru_lengths)
    en_mean, en_sd = np.mean(en_lengths), np.std(en_lengths)

    # Tokenize all sentences into words
    ru_words = [word for sentence in ru_sentences for word in sentence.split()]
    en_words = [word for sentence in en_sentences for word in sentence.split()]

    # Count word frequencies
    ru_word_counts = Counter(ru_words)
    en_word_counts = Counter(en_words)

    print(f"Length: {len(data)}")
    print(f"Russian Mean: {ru_mean:.2f}, SD: {ru_sd:.2f}")
    print(f"English Mean: {en_mean:.2f}, SD: {en_sd:.2f}")
    print(f"Unique Russian words: {len(ru_word_counts)}")
    print(f"Unique English words: {len(en_word_counts)}")

    # Plot histograms for sentence lengths
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.hist(ru_lengths, bins=30, color='blue', alpha=0.7, log=log)
    plt.title('Histogram of Sentence Lengths in Russian')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.hist(en_lengths, bins=30, color='green', alpha=0.7, log=log)
    plt.title('Histogram of Sentence Lengths in English')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    # Plot histograms for word frequencies
    plt.subplot(2, 2, 3)
    plt.hist(ru_word_counts.values(), bins=30, color='red', alpha=0.7, log=log)
    plt.title('Histogram of Word Frequencies in Russian')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Words')

    plt.subplot(2, 2, 4)
    plt.hist(en_word_counts.values(), bins=30, color='orange', alpha=0.7, log=log)
    plt.title('Histogram of Word Frequencies in English')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Words')



def compare_datasets(datasets, random_seqs):
    for i in np.random.randint(len(datasets[0]), size=(random_seqs)):
        for j, dataset in enumerate(datasets):
            print(f'dataset {j}:', dataset[i])
        print()

    


