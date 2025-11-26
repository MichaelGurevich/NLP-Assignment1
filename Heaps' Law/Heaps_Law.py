import matplotlib.pyplot as plt
from collections import Counter
import re


def plot_heaps_law(corpus_path, chapter_size=10000):
    """
    corpus_path: path to your large text file
    chapter_size: number of words to treat as one 'chapter'
    """

    vocab = set()
    total_words = 0

    # Stores results for plotting
    total_words_list = []
    vocab_size_list = []

    # Read corpus line by line (memory efficient)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        buffer_words = []

        for line in f:
            # Basic tokenization
            words = re.findall(r"\w+", line.lower())
            buffer_words.extend(words)

            # When buffer hits a 'chapter', process it
            while len(buffer_words) >= chapter_size:
                chunk = buffer_words[:chapter_size]
                buffer_words = buffer_words[chapter_size:]

                # Update totals
                total_words += len(chunk)
                vocab.update(chunk)

                # Store new accumulated measurements after every chapter
                total_words_list.append(total_words)
                vocab_size_list.append(len(vocab))

    # Process leftover words (if any)
    if buffer_words:
        total_words += len(buffer_words)
        vocab.update(buffer_words)
        total_words_list.append(total_words)
        vocab_size_list.append(len(vocab))

    # === Plot Heaps' Law ===
    plt.figure(figsize=(8, 6))
    plt.plot(total_words_list, vocab_size_list)
    plt.xlabel("Total Words (N) (in 10k chunks)")
    plt.ylabel("Vocabulary Size (V)")
    plt.title("Heaps' Law for Corpus")
    plt.grid(True)
    plt.show()

    return total_words_list, vocab_size_list


