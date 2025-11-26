
from Heaps_Law import plot_heaps_law



def main():
    corpus_path = r"C:/Users/noona/Desktop/אקדמית/עיבוד שפה טבעית/assignment/data/en.wikipedia2018.10M.txt"

    print("Loading corpus and generating Heaps' Law plot...")
    plot_heaps_law(corpus_path)
    print("Done.")

if __name__ == "__main__":
    main()
