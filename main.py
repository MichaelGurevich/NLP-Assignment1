from ClozeSolver import ClozeSolver
import time

if __name__ == '__main__':
    start_time = time.time()

    predictions = ClozeSolver.solve()

    print(predictions)

    elapsed_time = time.time() - start_time

    print(f"elapsed time: {elapsed_time:.2f} seconds")


