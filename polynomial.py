from main import auto
from evaluate import evaluate

if __name__ == '__main__':
    # Polynomial trials.
    auto("Data/COVID Ontario.csv", 1, 8, 2, 4, 500, 0, [0, 1, 2, 3, 4, 5], 5, None, None, False, output="POLYNOMIAL")
    evaluate("Results/POLYNOMIAL", ["Available", "Needed", "Arrived"], True, 10, 5, False)
