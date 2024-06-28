from evaluate import evaluate
from main import test

if __name__ == '__main__':
    # Run and evaluate a single, simple demo.
    test("Data/COVID Ontario.csv", 1, 8, 2, 2, 500, 0, 3, 5, None, None, True)
    evaluate("Results", ["Available", "Needed", "Arrived"], True, 10, 5, False)
