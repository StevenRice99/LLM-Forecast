from main import auto
from evaluate import evaluate

if __name__ == '__main__':
    auto("Data/COVID Ontario.csv", 1, 8, 2, 2, 100, 0, 5, 5, {"p": 2, "d": 2, "q": 1}, {"C": 10, "gamma": "auto", "epsilon": 0.01}, False, output="FINAL", model=[None, "gpt-3.5"])
    evaluate("Results/FINAL", ["Available", "Needed", "Arrived"], True, 10, 5, False)
