from main import auto
from evaluate import evaluate

if __name__ == '__main__':
    auto("Data/COVID Ontario.csv", 1, 8, 0, 0, 100, 0, 5, 5, {"p": 2, "d": 2, "q": 1}, {"C": 0.1, "gamma": "auto", "epsilon": 0.5}, True, output="FINAL", model=[None, "Meta-Llama-3.1-405B"], delay=60)
    evaluate("Results/FINAL", ["Available", "Needed", "Arrived"], True, 10, 5, False)
