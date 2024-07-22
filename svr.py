from main import auto
from evaluate import evaluate

if __name__ == '__main__':
    # SVR trials.
    for c in [0.1, 1, 10, 100]:
        for gamma in [0.001, 0.01, 0.1, 1]:
            for epsilon in [0.01, 0.1, 0.5]:
                auto("Data/COVID Ontario.csv", 1, 8, 2, 4, 500, 0, 0, 5, None, {"C": c, "gamma": gamma, "epsilon": epsilon}, False, output="SVR")
    evaluate("Results/SVR", ["Available", "Needed", "Arrived"], True, 10, 5, False)
