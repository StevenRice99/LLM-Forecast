from main import auto
from evaluate import evaluate

if __name__ == '__main__':
    # Polynomial trials.
    auto("Data/COVID Ontario.csv", 1, 8, 2, 2, 100, 0, [1, 2, 3, 4, 5], 5, None, None, False, output="POLYNOMIAL")
    evaluate("Results/POLYNOMIAL", ["Available", "Needed", "Arrived"], True, 10, 5, False)
    # ARIMA trials.
    for p in [0, 1, 2]:
        for d in [0, 1, 2]:
            for q in [0, 1, 2]:
                auto("Data/COVID Ontario.csv", 1, 8, 2, 2, 100, 0, 0, 5, {"p": p, "d": d, "q": q}, None, False,
                     output="ARIMA")
    evaluate("Results/ARIMA", ["Available", "Needed", "Arrived"], True, 10, 5, False)
    # SVR trials.
    for c in [0.1, 1, 10]:
        for gamma in ["auto", "scale"]:
            for epsilon in [0.01, 0.1, 0.5]:
                auto("Data/COVID Ontario.csv", 1, 8, 2, 2, 100, 0, 0, 5, None, {"C": c, "gamma": gamma, "epsilon": epsilon}, False, output="SVR")
    evaluate("Results/SVR", ["Available", "Needed", "Arrived"], True, 10, 5, False)
