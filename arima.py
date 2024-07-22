from main import auto
from evaluate import evaluate

if __name__ == '__main__':
    # ARIMA trials.
    for p in [0, 1, 2]:
        for d in [0, 1, 2]:
            for q in [0, 1, 2]:
                auto("Data/COVID Ontario.csv", 1, 8, 2, 4, 500, 0, 0, 5, {"p": p, "d": d, "q": q}, None, False, output="ARIMA")
    evaluate("Results/ARIMA", ["Available", "Needed", "Arrived"], True, 10, 5, False)
