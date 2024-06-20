import math
import os.path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


def load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("Item").T.to_dict()


def clean_parameters(d: dict, key: str, default: int or float or str = 1, acceptable: int or float or list = 0) -> dict:
    if key not in d:
        d[key] = [default]
    elif not isinstance(d[key], list):
        if isinstance(d[key], int) or isinstance(d[key], float):
            d[key] = [d[key]]
        else:
            d[key] = [default]
    d[key] = list(set(d[key]))
    if isinstance(acceptable, list):
        for i in range(len(d[key])):
            if d[key][i] not in acceptable:
                d[key].pop(i)
                i -= 1
    else:
        for i in range(len(d[key])):
            if d[key][i] < acceptable:
                d[key].pop(i)
                i -= 1
    if len(d[key]) == 0:
        d[key] = [default]
    return d


def clean_arima(arima: dict or None) -> dict or None:
    if isinstance(arima, dict):
        arima = clean_parameters(arima, "p", 1, 0)
        arima = clean_parameters(arima, "d", 1, 0)
        arima = clean_parameters(arima, "q", 1, 0)
    else:
        arima = None
    return arima


def clean_svr(svr: dict or None) -> dict or None:
    if isinstance(svr, dict):
        svr = clean_parameters(svr, "C", 100, 0)
        svr = clean_parameters(svr, "gamma", "scale", ["auto", "scale"])
        svr = clean_parameters(svr, "epsilon", 0.1, 0)
    else:
        svr = None
    return svr


def predict(data: dict, forecast: int = 0, buffer: int = 0, power: int = 1, mode: str = "max",
            arima: dict or None = None, svr: dict or None = None) -> dict:
    if data is None and "Inventory" not in data:
        return {}
    shipment = {}
    for item in data["Inventory"]:
        shipment[item] = 0
    if "History" in data:
        arima = clean_arima(arima)
        svr = clean_svr(svr)
        if arima is None and svr is None and power < 1:
            power = 1
        if mode != "max" and mode != "min" and mode != "average":
            mode = "max"
        if forecast < 0:
            forecast = 0
        for item in shipment:
            print(f"\nPredicting demand for {item}.")
            history = data["History"][item]
            results = []
            if len(history) > 0:
                results.append(history[-1])
            print(f"Previous orders are {history}.")
            for i in range(2, len(history) + 1):
                numbers = history[-i:]
                x = np.array(range(len(numbers))).reshape(-1, 1)
                y = np.array(numbers)
                for p in range(power + 1):
                    if p > len(numbers):
                        continue
                    model = make_pipeline(PolynomialFeatures(p), LinearRegression())
                    model.fit(x, y)
                    index = 0
                    result = 0
                    while index <= forecast:
                        result += model.predict(np.array([[len(numbers) + index]]))[0]
                        index += 1
                    if result < 0:
                        result = 0
                    results.append(math.ceil(result))
                if arima is not None:
                    for p in arima["p"]:
                        for d in arima["d"]:
                            for q in arima["q"]:
                                # noinspection PyBroadException
                                try:
                                    model = ARIMA(y, order=(p, d, q))
                                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                    warnings.filterwarnings("ignore", category=UserWarning)
                                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                                    model = model.fit()
                                    index = 1
                                    result = 0
                                    while index <= forecast + 1:
                                        result += model.forecast(index)[0]
                                        index += 1
                                    if result < 0:
                                        result = 0
                                    results.append(math.ceil(result))
                                except:
                                    continue
                if svr is not None:
                    for c in svr["C"]:
                        for g in svr["gamma"]:
                            for e in svr["epsilon"]:
                                # noinspection PyBroadException
                                try:
                                    model = SVR(kernel="rbf", C=c, gamma=g, epsilon=e)
                                    model.fit(x, y)
                                    index = 0
                                    result = 0
                                    while index <= forecast:
                                        result += model.predict(np.array([[len(numbers) + index]]))[0]
                                        index += 1
                                    if result < 0:
                                        result = 0
                                    results.append(math.ceil(result))
                                except:
                                    continue
            if len(results) < 1:
                if len(data["History"][item]) > 0:
                    results.append(data["History"][item])
                else:
                    results.append(0)
            print(f"Predictions are {results}.")
            if mode == "max":
                result = math.ceil(max(results))
            elif mode == "min":
                result = math.ceil(min(results))
            else:
                result = math.ceil(sum(results) / len(results))
            print(f"Predicted demand over next {forecast + 1} periods is {result} with {mode} estimation.")
            result -= data["Inventory"][item]
            print(f"Have {data['Inventory'][item]} in inventory, so need {result}.")
            for s in data["Shipments"]:
                if s["Time"] <= forecast and item in s:
                    if item in s and s[item] > 0:
                        result -= s[item]
                        print(f"Shipment arriving in {forecast} periods with {s[item]}, so now need {result}")
            if buffer > 0:
                result += buffer
                print(f"Want a buffer of {buffer}, so total needed is {result}.")
            if result < 0:
                result = 0
            shipment[item] = result
    return shipment


def test(path: str, start: int = 1, memory: int = 1, lead: int = 0, forecast: int = 2, buffer: int = 0,
         capacity: int = 0, power: int = 1, mode: str = "max", arima: dict or None = None,
         svr: dict or None = None) -> None:
    dataset = load(path)
    if dataset is None:
        return None
    data = {"Inventory": dataset["Starting"]}
    if start < 1:
        start = 1
    s = str(start)
    while s not in dataset:
        start -= 1
        if start < 1:
            return None
    history = {}
    for item in dataset[s]:
        history[item] = []
    memory_index = start - 1
    fixed_memory = memory > 0
    if not fixed_memory:
        memory = start - 1
    original_memory = memory
    while memory > 0:
        memory -= 1
        current = 1 if memory_index < 1 else memory_index
        for key in history:
            history[key].insert(0, dataset[str(current)][key])
        memory_index -= 1
    data["History"] = history
    data["Shipments"] = []
    succeeded = 0
    failed = 0
    loss = 0
    imported = 0
    required = 0
    for item in data["Inventory"]:
        required -= data["Inventory"][item]
    print(f"\nTesting on data from {path}.")
    arima = clean_arima(arima)
    svr = clean_svr(svr)
    if arima is None and svr is None and power < 1:
        power = 1
    if mode != "max" and mode != "min" and mode != "average":
        mode = "max"
    if forecast < 0:
        forecast = 0
    if start > 2:
        print(f"Starting at {start}.")
    if lead > 0:
        print(f"Order lead time of {lead} periods.")
    if buffer > 0:
        print(f"Want a supply buffer of {buffer}.")
    if capacity > 0:
        print(f"Warehouse capacity of {capacity}.")
    if fixed_memory:
        print(f"Memory of {memory}.")
    if power > 0:
        print(f"Fitting with polynomials up to {power}.")
    if arima is not None:
        print(f"Using ARIMA with P={arima['p']}, D={arima['d']}] and Q={arima['q']}.")
    if svr is not None:
        print(f"Using SVR with C={svr['C']}, Gamma={svr['gamma']}], and Epsilon={svr['epsilon']}.")
    if forecast > 0:
        print(f"Forecasting {forecast} periods.")
    print(f"Choosing prediction by {mode}.")
    results = []
    while True:
        if s not in dataset:
            print(f"\nSucceeded: {succeeded}")
            print(f"Failed: {failed}")
            if capacity > 0:
                print(f"Lost: {loss}")
            print(f"Imported: {imported}")
            print(f"Required: {required}")
            if not os.path.exists("Results"):
                os.mkdir("Results")
            name = (f"Start={start} Memory={original_memory if fixed_memory else 'All'} Lead={lead} Forecast={forecast}"
                    f"Buffer={buffer} Capacity={capacity if capacity > 0 else 'None'} "
                    f"Power={power if power > 0 else 'None'} ARIMA=")
            if arima is None:
                name += "None SVR="
            else:
                name += f"[P={arima['p']} D={arima['d']}] Q={arima['q']}] SVR="
            if svr is None:
                name += "None"
            else:
                name += f"[C={svr['C']} Gamma={svr['gamma']}] Epsilon={svr['epsilon']}]"
            save = os.path.join("Results", name)
            if not os.path.exists(save):
                os.mkdir(save)
            file = os.path.basename(path)
            file, _ = os.path.splitext(file)
            save = os.path.join(save, file)
            if not os.path.exists(save):
                os.mkdir(save)
            for key in data["Inventory"]:
                s = "Period,Inventory,Arrived,Available,Needed,Succeeded,Failed,Lost"
                index = 1
                for result in results:
                    inventory = result[key]["Inventory"]
                    arrived = result[key]["Arrived"]
                    succeeded = result[key]["Succeeded"]
                    failed = result[key]["Failed"]
                    lost = result[key]["Lost"]
                    s += (f"\n{index},{inventory},{arrived},{inventory + arrived},{succeeded + failed},{succeeded},"
                          f"{failed},{lost}")
                f = open(os.path.join(save, f"{key}.csv"), "w")
                f.write(s)
                f.close()
            return None
        inventory = {}
        arrived = {}
        successes = {}
        failures = {}
        lost = {}
        for key in data["Inventory"]:
            inventory[key] = data["Inventory"][key]
            arrived[key] = 0
            lost[key] = 0
        for i in range(len(data["Shipments"])):
            if i >= len(data["Shipments"]):
                break
            data["Shipments"][i]["Time"] -= 1
            if data["Shipments"][i]["Time"] <= 0:
                arrived = data["Shipments"][i]
                for item in arrived:
                    if item == "Time":
                        continue
                    data["Inventory"][item] += arrived[item]
                    arrived[item] += arrived[item]
                    imported += arrived[item]
                data["Shipments"].pop(i)
                i -= 1
        placed = predict(data, forecast, buffer, power, mode, arima, svr)
        if isinstance(placed, dict):
            for key in placed:
                if key not in data["Inventory"]:
                    placed.pop(key)
                elif placed[key] < 0:
                    placed[key] = 0
            for key in data["Inventory"]:
                if key not in placed:
                    placed[key] = 0
            placed["Time"] = lead
            data["Shipments"].append(placed)
        for key in dataset[s]:
            ordered = dataset[s][key]
            current = data["Inventory"][key]
            if current - ordered > 0:
                data["Inventory"][key] -= ordered
                failures[key] = 0
                successes[key] = ordered
            else:
                remaining = ordered - current
                failures[key] = remaining
                successes[key] = current
                data["Inventory"][key] = 0
            failed += failures[key]
            succeeded += successes[key]
            required += ordered
            history[key].append(ordered)
            if fixed_memory:
                history[key].pop(0)
        if capacity > 0:
            while sum(data["Inventory"].values()) > capacity:
                greatest = max(data["Inventory"], key=data["Inventory"].get)
                data["Inventory"][greatest] -= 1
                loss += 1
                lost[greatest] += 1
        result = {}
        for key in data["Inventory"]:
            result[key] = {"Inventory": inventory[key], "Arrived": arrived[key], "Succeeded": successes[key],
                           "Failed": failures[key], "Lost": lost[key]}
        results.append(result)
        start += 1
        s = str(start)


test("Data/peak.csv", start=1, memory=5, lead=2, forecast=2, buffer=25, capacity=0, power=3, mode="max",
     arima={}, svr={})
