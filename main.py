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


def test(path: str, index: int = 1, memory: int = 1, lead: int = 0, forecast: int = 2, buffer: int = 0,
         capacity: int = 0, power: int = 1, mode: str = "max", arima: dict or None = None,
         svr: dict or None = None) -> dict or None:
    dataset = load(path)
    if dataset is None:
        return None
    data = {"Inventory": dataset["Starting"]}
    if index < 1:
        index = 1
    s = str(index)
    while s not in dataset:
        index -= 1
        if index < 1:
            return None
    history = {}
    for item in dataset[s]:
        history[item] = []
    memory_index = index - 1
    fixed_memory = memory > 0
    if not fixed_memory:
        memory = index - 1
    while memory > 0:
        memory -= 1
        current = 1 if memory_index < 1 else memory_index
        for key in history:
            history[key].insert(0, dataset[str(current)][key])
        memory_index -= 1
    data["History"] = history
    data["Shipments"] = []
    successes = []
    failures = []
    losses = []
    imports = []
    succeeded = 0
    failed = 0
    lost = 0
    imported = 0
    required = 0
    for item in data["Inventory"]:
        required -= data["Inventory"][item]
    print(f"\nTesting on data from {path}.")
    if index > 2:
        print(f"Starting at {index}.")
    if lead > 0:
        print(f"Order lead time of {lead} periods.")
    if fixed_memory:
        print(f"Memory of {memory}.")
    if buffer > 0:
        print(f"Supply buffer of {buffer}.")
    if capacity > 0:
        print(f"Warehouse capacity of {capacity}.")
    print(f"Fitting with polynomials of {power}.")
    print(f"Choosing prediction by {mode}.")
    while True:
        if s not in dataset:
            print(f"\nSucceeded: {succeeded}")
            print(f"Failed: {failed}")
            if capacity > 0:
                print(f"Lost: {lost}")
            print(f"Imported: {imported}")
            print(f"Required: {required}")
            return {"Succeeded": succeeded, "Failed": failed, "Lost": lost, "Imported": imported, "Required": required,
                    "Successes": successes, "Failures": failures, "Losses": losses, "Imports": imports}
        current_import = {}
        for key in dataset[s]:
            current_import[key] = 0
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
                    current_import[item] += arrived[item]
                    imported += arrived[item]
                data["Shipments"].pop(i)
                i -= 1
        imports.append(current_import)
        shipment = predict(data, lead + forecast, buffer, power, mode, arima, svr)
        if isinstance(shipment, dict):
            for key in shipment:
                if key not in data["Inventory"]:
                    shipment.pop(key)
                elif shipment[key] < 0:
                    shipment[key] = 0
            for key in data["Inventory"]:
                if key not in shipment:
                    shipment[key] = 0
            shipment["Time"] = lead
            data["Shipments"].append(shipment)
        failure = {}
        success = {}
        for key in dataset[s]:
            order = dataset[s][key]
            current = data["Inventory"][key]
            if current - order > 0:
                data["Inventory"][key] -= order
                failure[key] = 0
                success[key] = order
            else:
                remaining = order - current
                failure[key] = remaining
                success[key] = current
                data["Inventory"][key] = 0
            failed += failure[key]
            succeeded += success[key]
            required += order
            history[key].append(order)
            if fixed_memory:
                history[key].pop(0)
        successes.append(success)
        failures.append(failure)
        loss = {}
        for key in dataset[s]:
            loss[key] = 0
        if capacity > 0:
            while sum(data["Inventory"].values()) > capacity:
                greatest = max(data["Inventory"], key=data["Inventory"].get)
                data["Inventory"][greatest] -= 1
                lost += 1
                loss[greatest] += 1
        losses.append(loss)
        index += 1
        s = str(index)


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


def predict(data: dict, forecast: int = 0, buffer: int = 0, power: int = 1, mode: str = "max",
            arima: dict or None = None, svr: dict or None = None) -> dict:
    if data is None and "Inventory" not in data:
        return {}
    shipment = {}
    for item in data["Inventory"]:
        shipment[item] = 0
    if "History" in data:
        if isinstance(arima, dict):
            arima = clean_parameters(arima, "p", 1, 0)
            arima = clean_parameters(arima, "d", 1, 0)
            arima = clean_parameters(arima, "q", 1, 0)
        else:
            arima = None
        if isinstance(svr, dict):
            svr = clean_parameters(svr, "C", 100, 0)
            svr = clean_parameters(svr, "gamma", "scale", ["auto", "scale"])
            svr = clean_parameters(svr, "epsilon", 0.1, 0)
        else:
            svr = None
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


test("Data/peak.csv", index=1, memory=5, lead=0, forecast=1, buffer=25, capacity=0, power=3, mode="max",
     arima={}, svr={})
