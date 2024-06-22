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
    """
    Load a file for testing.
    :param path: The path to the file.
    :return: The loaded file or nothing if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("Item").T.to_dict()


def clean_parameters(d: dict, key: str, default: int or float or str = 1, acceptable: int or float or list = 0) -> dict:
    """
    Ensure parameters in a dictionary are valid.
    :param d: The dictionary to validate.
    :param key: The key to validate.
    :param default: The default value to apply if needed.
    :param acceptable: Acceptable values, either a list of possible values or a minimum value.
    :return: The validated dictionary.
    """
    # If the key does not exist, add the default.
    if key not in d:
        d[key] = [default]
    # Otherwise, check if it is a list.
    elif not isinstance(d[key], list):
        # If it is a single parameter, convert it to a list.
        if isinstance(d[key], int) or isinstance(d[key], float) or isinstance(d[key], str):
            d[key] = [d[key]]
        # Otherwise, it is an invalid type so make the default value.
        else:
            d[key] = [default]
    # Ensure it is distinct.
    d[key] = list(set(d[key]))
    # If there is a list of acceptable values, check them.
    if isinstance(acceptable, list):
        for i in range(len(d[key])):
            if d[key][i] not in acceptable:
                d[key].pop(i)
                i -= 1
    # Otherwise, remove values below the lower limit.
    else:
        for i in range(len(d[key])):
            if isinstance(d[key][i], str) or d[key][i] < acceptable:
                d[key].pop(i)
                i -= 1
    # Ensure there is at least one value.
    if len(d[key]) == 0:
        d[key] = [default]
    return d


def clean_arima(arima: dict or None) -> dict or None:
    """
    Ensure ARIMA parameters are valid.
    :param arima: The ARIMA dictionary to validate.
    :return: The validated ARIMA dictionary.
    """
    if isinstance(arima, dict):
        arima = clean_parameters(arima, "p", 1, 0)
        arima = clean_parameters(arima, "d", 1, 0)
        arima = clean_parameters(arima, "q", 1, 0)
    else:
        arima = None
    return arima


def clean_svr(svr: dict or None) -> dict or None:
    """
    Ensure SVR parameters are valid.
    :param svr: The SVR dictionary to validate.
    :return: The validated SVR dictionary.
    """
    if isinstance(svr, dict):
        svr = clean_parameters(svr, "C", 100, 0)
        svr = clean_parameters(svr, "gamma", "scale", ["auto", "scale"])
        svr = clean_parameters(svr, "epsilon", 0.1, 0)
    else:
        svr = None
    return svr


def predict(data: dict, forecast: int = 0, buffer: int = 0, power: int = 1, top: int = 1,
            arima: dict or None = None, svr: dict or None = None, verbose: bool = False) -> dict:
    """
    Predict what to order for an inbound shipment.
    :param data: The data with inventory, past orders, and upcoming arrivals.
    :param forecast: How much to forecast with the model.
    :param buffer: The desired inventory buffer to have.
    :param power: The polynomial up which to compute predictions with.
    :param top: The number of top parameters to take into account for averaging the prediction.
    :param arima: The ARIMA configuration to use.
    :param svr: The SVR configuration to use.
    :param verbose: Whether outputs should be verbose or not.
    :return: The order to place for an inbound shipment.
    """
    # Nothing to predict if there is no data.
    if data is None or "Inventory" not in data or "History" not in data:
        return {}
    # Start all orders at zero.
    shipment = {}
    for item in data["Inventory"]:
        shipment[item] = 0
    # Ensure the configuration is valid.
    arima = clean_arima(arima)
    svr = clean_svr(svr)
    if arima is None and svr is None and power < 1:
        power = 1
    if forecast < 0:
        forecast = 0
    # Predict every item.
    for item in shipment:
        if verbose:
            print(f"\nPredicting demand for {item}.")
        history = data["History"][item]
        results = []
        # Store the last order as a possible prediction.
        if len(history) > 0:
            results.append(history[-1])
        if verbose:
            print(f"Previous orders are {history}.")
        # Check with every step back in history.
        for i in range(2, len(history) + 1):
            numbers = history[-i:]
            x = np.array(range(len(numbers))).reshape(-1, 1)
            y = np.array(numbers)
            # Compute possible polynomials.
            for p in range(power + 1):
                # Do not try to compute polynomials that are a higher degree than there are numbers.
                if p >= len(numbers):
                    continue
                # Fit the data.
                model = make_pipeline(PolynomialFeatures(p), LinearRegression())
                model.fit(x, y)
                # Predict the demand over the forecasted period.
                index = 0
                result = 0
                while index <= forecast:
                    result += model.predict(np.array([[len(numbers) + index]]))[0]
                    index += 1
                # Cannot predict negative values.
                if result < 0:
                    result = 0
                results.append(math.ceil(result))
            # Run ARIMA.
            if arima is not None:
                for p in arima["p"]:
                    for d in arima["d"]:
                        for q in arima["q"]:
                            # noinspection PyBroadException
                            try:
                                # Fit the model.
                                model = ARIMA(y, order=(p, d, q))
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                warnings.filterwarnings("ignore", category=UserWarning)
                                warnings.filterwarnings("ignore", category=RuntimeWarning)
                                model = model.fit()
                                # Predict the demand over the forecasted period.
                                index = 1
                                result = 0
                                while index <= forecast + 1:
                                    result += model.forecast(index)[0]
                                    index += 1
                                # Cannot predict negative values.
                                if result < 0:
                                    result = 0
                                results.append(math.ceil(result))
                            except:
                                continue
            # Run SVR.
            if svr is not None:
                for c in svr["C"]:
                    for g in svr["gamma"]:
                        for e in svr["epsilon"]:
                            # noinspection PyBroadException
                            try:
                                # Fit the model.
                                model = SVR(kernel="rbf", C=c, gamma=g, epsilon=e)
                                model.fit(x, y)
                                # Predict the demand over the forecasted period.
                                index = 0
                                result = 0
                                while index <= forecast:
                                    result += model.predict(np.array([[len(numbers) + index]]))[0]
                                    index += 1
                                # Cannot predict negative values.
                                if result < 0:
                                    result = 0
                                results.append(math.ceil(result))
                            except:
                                continue
        # If no predictions were successful, assume zero required.
        if len(results) < 1:
            results.append(0)
        if verbose:
            print(f"Predictions are {results}.")
        # Determine the best prediction by averaging the top best values.
        if 0 < top < len(results):
            results = sorted(results, reverse=True)[:top]
            if verbose:
                print(f"Top {top} predictions are {results}.")
        result = math.ceil(sum(results) / len(results))
        if verbose:
            print(f"Predicted demand over next {forecast + 1} periods is {result}.")
        # Use what is in inventory to contribute towards the predicted requirements.
        result -= data["Inventory"][item]
        if verbose:
            print(f"Have {data['Inventory'][item]} in inventory, so need {result}.")
        # If any shipments will arrive during this period, contribute them towards the predicted requirements.
        for s in data["Shipments"]:
            if s["Time"] <= forecast and item in s:
                if item in s and s[item] > 0:
                    result -= s[item]
                    if verbose:
                        print(f"Shipment arriving in {forecast} periods with {s[item]}, so now need {result}")
        # Add the buffer that should be maintained.
        if buffer > 0:
            result += buffer
            if verbose:
                print(f"Want a buffer of {buffer}, so total needed is {result}.")
        # Cannot order a negative amount.
        if result < 0:
            result = 0
        if verbose:
            if result > 0:
                print(f"Placing order of {result}.")
            else:
                print(f"Not ordering anything.")
        shipment[item] = result
    return shipment


def test(path: str, start: int = 1, memory: int = 1, lead: int = 1, forecast: int = 0, buffer: int = 0,
         capacity: int = 0, power: int = 1, top: int = 1, arima: dict or None = None,
         svr: dict or None = None, verbose: bool = False) -> None:
    """
    Test a forecasting model given a CSV file.
    :param path: The path to the file.
    :param start: Which index in the data to start testing from.
    :param memory: How much past orders should be given to the forecasting model.
    :param lead: How long is the lead time before orders arrive.
    :param forecast: How much to forecast with the model.
    :param buffer: The desired inventory buffer to have.
    :param capacity: The maximum inventory capacity to hold.
    :param power: The polynomial up which to compute predictions with.
    :param top: The number of top parameters to take into account for averaging the prediction.
    :param arima: The ARIMA configuration to use.
    :param svr: The SVR configuration to use.
    :param verbose: Whether outputs should be verbose or not.
    :return: Nothing.
    """
    # Nothing to do if the dataset cannot be loaded.
    dataset = load(path)
    if dataset is None:
        return None
    # Ensure the starting index is valid.
    if start < 1:
        start = 1
    # Ensure there is a starting index, otherwise return nothing.
    while start not in dataset:
        start -= 1
        if start < 1:
            return None
    index = start
    # Initialize the inventory needed to succeed for the first day.
    data = {"Inventory": {}}
    for key in dataset[start]:
        data["Inventory"][key] = dataset[start][key]
    # Account for the lead time needed so the first orders can be fulfilled.
    if lead < 1:
        lead = 1
    first = start + 1
    last = first + lead - 1
    for i in range(first, last):
        for key in dataset[i]:
            data["Inventory"][key] += dataset[i][key]
    # Configure the order history.
    history = {}
    for item in dataset[start]:
        history[item] = []
    # If zero or a negative memory was passed, all previous values are passed to the forecasting model.
    fixed_memory = memory > 0
    if not fixed_memory:
        memory = start - 1
    original_memory = memory
    memory_index = start - 1
    # With fixed memory, account for the given number of past instances.
    # If there are not enough, pad the history with the first instance.
    if fixed_memory:
        while memory > 0:
            memory -= 1
            current = 1 if memory_index < 1 else memory_index
            for key in history:
                history[key].insert(0, dataset[current][key])
            memory_index -= 1
    # Otherwise, add all previous instances.
    else:
        while memory_index > 0:
            for key in history:
                history[key].insert(0, dataset[memory_index][key])
            memory_index -= 1
    # Start the history and shipments.
    data["History"] = history
    data["Shipments"] = []
    # Ensure parameters are valid.
    arima = clean_arima(arima)
    svr = clean_svr(svr)
    if arima is None and svr is None and power < 1:
        power = 1
    if forecast < 0:
        forecast = 0
    if top < 0:
        top = 0
    # Build the name for the current test.
    name = (f"Start={start} Memory={original_memory if fixed_memory else 'All'} Lead={lead} "
            f"Forecast={forecast} Buffer={buffer} Capacity={capacity if capacity > 0 else 'None'} "
            f"Top={top if top > 0 else 'All'} Power={power if power > 0 else 'None'} ARIMA=")
    if arima is None:
        name += "None SVR="
    else:
        name += f"[P={arima['p']},D={arima['d']}],Q={arima['q']}] SVR="
    if svr is None:
        name += "None"
    else:
        name += f"[C={svr['C']},Gamma={svr['gamma']}],Epsilon={svr['epsilon']}]"
    # Get the directories to save in.
    model_path = os.path.join("Results", name)
    file = os.path.basename(path)
    file, _ = os.path.splitext(file)
    data_path = os.path.join(model_path, file)
    # If this test has already been done, no need to do it again.
    if os.path.exists(data_path):
        return None
    print(f"\nTesting on data from {path}.")
    if start > 2:
        print(f"Starting at {start}.")
    if lead > 0:
        print(f"Order lead time of {lead} periods.")
    if buffer > 0:
        print(f"Want a supply buffer of {buffer}.")
    if capacity > 0:
        print(f"Warehouse capacity of {capacity}.")
    if fixed_memory:
        print(f"Memory of {original_memory}.")
    if power > 0:
        print(f"Fitting with polynomials up to {power}.")
    if arima is not None:
        print(f"Using ARIMA with P={arima['p']}, D={arima['d']}] and Q={arima['q']}.")
    if svr is not None:
        print(f"Using SVR with C={svr['C']}, Gamma={svr['gamma']}], and Epsilon={svr['epsilon']}.")
    if forecast > 0:
        print(f"Forecasting {forecast} periods.")
    if top > 0:
        print(f"Choosing prediction by the average of the top {top} predictions.")
    else:
        print("Choosing prediction by the average of all predictions.")
    # Loop until the end of the data is reached.
    results = []
    while True:
        # If the current step is not in the dataset, the testing is done.
        if index not in dataset:
            # Ensure the folder exists to hold all results.
            if not os.path.exists("Results"):
                os.mkdir("Results")
            # Ensure a folder for this configuration exists.
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            # Make a folder for this dataset.
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            # Write the data for every item type.
            for key in data["Inventory"]:
                s = "Period,Inventory,Arrived,Available,Needed,Succeeded,Failed,Lost"
                index = 1
                total_succeeded = 0
                total_failed = 0
                total_arrived = 0
                total_required = 0
                total_lost = 0
                # Write the data for every period.
                for result in results:
                    inventory = result[key]["Inventory"]
                    arrived = result[key]["Arrived"]
                    succeeded = result[key]["Succeeded"]
                    failed = result[key]["Failed"]
                    lost = result[key]["Lost"]
                    s += (f"\n{index},{inventory},{arrived},{inventory + arrived},{succeeded + failed},{succeeded},"
                          f"{failed},{lost}")
                    if index == 1:
                        total_required -= inventory
                    total_succeeded += succeeded
                    total_failed += failed
                    total_arrived += arrived
                    total_required += (succeeded + failed)
                    total_lost += lost
                    index += 1
                # Write the series to a CSV file.
                f = open(os.path.join(data_path, f"{key}.csv"), "w")
                f.write(s)
                f.close()
                s = (f"Succeeded: {total_succeeded}"
                     f"\nFailed: {total_failed}"
                     f"\nLost: {total_lost}"
                     f"\nArrived: {total_arrived}"
                     f"\nRequired: {total_required}"
                     f"\nRemaining: {data['Inventory'][key]}")
                if verbose:
                    print()
                print(f"{file} | {key}\n{s}")
                # Write the overview to a text file.
                f = open(os.path.join(data_path, f"{key}.txt"), "w")
                f.write(s)
                f.close()
            return None
        if verbose:
            print(f"\nPeriod {index}.")
        # Store the information for this period.
        inventory = {}
        arrived = {}
        successes = {}
        failures = {}
        lost = {}
        for key in data["Inventory"]:
            inventory[key] = data["Inventory"][key]
            arrived[key] = 0
            lost[key] = 0
        # Handle shipments.
        for i in range(len(data["Shipments"])):
            if i >= len(data["Shipments"]):
                break
            # Reduce the periods remaining before arriving for each shipment.
            data["Shipments"][i]["Time"] -= 1
            # If the shipment has arrived, add them to inventory.
            if data["Shipments"][i]["Time"] <= 0:
                curr = data["Shipments"][i]
                for item in curr:
                    if item == "Time":
                        continue
                    data["Inventory"][item] += curr[item]
                    arrived[item] += curr[item]
                data["Shipments"].pop(i)
                i -= 1
        # Get the order to be placed by the forecasting model.
        placed = predict(data, forecast, buffer, power, top, arima, svr, verbose)
        # Make the request for the order.
        if isinstance(placed, dict):
            # Ensure only valid items are ordered.
            # No use yet, but could be useful for future LLM implementations.
            for key in placed:
                if key not in data["Inventory"]:
                    placed.pop(key)
                elif placed[key] < 0:
                    placed[key] = 0
            # If an item was not ordered, ensure it is represented.
            for key in data["Inventory"]:
                if key not in placed:
                    placed[key] = 0
            # Set the lead time for it to arrive and add it to the shipments.
            placed["Time"] = lead
            data["Shipments"].append(placed)
        # Check what orders need to be fulfilled.
        for key in dataset[index]:
            ordered = dataset[index][key]
            current = data["Inventory"][key]
            # If there is enough in inventory to cover the order, cover it fully.
            if current - ordered > 0:
                data["Inventory"][key] -= ordered
                failures[key] = 0
                successes[key] = ordered
            # Otherwise, determine how much was covered and how much failed.
            else:
                remaining = ordered - current
                failures[key] = remaining
                successes[key] = current
                data["Inventory"][key] = 0
            # Add the order to the history.
            history[key].append(ordered)
            # Trim the oldest order if using a fixed memory size.
            if fixed_memory:
                history[key].pop(0)
        # If there is a maximum capacity, it must be met.
        if capacity > 0:
            # Discard items until the capacity is met.
            while sum(data["Inventory"].values()) > capacity:
                # Remove the item which has the largest capacity.
                greatest = max(data["Inventory"], key=data["Inventory"].get)
                data["Inventory"][greatest] -= 1
                lost[greatest] += 1
        # Store results.
        result = {}
        for key in data["Inventory"]:
            result[key] = {"Inventory": inventory[key], "Arrived": arrived[key], "Succeeded": successes[key],
                           "Failed": failures[key], "Lost": lost[key]}
        results.append(result)
        # Go to the next period.
        index += 1


def auto(path: str or list, start: int or list = 1, memory: int or list = 1, lead: int or list = 1,
         forecast: int or list = 0, buffer: int or list = 0, capacity: int or list = 0, power: int or list = 1,
         top: int or list = 1, arima: list or dict or None = None, svr: dict or list or None = None,
         verbose: bool = False) -> None:
    """
    Automatically test multiple options.
    :param path: The path to the file.
    :param start: Which index in the data to start testing from.
    :param memory: How much past orders should be given to the forecasting model.
    :param lead: How long is the lead time before orders arrive.
    :param forecast: How much to forecast with the model.
    :param buffer: The desired inventory buffer to have.
    :param capacity: The maximum inventory capacity to hold.
    :param power: The polynomial up which to compute predictions with.
    :param top: The number of top parameters to take into account for averaging the prediction.
    :param arima: The ARIMA configuration to use.
    :param svr: The SVR configuration to use.
    :param verbose: Whether outputs should be verbose or not.
    :return: Nothing.
    """
    # Ensure values are all converted to lists.
    if isinstance(path, str):
        path = [path]
    if isinstance(start, int):
        start = [start]
    if isinstance(memory, int):
        memory = [memory]
    if isinstance(lead, int):
        lead = [lead]
    if isinstance(forecast, int):
        forecast = [forecast]
    if isinstance(buffer, int):
        buffer = [buffer]
    if isinstance(capacity, int):
        capacity = [capacity]
    if isinstance(power, int):
        power = [power]
    if isinstance(top, int):
        top = [top]
    if not isinstance(arima, list):
        arima = [arima]
    if not isinstance(svr, list):
        svr = [svr]
    # Test all configurations.
    for p in path:
        for s in start:
            for m in memory:
                for le in lead:
                    for f in forecast:
                        for b in buffer:
                            for c in capacity:
                                for po in power:
                                    for t in top:
                                        for a in arima:
                                            for sv in svr:
                                                test(p, s, m, le, f, b, c, po, t, a, sv, verbose)


if __name__ == '__main__':
    auto("Data/COVID Ontario.csv", 1, 8, 2, [2, 4], 500, 0, [1, 2, 3], [0, 1, 3, 5, 10], [None, {}], [None, {}], False)
