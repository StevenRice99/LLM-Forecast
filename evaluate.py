import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def evaluate(keys: list or None = None, x: int = 10, y: int = 5, show: bool = False) -> None:
    """
    Evaluate all results and make plots.
    :param keys: The keys to consider for plotting.
    :param x: The X size of the plots.
    :param y: The Y size of the plots.
    :param show: Whether to show the plots or not.
    :return: Nothing.
    """
    # Nothing to do if there are no results.
    if not os.path.exists("Results"):
        return None
    # Store all possible keys.
    all_keys = ["Inventory", "Arrived", "Available", "Needed", "Succeeded", "Failed", "Lost"]
    # Only allow valid keys.
    if isinstance(keys, list):
        for key in keys:
            if key not in all_keys:
                keys.remove(key)
    # If no keys, use all keys.
    if keys is None or len(keys) < 1:
        keys = all_keys
    # All possible titles to read from text files for final tabulation.
    titles = ["Succeeded", "Failed", "Lost", "Arrived", "Remaining"]
    # Store results for final tabulation.
    results = {}
    # Loop through all models.
    models = os.listdir("Results")
    for model in models:
        model_path = os.path.join("Results", model)
        if not os.path.isdir(model_path):
            continue
        # Loop through all datasets the model has been tested on.
        datasets = os.listdir(model_path)
        for dataset in datasets:
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue
            # Loop through all items which were in the dataset.
            items = os.listdir(dataset_path)
            for item in items:
                item_path = os.path.join(dataset_path, item)
                if not os.path.isfile(item_path):
                    continue
                # Create a plot from the CSV data.
                if item_path.endswith(".csv"):
                    df = pd.read_csv(item_path)
                    # Plot all desired series.
                    data = {}
                    for key in keys:
                        data[key] = []
                    for index, row in df.iterrows():
                        for key in data:
                            data[key].append(int(row[key]))
                    # Create the X axis.
                    time = 0
                    for key in data:
                        if len(data[key]) > 0:
                            time = len(data[key])
                    time = range(1, time + 1)
                    # Create the plot.
                    plt.figure(figsize=(x, y))
                    for key in data:
                        plt.plot(time, data[key], label=key)
                    plt.xlabel('Period')
                    plt.ylabel(item[:-4])
                    plt.legend()
                    ax = plt.gca()
                    # Ensure the values for each axis are whole numbers.
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    # Save the plot and optionally show it.
                    plt.savefig(f"{item_path[:-3]}png")
                    if show:
                        # noinspection PyBroadException
                        try:
                            plt.show()
                        except:
                            pass
                    plt.close()
                # Read the text file to get final tabulation results.
                if item_path.endswith(".txt"):
                    # Initialize all values to zero in case they are not found.
                    result = {}
                    for title in titles:
                        result[title] = 0
                    file = open(item_path, "r")
                    content = file.read()
                    file.close()
                    # Read each line of the file.
                    content = content.split("\n")
                    for line in content:
                        line = line.split(" ")
                        # If not enough data, skip it.
                        if len(line) < 2:
                            continue
                        # The values in the text file should end in ":" so remove it.
                        title = line[0][:-1]
                        # If the title is needed, set it.
                        if title in titles:
                            result[title] = int(line[1])
                    # Save it under the dataset and item.
                    category = f"{dataset} {item[:-4]}"
                    # Initialize the category if it does not yet exist.
                    if category not in results:
                        results[category] = {}
                    results[category][model] = result
    # Save tabulated results.
    for dataset_item in results:
        # Sort the data so the best models are listed first.
        current = dict(sorted(results[dataset_item].items(), key=lambda val: (
            -val[1]["Succeeded"],
            val[1]["Failed"],
            val[1]["Lost"],
            val[1]["Arrived"],
            val[1]["Remaining"],
            val[0]
        )))
        # Format the results as a CSV file and save it.
        s = "Start,Memory,Lead,Forecast,Buffer,Capacity,Top,Power,ARIMA,SVR"
        for title in titles:
            s += f",{title}"
        for model in current:
            parameters = model.split(" ")
            s += "\n"
            first = True
            for p in parameters:
                p = p.split("=")
                p = p[1] if len(p) > 1 else "-"
                s += p if first else f",{p}"
                first = False
            for title in current[model]:
                s += f",{current[model][title]}"
        f = open(os.path.join("Results", f"{dataset_item}.csv"), "w")
        f.write(s)
        f.close()


if __name__ == '__main__':
    evaluate(["Available", "Needed"], show=False)
