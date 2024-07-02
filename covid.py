import os
import requests

import pandas as pd


def convert():
    """
    Convert Ontario COVID hospitalizations into the format for processing.
    :return: Nothing.
    """
    # Ensure the directory for the dataset exists.
    if not os.path.exists("COVID"):
        os.mkdir("COVID")
    if not os.path.exists("COVID"):
        return
    # The URL of the dataset if it needs to be downloaded.
    url = "https://raw.githubusercontent.com/ccodwg/CovidTimelineCanada/main/data/pt/hosp_admissions_pt.csv"
    # If not file does not exist, it needs to be downloaded.
    path = os.path.join("COVID", os.path.basename(url))
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()
        if response.ok:
            f = open(os.path.join(path), "wb")
            f.write(response.content)
            f.close()
    # If the file still does not exist, it failed downloading.
    if not os.path.basename(path):
        return
    # Read the file and select only Ontario cases.
    df = pd.read_csv(path)
    df = df[df["region"] == "ON"]
    # Get the data into a list for indexing.
    data = []
    dates = []
    for index, row in df.iterrows():
        data.append(row["value_daily"])
        dates.append(row["date"])
    # The Ontario data was only done weekly, so keep only those value.
    week = 0
    step = 7
    cleaned = []
    cleaned_dates = []
    for i in range(len(data)):
        if i == 1 or i == week:
            cleaned.append(int(data[i]))
            cleaned_dates.append(dates[i])
            week += step
    # Ensure the directory to save the formatted data exists.
    if not os.path.exists("Data"):
        os.mkdir("Data")
    if not os.path.exists("Data"):
        return
    # Format the data and save it.
    s = "Item,Hospitalizations"
    for i in range(len(cleaned)):
        s += f"\n{i + 1},{cleaned[i]}"
    f = open(os.path.join("Data", "COVID Ontario.csv"), "w")
    f.write(s)
    f.close()
    # Write dates for use in web scraping later.
    path = os.path.join("Data", "Dates")
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path):
        return
    s = cleaned_dates[0]
    for i in range(1, len(cleaned_dates)):
        s += f"\n{cleaned_dates[i]}"
    f = open(os.path.join(path, "COVID Ontario.txt"), "w")
    f.write(s)
    f.close()


if __name__ == '__main__':
    convert()
