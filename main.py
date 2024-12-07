import argparse
import datetime
import itertools
import logging
import math
import os
import re
import warnings
from urllib.parse import urlparse

import inflect
import numpy as np
import ollama
import pandas
import pandas as pd
import requests
from gnews import GNews
from matplotlib import pyplot as plt, pyplot
from newspaper import Article
from pandas import DataFrame
from scipy.stats import norm
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from webdriver_manager.firefox import GeckoDriverManager

from publishers import load_trusted


def load() -> DataFrame:
    """
    Convert Ontario COVID hospitalizations into the format for processing.
    :return: The loaded dataset.
    """
    # If the correct version of the dataset already exists,
    path = os.path.join("Data", "Data.csv")
    if os.path.exists(path):
        return pd.read_csv(os.path.join(path))
    print("Loading the data.")
    if not os.path.exists("Data"):
        os.mkdir("Data")
    path = os.path.join("Data", "Raw.csv")
    # If not file does not exist, it needs to be downloaded.
    if not os.path.exists(path):
        response = requests.get(
            "https://raw.githubusercontent.com/ccodwg/CovidTimelineCanada/main/data/pt/hosp_admissions_pt.csv")
        response.raise_for_status()
        if response.ok:
            f = open(path, "wb")
            f.write(response.content)
            f.close()
    # Read the file and select only Ontario cases.
    df = pd.read_csv(path)
    # We no longer need the raw file.
    os.remove(path)
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
    # Format the data and save it.
    s = "Date,Hospitalizations"
    for i in range(len(cleaned)):
        s += f"\n{cleaned_dates[i]},{cleaned[i]}"
    path = os.path.join("Data", "Data.csv")
    f = open(path, "w")
    f.write(s)
    f.close()
    return pd.read_csv(os.path.join(path))


def clean(message: str) -> str:
    """
    Clean a message.
    :param message: The message to clean.
    :return: The message with all newlines being at most single, and all other whitespace being replaced by a space.
    """
    # Remove markdown symbols.
    for symbol in ["*", "#", "_", ">"]:
        message = message.replace(symbol, " ")
    # Replace all whitespace with spaces, except newlines.
    message = re.sub(r"[^\S\n]+", " ", message)
    # Ensure no duplicate newlines.
    while message.__contains__("\n\n"):
        message = message.replace("\n\n", "\n")
    # Strip and return the message.
    return message.strip()


def clean_response(s: str) -> str:
    """
    Clean a response.
    :param s: The string to clean.
    :return: The cleaned string.
    """
    # Remove any generic statement saying how it is relevant.
    s = s.replace("The article is relevant for forecasting COVID-19 hospitalizations in Ontario, Canada.\n", "")
    s = s.replace("Relevant for forecasting COVID-19 hospitalizations in Ontario, Canada.\n", "")
    s = s.replace("The article is relevant.\n", "")
    s = s.replace("Summary:\n", "")
    s = s.replace(". . .", "...")
    s = s.replace(" :", ":")
    s = s.replace("\n ", "\n")
    # The LLM tends to respond with a variation of TRUE which is not needed so remove it.
    s = s.replace("TRUE: ", "")
    s = s.replace("TRUE. ", "")
    s = s.replace("TRUE:", "")
    s = s.replace("TRUE.", "")
    s = s.replace("TRUE:", "")
    s = s.replace("TRUE", "")
    s = s.strip()
    return s


def initialize() -> None:
    """
    Ensure the model can be set up for Ollama.
    :return: Nothing.
    """
    logging.getLogger().setLevel(logging.WARNING)
    ollama.pull("llama3.1")


def generate(prompt: str, prompt_clean: bool = False, response_clean: bool = True) -> str:
    """
    Generate a response from a large language model from Ollama.
    :param prompt: The prompt you wish to pass.
    :param prompt_clean: If the prompt should be cleaned before being passed to the large language model.
    :param response_clean: If the response from the large language model should be cleaned.
    :return: The generated response.
    """
    if prompt_clean:
        prompt = clean(prompt)
    # Use no temperature for consistent results.
    response = ollama.generate("llama3.1", prompt, options={"temperature": 0, "num_ctx": 8196})["response"]
    return clean(response) if response_clean else response


def get_article(result: dict, driver) -> dict or None:
    """
    Get an article.
    :param result: The Google News result.
    :param driver: The Selenium webdriver.
    :return: The details of the article if it was relevant, otherwise nothing.
    """
    # Try to get the full article and then summarize it with an LLM.
    # noinspection PyBroadException
    try:
        # See if there is a redirect URL.
        url = urlparse(result["url"])
        if url.hostname == "google.com" or url.hostname == "news.google.com":
            # Get what the web driver is starting at.
            driver_url = driver.current_url
            # Store the starting URL.
            redirect_url = result["url"]
            # Go to the redirect URL.
            driver.get(redirect_url)
            # Wait until the web driver hits the page.
            while True:
                current_url = driver.current_url
                # Failsafe in case the URL is none to start.
                if current_url is None:
                    continue
                # Failsafe in case the URL is empty to start.
                if current_url == "":
                    continue
                # Skip the default starting page.
                if current_url == "about:blank":
                    continue
                # Ensure we are not at the initial URL when loading the web driver.
                if current_url == driver_url:
                    continue
                # Ensure we are not still at the redirect URL.
                if current_url == redirect_url:
                    continue
                # Update the previously visited URL for future redirect handling.
                driver_url = current_url
                break
            # Download the final article.
            article = Article(driver_url)
        else:
            article = Article(result["url"])
        article.download()
        article.parse()
        title = article.title
        # Clean the text.
        summary = clean(article.text)
        # Summarize the summary with an LLM if requested to.
        if summary != "":
            prompt = (f"You are trying to help forecast COVID-19 hospitalizations in Ontario, Canada. Below is an "
                      f"article. If the article is not relevant for forecasting COVID-19 hospitalizations in Ontario, "
                      f'Canada, respond with "FALSE". If the article is relevant for forecasting COVID-19 '
                      f"hospitalizations in Ontario, Canada, respond with a brief summary highlighting values most "
                      f"important for forecasting COVID-19 hospitalizations in Ontario, Canada. Only state facts and "
                      f"keep sentences short:\n\n{summary}")
            summary = clean_response(generate(prompt, False, True))
    # If the full article cannot be downloaded or the summarization fails, use the initial news info.
    except:
        return None
    # No point in having the summary if it is just equal to the title.
    if title is None or title == "":
        return None
    # If the article was not relevant or summarizing failed, discard it.
    if (summary is None or title == summary or summary == "" or summary.isspace() or summary.startswith("FALSE") or
            summary.startswith("I'm sorry, ") or summary.startswith("I apologize, ")):
        return None
    # Parse the date.
    published_date = result["published date"].split(" ")
    if published_date[2] == "Jan":
        published_date[2] = 1
    elif published_date[2] == "Feb":
        published_date[2] = 2
    elif published_date[2] == "Mar":
        published_date[2] = 3
    elif published_date[2] == "Apr":
        published_date[2] = 4
    elif published_date[2] == "May":
        published_date[2] = 5
    elif published_date[2] == "Jun":
        published_date[2] = 6
    elif published_date[2] == "Jul":
        published_date[2] = 7
    elif published_date[2] == "Aug":
        published_date[2] = 8
    elif published_date[2] == "Sep":
        published_date[2] = 9
    elif published_date[2] == "Oct":
        published_date[2] = 10
    elif published_date[2] == "Nov":
        published_date[2] = 11
    else:
        published_date[2] = 12
    # Parse the time
    published_time = published_date[4].split(":")
    # Get the publisher.
    publisher = result["publisher"]["title"]
    # Store the formatted data.
    return {"Title": title, "Summary": summary, "Publisher": publisher, "Year": int(published_date[3]),
            "Month": published_date[2], "Day": int(published_date[1]), "Hour": int(published_time[0]),
            "Minute": int(published_time[1]), "Second": int(published_time[2])}


def prepare_articles(dataset: pandas.DataFrame) -> None:
    """
    Prepare all news articles.
    :param dataset: The dataset to test on.
    :return: Nothing.
    """
    # Get the dates.
    dates = dataset["Date"]
    driver = None
    index = 0
    for date in dates:
        index += 1
        if os.path.exists(os.path.join("Data", f"{date}.txt")):
            continue
        # Load the Firefox webdriver the first time it is needed, and initialize the LLM.
        if driver is None:
            driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
            initialize()
        print(f"Summarizing news for period {index} of {len(dates)} periods.")
        parsed = date.split("-")
        end = datetime.datetime(int(parsed[0]), int(parsed[1]), int(parsed[2]))
        start = end - datetime.timedelta(days=7)
        google_news = GNews(language="en", country="CA", start_date=start, end_date=end, max_results=10)
        formatted = []
        results = google_news.get_news("COVID-19 Hospitalizations Ontario Canada")
        for result in results:
            result = get_article(result, driver)
            if result is not None:
                formatted.append(result)
        # Sort results by newest to oldest.
        formatted = sorted(formatted, key=lambda val: (
            -val["Year"],
            -val["Month"],
            -val["Day"],
            -val["Hour"],
            -val["Minute"],
            -val["Second"]
        ))
        s = ""
        # If there are results, format them.
        if len(formatted) > 0:
            single = len(formatted) == 1
            count = "a" if single else f"{len(formatted)}"
            s += (f"Below {'is' if single else 'are'} {count} news article{'' if single else 's'} from the past week to"
                  f" help guide you in making your decision.")
            # Add every result.
            for i in range(len(formatted)):
                result = formatted[i]
                days = (end - datetime.datetime(result["Year"], result["Month"], result["Day"])).days
                if days < 1:
                    posted = "Today"
                else:
                    posted = f"{days} day{'s' if days > 1 else ''} ago"
                s += "\n\n"
                if len(formatted) > 1:
                    s += f"Article {i + 1} of {len(formatted)}\n"
                s += f"Title: {result['Title']}\nPublisher: {result['Publisher']}\nPosted: {posted}"
                if result["Summary"] is not None:
                    s += f"\n{result['Summary']}"
        f = open(os.path.join("Data", f"{date}.txt"), "w", errors="ignore")
        f.write(s)
        f.close()
    if driver is not None:
        driver.quit()
    # Clean all articles if anything was missed during the summarizing process.
    files = os.listdir("Data")
    for file in files:
        if not file.endswith(".txt"):
            continue
        file = os.path.join("Data", file)
        if not os.path.isfile(file):
            continue
        f = open(file, "r")
        s = f.read()
        f.close()
        # There should be no entries with blank titles.
        if s.__contains__("Title: \n"):
            file = os.path.splitext(os.path.basename(file))[0]
            print(f"{file} has blank entries.")
    for file in files:
        if not file.endswith(".txt"):
            continue
        file = os.path.join("Data", file)
        if not os.path.isfile(file):
            continue
        f = open(file, "r")
        s = f.read()
        f.close()
        file = os.path.splitext(os.path.basename(file))[0]
        # Nothing should have any "FALSE" entries for articles that are not relevant.
        count = s.count("FALSE") - 1
        if count > 0:
            print(f"{file} has {count} FALSE entries.")
        # Nothing should have "TRUE" written in the summaries.
        count = s.count("TRUE")
        if count > 0:
            print(f"{file} has {count} TRUE entries.")


def llm_forecast(prediction: int, history: list[float], forecast: int, date: str, terms: list[str] or None = None,
                 trusted: list[str] or None = None, clamp: int = 100, mask: bool = False, latest: int = 0) -> int:
    """
    Perform the LLM forecasting.
    :param prediction: The prediction made by the analytical model.
    :param history: Previous values.
    :param forecast: How far into the future to forecast.
    :param date: The date which is being forecast.
    :param terms: Terms to mask.
    :param trusted: List of trusted sources.
    :param clamp: By how much should forecast values be clamped around the baseline prediction.
    :param mask: If we should mask COVID-19 or not.
    :param latest: Up to how many latest weeks of data should we keep.
    :return: The value predicted by the LLM.
    """
    # If we already have the response for this date, load it.
    response = os.path.join("Responses", f"{'Masked' if mask else 'Unmasked'} {forecast} {date}.txt")
    if os.path.exists(response):
        f = open(response, "r")
        s = f.read()
        f.close()
    # Otherwise, query the LLM.
    else:
        # If the file does not exist, we cannot use the LLM.
        path = os.path.join("Data", f"{date}.txt")
        if not os.path.exists(path):
            return prediction
        # Build the prompt.
        virus = "RAPID-VIRUS" if mask else "COVID-19"
        timeframe = "week" if forecast == 1 else f"{forecast} weeks"
        prompt = (f"You are tasked with forecasting {virus} hospitalizations in Ontario, Canada you predict will occur "
                  f"over the next {timeframe}. You must respond with a single integer and nothing else.")
        if mask:
            prompt += " RAPID-VIRUS is a codename for a highly infectious disease."
        if len(history) == 0:
            prompt += (f" There is currently no data on this, thus we can assume there are currently no {virus} "
                       "hospitalizations in Ontario, Canada.")
        elif len(history) == 1:
            prompt += (f" Here is the number of {virus} hospitalizations in Ontario, Canada from the previous week:"
                       f" {history[0]}.")
        else:
            # Trim the history if we should.
            if 0 < latest < len(history):
                history = history[-latest:]
            prompt += (f" Here are the number of {virus} hospitalizations in Ontario, Canada from the previous "
                       f"{len(history)} weeks from oldest to newest: {history[0]}")
            for i in range(1, len(history)):
                prompt += f", {history[i]}"
            prompt += "."
        prompt += (f" An ARIMA forecasting model has predicted that over the next {timeframe}, there will be "
                   f"{prediction} {virus} hospitalizations in Ontario, Canada. ARIMA is known to react too slow to "
                   f"surges or drops. Based on the provided information and this knowledge of how ARIMA is with surges "
                   f"and drops, keep or adjust the ARIMA forecast to best forecast {virus} hospitalizations in "
                   f"Ontario, Canada.")
        # Load the summaries.
        f = open(path, "r")
        articles = f.read()
        f.close()
        # Mask terms.
        if mask and terms is not None:
            for term in terms:
                articles = articles.replace(term, virus)
        # Set trusted sources.
        has_trusted = False
        if trusted is not None:
            for publisher in trusted:
                original = f"Publisher: {publisher}"
                if original in articles:
                    articles = articles.replace(original, f"Publisher (TRUSTED): {publisher}")
                    has_trusted = True
        # If there was a trusted source, indicate its importance.
        if has_trusted:
            articles = articles.replace("to help guide you in making your decision.",
                                        "to help guide you in making your decision. Highly reliable news source"
                                        'publishers have been flagged as "TRUSTED". Consider their information as more '
                                        "reliable. However, you should still consider the reports from other sources.")
        # If there are summaries, add them.
        if articles != "":
            prompt = f"{prompt} {articles}"
        # Get the response.
        s = generate(prompt, False, True)
        # Save the response for future lookups.
        if not os.path.exists("Responses"):
            os.mkdir("Responses")
        f = open(response, "w", errors="ignore")
        f.write(s)
        f.close()
    # We need to ensure there is only one value, as otherwise the LLM either could not or failed to do what was asked.
    words = s.split()
    if len(words) > 1:
        return prediction
    # noinspection PyBroadException
    try:
        parsed = int(words[0].replace(",", ""))
        # Clamp the value if it has been passed.
        if clamp > 0:
            clamp *= forecast
            parsed = max(prediction - clamp, min(parsed, prediction + clamp))
        return parsed
    except:
        # Return the baseline prediction if the LLM did not return a prediction.
        return prediction


def actual(dataset: pandas.DataFrame) -> pandas.DataFrame:
    """
    Get the actual values for hospitalizations.
    :param dataset: The dataset to test on.
    :return: The actual values for hospitalizations.
    """
    # Nothing to do if this has already been determined.
    path = os.path.join("Results", "Actual.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print("Determining actual amounts.")
    total = len(dataset)
    periods = total - 1
    dates = dataset["Date"]
    hospitalizations = dataset["Hospitalizations"]
    s = "Date"
    for i in range(periods):
        s += f",{i + 1}"
    # Loop for all date ranges that let us forecasting that far into the future.
    for i in range(periods):
        # Look at future periods to determine the real amount.
        real = 0
        s += f"\n{dates[i]}"
        for j in range(periods):
            index = i + j + 1
            # If this period cannot be forecast as it extends beyond the data, then list it as -1.
            if index > periods:
                s += ",-1"
                continue
            real += hospitalizations[index]
            s += f",{real}"
    # Write to a CSV file.
    if not os.path.exists("Results"):
        os.mkdir("Results")
    f = open(path, "w")
    f.write(s)
    f.close()
    return pd.read_csv(path)


def baseline(dataset: pandas.DataFrame) -> pandas.DataFrame:
    """
    Get the baseline analytical predictions.
    :param dataset: The dataset to test on.
    :return: The baseline analytical predictions.
    """
    # Nothing to do if this has already been determined.
    path = os.path.join("Results", "Baseline.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    total = len(dataset)
    periods = total - 1
    dates = dataset["Date"]
    hospitalizations = dataset["Hospitalizations"]
    s = "Date"
    for i in range(periods):
        s += f",{i + 1}"
    # Loop for all date ranges that let us forecasting that far into the future.
    history = []
    for i in range(periods):
        print(f"Baseline forecasting for period {i + 1} of {periods}")
        # Add this instance to our forecasting history.
        history.append(hospitalizations[i])
        # If there is more than one value, we can fit a forecasting model.
        if len(history) > 1:
            # Store the best ARIMA model.
            best = None
            # Try p values from 0 to 5.
            p = range(0, 6)
            # Try d values from 0 to 1.
            d = range(0, 2)
            # Try q values from 0 to 2.
            q = range(0, 3)
            # Generate all different combinations of p, d, q
            pdq_combinations = list(itertools.product(p, d, q))
            # Store the results
            best_score = float("inf")
            # Suppress convergence warnings for the ARIMA.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Loop through all combinations and evaluate each ARIMA model.
            for pdq in pdq_combinations:
                # noinspection PyBroadException
                try:
                    # Fit the ARIMA model.
                    model = ARIMA(history, order=pdq).fit()
                    # Get the score.
                    score = model.aic()
                    # If the score is lower, update the best model
                    if score < best_score:
                        best_score = score
                        best = model
                except Exception:
                    # Skip invalid models.
                    continue
            # If ARIMA failed to fit, use EXP smoothing.
            if best is None:
                best = SimpleExpSmoothing(history).fit(optimized=True)
            predictions = best.forecast(steps=total)
        else:
            # Otherwise, there is only one data point, so we just need to use this for all values.
            predictions = []
            for j in range(total):
                predictions.append(history[0])
        s += f"\n{dates[i]}"
        # Determine how much has been predicted.
        forecast = 0
        for j in range(periods):
            index = i + j + 1
            if index > periods:
                # If this period cannot be forecast as it extends beyond the data, then list it as -1.
                s += ",-1"
                continue
            # Add the prediction.
            forecast += 0 if math.isnan(predictions[index]) else predictions[index]
            s += f",{int(forecast)}"
    # Write to a CSV file.
    if not os.path.exists("Results"):
        os.mkdir("Results")
    f = open(path, "w")
    f.write(s)
    f.close()
    return pd.read_csv(path)


def llm(dataset: pandas.DataFrame, dataset_baseline: pandas.DataFrame, forecast: int = 0,
        clamp: int = 100, mask: bool = False, latest: int = 0) -> pandas.DataFrame:
    """
    Get the LLM predictions.
    :param dataset: The dataset to test on.
    :param dataset_baseline: The baseline analytical predictions.
    :param forecast: How many weeks in advance should the LLM model forecast.
    :param clamp: By how much should forecast values be clamped around the baseline prediction.
    :param mask: If we should mask COVID-19 or not.
    :param latest: Up to how many latest weeks of data should we keep.
    :return: The LLM predictions.
    """
    # Ensure all articles exist.
    initialize()
    prepare_articles(dataset)
    dates = dataset_baseline["Date"]
    periods = len(dates)
    # If no forecast was given or trying to forecast too far into the future, forecast everything.
    if forecast > periods or forecast < 1:
        forecast = periods
    # Load terms.
    path = os.path.join(os.getcwd(), "Terms.txx")
    if os.path.exists(path):
        with open(path, "r") as file:
            terms = [line.strip() for line in file]
    else:
        terms = None
    # Load trusted sources.
    trusted = load_trusted()
    history = []
    s = "Date"
    for i in range(forecast):
        s += f",{i + 1}"
    title = "Masked" if mask else "Unmasked"
    # Forecast for every period.
    for i in range(periods):
        print(f"{title} model forecasting for period {i + 1} of {periods}.")
        history.append(dataset["Hospitalizations"][i])
        s += f"\n{dates[i]}"
        for j in range(forecast):
            index = j + 1
            # If the baseline model did not fit for this, then neither will the LLM.
            current_baseline = dataset_baseline[f'{index}'][i]
            if current_baseline < 0:
                s += ",-1"
                continue
            s += f",{llm_forecast(current_baseline, history, index, dates[i], terms, trusted, clamp, mask, latest)}"
    # Save the data and return it.
    path = os.path.join("Results", f"{title}.csv")
    f = open(path, "w")
    f.write(s)
    f.close()
    return pd.read_csv(path)


def metrics(title: str, dataset_results: pandas.DataFrame, dataset_actual: pandas.DataFrame) -> None:
    """
    Get the metrics for a model compared to the actual values.
    :param title: The title of the model for which the metrics are of.
    :param dataset_results: The predictions of the model.
    :param dataset_actual: The actual values for hospitalizations.
    :return: Nothing.
    """
    # The results may not extend as far as the actual data, so ensure we only go as far as needed.
    columns = len(dataset_results.columns.tolist()) - 1
    # Start off the difference and failures.
    diff = "Date"
    for i in range(columns):
        diff += f",{i + 1}"
    # Get the data for every date.
    dates = dataset_results["Date"]
    for i in range(len(dates)):
        diff += f"\n{dates[i]}"
        # Get the results for all forecasting windows and add them to the documents.
        for j in range(columns):
            index = f"{j + 1}"
            diff += f",{dataset_results[index][i] - dataset_actual[index][i]}"
    # Save the data for the differences and failures.
    f = open(os.path.join("Results", f"Difference {title}.csv"), "w")
    f.write(diff)
    f.close()


def interval_score(true: float, lower: float, upper: float, alpha: float = 0.05) -> float:
    """
    Compute the interval score for a single prediction interval.
    :param true: A true value.
    :param lower: A lower value.
    :param upper: An upper value.
    :param alpha: The alpha factor.
    :return: The interval score for this instance.
    """
    if true < lower:
        return (upper - lower) + (2 / alpha) * (lower - true)
    elif true > upper:
        return (upper - lower) + (2 / alpha) * (true - upper)
    else:
        return upper - lower


def calculate_scores(index: str, dataset_actual: pandas.DataFrame,
                     dataset_diff: pandas.DataFrame, alpha: float = 0.95) -> (float, float, int, int, float, float):
    """
    Calculate the success rate, average difference, total failures, and total excess.
    :param index: The forecasting index in the datasets.
    :param dataset_actual: The actual results.
    :param dataset_diff: The differences the model had.
    :param alpha: The alpha factor for the desired confidence level which by default is 95%.
    :return: The success rate, average difference, total failures, total excess, WIS, MAE, and coverage.
    """
    total = 0
    successes = 0
    failures = 0
    excess = 0
    # Build lists to store values we will be scoring.
    true = []
    pred = []
    # Loop through all possible entries.
    entries = len(dataset_actual[index])
    for i in range(entries):
        # If this entry does not exist in the actual data, stop here.
        if dataset_actual[index][i] < 0:
            break
        # Increment the total number of entries.
        total += 1
        # Append values for later calculation.
        true.append(dataset_actual[index][i])
        pred.append(dataset_actual[index][i] + dataset_diff[index][i])
        # If the hospitalizations were met, this is a success.
        if dataset_diff[index][i] >= 0:
            successes += 1
            # Check how much excess there was.
            excess += dataset_diff[index][i]
        else:
            # Otherwise, it was a failure, so see by how much it failed.
            failures -= dataset_diff[index][i]
    # Ensure values are numpy arrays.
    true = np.array(true)
    pred = np.array(pred)
    # Calculate residuals.
    residuals = true - pred
    # Estimate mean standard deviation of residuals.
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    # Get the z-score.
    z_score = norm.ppf((1 + alpha) / 2)
    # Generate prediction intervals.
    lower = [p + mean_residual - z_score * std_residual for p in pred]
    upper = [p + mean_residual + z_score * std_residual for p in pred]
    # Calculate the weighted interval score.
    wis = np.mean([interval_score(obs, low, up, 1 - alpha) for obs, low, up in zip(true, lower, upper)])
    # Calculate the coverage score.
    within_interval = [l <= t <= u for t, l, u in zip(true, lower, upper)]
    coverage = sum(within_interval) / len(true) * 100
    return (max(successes / total * 100, 0), dataset_diff[index].mean(), failures, excess, wis,
            mean_absolute_error(true, pred), coverage)


def evaluate(dataset_actual: pandas.DataFrame, dataset_baseline: pandas.DataFrame,
             dataset_unmasked: pandas.DataFrame, dataset_masked: pandas.DataFrame, width: float = 5, height: float = 4,
             decimals: int = 2, alpha: float = 0.95) -> None:
    """
    Get the metrics for both the baseline and LLM models and plot the results.
    :param dataset_actual: The actual values for hospitalizations.
    :param dataset_baseline: The baseline analytical predictions.
    :param dataset_unmasked: The LLM predictions.
    :param dataset_masked: The masked LLM predictions.
    :param width: How wide figures should be.
    :param height: How tall figures should be.
    :param decimals: The many decimal spaces final results should be saved to.
    :param alpha: The alpha factor for the desired confidence level which by default is 95%.
    :return: Nothing.
    """
    # Ensure values are valid.
    if width <= 0:
        width = 5
    if height <= 0:
        height = 4
    if decimals < 0:
        decimals = 0
    # Compute the metrics for the baseline and LLM models.
    metrics("Baseline", dataset_baseline, dataset_actual)
    metrics("Unmasked", dataset_unmasked, dataset_actual)
    metrics("Masked", dataset_masked, dataset_actual)
    # Get average scores for each model.
    dataset_baseline_diff = pd.read_csv(os.path.join("Results", "Difference Baseline.csv"))
    dataset_unmasked_diff = pd.read_csv(os.path.join("Results", "Difference Unmasked.csv"))
    dataset_masked_diff = pd.read_csv(os.path.join("Results", "Difference Masked.csv"))
    # Create the headers.
    success_rate = "Forecast,Baseline,Unmasked,Masked"
    average_diff = "Forecast,Baseline,Unmasked,Masked"
    total_failures = "Forecast,Baseline,Unmasked,Masked"
    total_excess = "Forecast,Baseline,Unmasked,Masked"
    wis = "Forecast,Baseline,Unmasked,Masked"
    mae = "Forecast,Baseline,Unmasked,Masked"
    coverage = "Forecast,Baseline,Unmasked,Masked"
    # Loop through every column, calculating the metrics.
    columns_baseline = len(dataset_baseline.columns.tolist()) - 1
    columns_unmasked = len(dataset_unmasked.columns.tolist()) - 1
    columns_masked = len(dataset_unmasked.columns.tolist()) - 1
    columns = max(columns_baseline, columns_unmasked, columns_masked)
    for i in range(columns):
        index = f"{i + 1}"
        # Baseline metrics.
        if i < columns_baseline:
            b_s, b_d, b_f, b_e, b_wis, b_mae, b_coverage = calculate_scores(index, dataset_actual,
                                                                            dataset_baseline_diff, alpha)
            b_s = f"{b_s:.{decimals}f}%"
            b_d = f"{b_d:.{decimals}f}"
            b_f = f"{b_f:.{decimals}f}"
            b_e = f"{b_e:.{decimals}f}"
            b_wis = f"{b_wis:.{decimals}f}"
            b_mae = f"{b_mae:.{decimals}f}"
            b_coverage = f"{b_coverage:.{decimals}f}"
        else:
            b_s = ""
            b_d = ""
            b_f = ""
            b_e = ""
            b_wis = ""
            b_mae = ""
            b_coverage = ""
        # Unmasked LLM metrics.
        if i < columns_unmasked:
            n_s, n_d, n_f, n_e, n_wis, n_mae, n_coverage = calculate_scores(index, dataset_actual,
                                                                            dataset_unmasked_diff, alpha)
            n_s = f"{n_s:.{decimals}f}%"
            n_d = f"{n_d:.{decimals}f}"
            n_f = f"{n_f:.{decimals}f}"
            n_e = f"{n_e:.{decimals}f}"
            n_wis = f"{n_wis:.{decimals}f}"
            n_mae = f"{n_mae:.{decimals}f}"
            n_coverage = f"{n_coverage:.{decimals}f}%"
        else:
            n_s = ""
            n_d = ""
            n_f = ""
            n_e = ""
            n_wis = ""
            n_mae = ""
            n_coverage = ""
        # Masked LLM metrics.
        if i < columns_unmasked:
            m_s, m_d, m_f, m_e, m_wis, m_mae, m_coverage = calculate_scores(index, dataset_actual, dataset_masked_diff,
                                                                            alpha)
            m_s = f"{m_s:.{decimals}f}%"
            m_d = f"{m_d:.{decimals}f}"
            m_f = f"{m_f:.{decimals}f}"
            m_e = f"{m_e:.{decimals}f}"
            m_wis = f"{m_wis:.{decimals}f}"
            m_mae = f"{m_mae:.{decimals}f}"
            m_coverage = f"{m_coverage:.{decimals}f}%"
        else:
            m_s = ""
            m_d = ""
            m_f = ""
            m_e = ""
            m_wis = ""
            m_mae = ""
            m_coverage = ""
        # Add to the data to be written.
        success_rate += f"\n{index},{b_s},{n_s},{m_s}"
        average_diff += f"\n{index},{b_d},{n_d},{m_d}"
        total_failures += f"\n{index},{b_f},{n_f},{m_f}"
        total_excess += f"\n{index},{b_e},{n_e},{m_e}"
        wis += f"\n{index},{b_wis},{n_wis},{m_wis}"
        mae += f"\n{index},{b_mae},{n_mae},{m_mae}"
        coverage += f"\n{index},{b_coverage},{n_coverage},{m_coverage}%"
    # Save the data for the success rate and average differences.
    f = open(os.path.join("Results", "Success Rate.csv"), "w")
    f.write(success_rate)
    f.close()
    f = open(os.path.join("Results", "Average Difference.csv"), "w")
    f.write(average_diff)
    f.close()
    f = open(os.path.join("Results", "Total Failures.csv"), "w")
    f.write(total_failures)
    f.close()
    f = open(os.path.join("Results", "Total Excess.csv"), "w")
    f.write(total_excess)
    f.close()
    f = open(os.path.join("Results", "WIS.csv"), "w")
    f.write(wis)
    f.close()
    f = open(os.path.join("Results", "MAE.csv"), "w")
    f.write(mae)
    f.close()
    f = open(os.path.join("Results", "Coverage.csv"), "w")
    f.write(coverage)
    f.close()
    # Create plots for each of the forecasted periods.
    total = len(dataset_actual["Date"])
    for i in range(columns):
        # Store the results.
        points_actual = []
        points_baseline = []
        points_unmasked = []
        points_masked = []
        index = f"{i + 1}"
        # At most, we can check every entry.
        for j in range(total):
            # However, we will hit non-existent items, so stop there.
            if dataset_actual[index][j] < 0:
                break
            # Add valid points for each.
            points_actual.append(dataset_actual[index][j])
            if i < columns_baseline:
                points_baseline.append(dataset_baseline[index][j])
            if i < columns_unmasked:
                points_unmasked.append(dataset_unmasked[index][j])
            if i < columns_masked:
                points_masked.append(dataset_masked[index][j])
        # Configure the plot.
        if len(points_baseline) < 2 and len(points_unmasked) < 2 and len(points_masked) < 2:
            continue
        fig = plt.figure(figsize=(width, height))
        word = inflect.engine().number_to_words(i + 1)
        week = "week" if i < 1 else f"weeks"
        plt.title(f"Forecasting {word} {week}")
        plt.xlabel("Week")
        plt.ylabel(f"COVID-19 hospitalizations in the next {week if i < 1 else f'{word} {week}'}")
        # Plot all three values.
        plt.plot(points_actual, color="red", label="Actual")
        if len(points_baseline) > 0:
            plt.plot(points_baseline, color="blue", label="Baseline", alpha=0.75)
        if len(points_unmasked) > 0:
            plt.plot(points_unmasked, color="green", label="Unmasked", alpha=0.75)
        if len(points_masked) > 0:
            plt.plot(points_masked, color="yellow", label="Masked", alpha=0.75)
        plt.xlim(0, len(points_actual) - 1)
        bottom, top = plt.ylim()
        plt.ylim(0, top)
        plt.legend()
        plt.tight_layout()
        # Save the plot.
        fig.savefig(os.path.join("Results", f"{index}.png"))
        pyplot.close(fig)


def update_articles(old: str, new: str) -> None:
    """
    Update any existing articles.
    :param old: An old value in the text to replace.
    :param new: The new value to replace in the text.
    :return: Nothing.
    """
    for file in os.listdir("Data"):
        if not file.endswith(".txt"):
            continue
        file = os.path.join("Data", file)
        f = open(file, "r")
        s = f.read()
        f.close()
        s = s.replace(old, new)
        f = open(file, "w")
        f.write(s)
        f.close()


def main(forecast: int = 0, width: float = 5, height: float = 4, decimals: int = 2, alpha: float = 0.95,
         clamp: int = 100, latest: int = 0) -> None:
    """
    Run all required code.
    :param forecast: How many weeks in advance should the LLM model forecast.
    :param width: How wide figures should be.
    :param height: How tall figures should be.
    :param decimals: The many decimal spaces final results should be saved to.
    :param alpha: The alpha factor for the desired confidence level which by default is 95%.
    :param clamp: By how much should forecast values be clamped around the baseline prediction.
    :param latest: Up to how many latest weeks of data should we keep.
    :return: Nothing.
    """
    # Get the initial dataset.
    dataset = load()
    # Get the baseline results for the LLM model to use.
    dataset_baseline = baseline(dataset)
    # Evaluate both models and make plots.
    evaluate(actual(dataset), dataset_baseline, llm(dataset, dataset_baseline, forecast, clamp, False, latest),
             llm(dataset, dataset_baseline, forecast, clamp, True, latest), width, height, decimals, alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Forecast")
    parser.add_argument("-f", "--forecast", type=int, default=12, help="Number of weeks to forecast.")
    parser.add_argument("-w", "--width", type=float, default=5, help="The width of the figures.")
    parser.add_argument("-t", "--height", type=float, default=4, help="The height of the figures.")
    parser.add_argument("-d", "--decimals", type=int, default=2, help="The number of decimal spaces.")
    parser.add_argument("-a", "--alpha", type=float, default=0.95, help="The alpha factor for the desired confidence "
                                                                        "level which by default is 95%.")
    parser.add_argument("-c", "--clamp", type=int, default=100, help="By how much should forecast values be clamped "
                                                                     "around the baseline prediction.")
    parser.add_argument("-l", "--latest", type=int, default=0, help="Up to how many latest weeks of data should we "
                                                                    "keep.")
    args = parser.parse_args()
    main(args.forecast, args.width, args.height, args.decimals, args.alpha, args.clamp, args.latest)
