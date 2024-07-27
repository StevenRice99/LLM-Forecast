import datetime
import math
import os
import re
import time

import unicodedata
from duckduckgo_search import DDGS
from gnews import GNews
from hugchat import hugchat
from hugchat.login import Login
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


def valid_encoding(message: str) -> str:
    """
    Ensure certain illegal characters are replaced.
    :param message: The message to ensure is valid.
    :return: The string with illegal characters replaced.
    """
    # "\u2011" type of dash that cannot be encoded, so replace it with a "-"
    return message.replace("\u2011", '-')


def hugging_face() -> hugchat.ChatBot or None:
    """
    Try to use HuggingChat.
    :return: The hugging Chat interface, otherwise nothing.
    """
    # Nothing to do if there is no login file.
    if os.path.exists("hugging_face.txt"):
        f = open("hugging_face.txt", "r")
        s = f.read()
        f.close()
        # Split the contents as the username/email and password should each be on a line.
        s = s.split()
        # If not enough lines, return.
        if len(s) > 1:
            # noinspection PyBroadException
            try:
                # Login and create the HuggingChat instance.
                sign = Login(s[0], s[1])
                cookies = sign.login(cookie_dir_path="./cookies/", save_cookies=True)
                return hugchat.ChatBot(cookies=cookies.get_dict())
            except:
                pass
    return None


def chat(prompt: str, hugging_chat: hugchat.ChatBot or None = None, model: str or list = "Meta-Llama-3.1-405B",
         attempts: int = 10, delay: float = 0) -> str:
    """
    Chat with a large language model.
    :param prompt: The prompt to
    :param hugging_chat: HuggingChat instance to use.
    :param model: Which model to use for LLM summaries.
    :param attempts: The number of times to attempt LLM summarization.
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :return: The response from the LLM.
    """
    # Both HuggingChat and DuckDuckGo AI Chat have an input limit of 16,000 characters.
    if len(prompt) > 16000:
        # Using "..." may help the model know the input was intentionally cut off.
        prompt = prompt[:15997] + "..."
    if isinstance(model, str):
        model = [model]
    if hugging_chat is None:
        hugging_chat = hugging_face()
    message = None
    # Try to create a HuggingChat instance if it was not passed as one.
    if hugging_chat is None:
        hugging_chat = hugging_face()
    if hugging_chat is not None:
        # noinspection PyBroadException
        try:
            # Get all models on HuggingChat
            available = hugging_chat.get_available_llm_models()
            selected = -1
            # Find a model which matches the ideal name.
            for i in range(len(model)):
                for j in range(len(available)):
                    # If one is found, select this index.
                    if model[i] in str(available[j]):
                        selected = j
                        break
                if selected >= 0:
                    break
            # If none was found, use the first model.
            if selected < 0:
                selected = 0
            # Try for the given number of attempts.
            for i in range(attempts):
                # noinspection PyBroadException
                try:
                    # CLEAR ALL PAST CONVERSATION. You may wish to change this if you have ones you wish to save.
                    hugging_chat.delete_all_conversations()
                    # Use the selected model.
                    hugging_chat.new_conversation(modelIndex=selected, switch_to=True)
                    # Prompt and wait until done.
                    result = hugging_chat.chat(prompt)
                    message = result.wait_until_done()
                    # CLEAR ALL PAST CONVERSATION. You may wish to change this if you have ones you wish to save.
                    hugging_chat.delete_all_conversations()
                    # If a message was received, stop.
                    if message != "":
                        break
                except:
                    pass
                # Wait if a delay is given.
                if delay > 0:
                    time.sleep(delay)
        except:
            pass
    # If HuggingChat did not work, try DuckDuckGo AI Chat.
    if message is None:
        # Find a model for DuckDuckGo AI Chat.
        m = None
        for i in range(len(model)):
            if model[i] in ["gpt-3.5", "claude-3-haiku", "llama-3-70b", "mixtral-8x7b"]:
                m = model[i]
                break
        if m is None:
            m = "gpt-3.5"
        # noinspection PyBroadException
        try:
            # Request a response from DuckDuckGo AI Chat.
            message = DDGS().chat(prompt, model=m)
        except:
            message = None
    # If nothing succeeded, return an empty string.
    if message is None:
        return ""
    # Replace all whitespace with spaces.
    message = re.sub(r'\s+', ' ', message)
    # Remove Markdown formatting.
    message = message.replace("#", "")
    message = message.replace("- ", "")
    message = message.replace("> ", "")
    message = message.replace("`", "")
    # Remove characters with invalid encodings.
    message = valid_encoding(message)
    message = ''.join(c for c in message if unicodedata.category(c) in {"Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Nl", "No",
                                                                        "Zs", "Zl", "Zp", "Pc", "Pd", "Ps", "Pe", "Pi",
                                                                        "Pf", "Po", "Sm", "Sc", "Sk", "So"})
    # Remove duplicate whitespaces.
    while message.__contains__("  "):
        message = message.replace("  ", " ")
    # Strip and return the message.
    return message.strip()


def get_article(result: dict, driver, trusted: list, forecasting: str = "COVID-19 hospitalizations",
                location: str or None = "Ontario, Canada", attempts: int = 10, delay: float = 0,
                model: str or list = "Meta-Llama-3.1-405B",
                hugging_chat: hugchat.ChatBot or None = None) -> dict or None:
    """

    :param result: The Google News result.
    :param driver: The Selenium webdriver.
    :param trusted: Publishers which are trusted.
    :param forecasting: What is being forecast.
    :param location: The location the forecasts are for.
    :param attempts: The number of times to attempt LLM summarization.
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :param model: Which model to use for LLM summaries.
    :param hugging_chat: HuggingChat instance to use.
    :return: The details of the article if it was relevant, otherwise nothing.
    """
    # Get the publisher and title.
    publisher = result["publisher"]["title"]
    title = result["title"].replace(publisher, "").strip().strip("-").strip()
    # If the article was broken to start, return.
    if title == "":
        return None
    # Try to get the full article and then summarize it with an LLM.
    # noinspection PyBroadException
    try:
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
        article.download()
        article.parse()
        title = article.title
        # Clean the summary.
        summary = re.sub(r"\s+", " ", article.text)
        while summary.__contains__("  "):
            summary = summary.replace("  ", " ")
        summary = summary.strip()
        if delay > 0:
            time.sleep(delay)
        # Summarize the summary with an LLM if requested to.
        if summary != "":
            prompt = (f"You are trying to help forecast {forecasting}{location}. Below is an article. If the article is"
                      f" not relevant for forecasting {forecasting}{location}, respond with \"FALSE\". Otherwise, if "
                      f"the article is relevant for forecasting {forecasting}{location}, respond with a brief summary, "
                      f"highlighting values most important for forecasting {forecasting}{location}:\n\n{summary}")
            summary = chat(prompt, hugging_chat, model, attempts, delay)
            # The LLM tends to respond with a variation of TRUE which is not needed so remove it.
            summary = summary.replace("TRUE: ", "")
            summary = summary.replace("TRUE. ", "")
            summary = summary.replace("TRUE:", "")
            summary = summary.replace("TRUE.", "")
            summary = summary.replace("TRUE:", "")
            summary = summary.replace("TRUE", "")
            summary = summary.strip()
            if delay > 0:
                time.sleep(delay)
    # If the full article cannot be downloaded or the summarization fails, use the initial news info.
    except:
        return None
    # No point in having the summary if it is just equal to the title.
    if title is None or title == "":
        return None
    if (summary is None or title == summary or summary == "" or summary.isspace() or summary.startswith("I'm sorry, ")
            or summary.startswith("I apologize, ") or summary.startswith("FALSE")):
        return None
    title = valid_encoding(title)
    summary = valid_encoding(summary)
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
    # Store the formatted data.
    return {"Title": title, "Summary": summary, "Publisher": publisher, "Year": int(published_date[3]),
            "Month": published_date[2], "Day": int(published_date[1]), "Hour": int(published_time[0]),
            "Minute": int(published_time[1]), "Second": int(published_time[2]), "Trusted": publisher in trusted}


def search_news(keywords: str or list or None = "COVID-19", max_results: int = 10, language: str = "en",
                country: str = "CA", location: str or None = "Ontario, Canada",
                end_date: tuple or datetime.datetime or None = None, days: int = 7,
                exclude_websites: list or None = None, trusted: list or None = None,
                model: str or list = "Meta-Llama-3.1-405B", attempts: int = 10, delay: float = 0,
                forecasting: str = "COVID-19 hospitalizations", folder: str = "COVID Ontario", driver=None,
                hugging_chat: hugchat.ChatBot or None = None) -> str:
    """
    Search the web for news.
    :param keywords: The keywords to search for.
    :param max_results: The maximum number of results to return.
    :param language: The language to search in.
    :param country: The Country to search in.
    :param location: Keyword location to search with.
    :param end_date: The latest date that news results can be from.
    :param days: How many days prior to the end date to search from.
    :param exclude_websites: Websites to exclude.
    :param trusted: What websites should be labelled as trusted.
    :param model: Which model to use for LLM summaries.
    :param attempts: The number of times to attempt LLM summarization.
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :param forecasting: What is being forecast.
    :param folder: The name of the file to save the results.
    :param driver: Selenium Firefox driver.
    :param hugging_chat: HuggingChat instance to use.
    :return: The news articles.
    """
    # Configure the time period if one should be used.
    if end_date is None:
        end_date = datetime.datetime.now()
    elif isinstance(end_date, tuple):
        end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])
    if days < 0:
        start_date = None
    else:
        start_date = end_date - datetime.timedelta(days=days)
    # Get the filepath.
    filename = f"{end_date.year}-{end_date.month}-{end_date.day}.txt"
    full = os.path.join("Data", "Articles", folder, filename)
    # If the file already exists, we can simply load it.
    if os.path.exists(full):
        f = open(full, "r")
        s = f.read()
        f.close()
        return s
    # If there are no trusted sites, make an empty list for comparisons.
    if trusted is None:
        trusted = []
    if hugging_chat is None:
        hugging_chat = hugging_face()
    # Set up Firefox web driver to handle Google News URL redirects.
    if driver is None:
        handle_driver = True
        driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
    else:
        handle_driver = False
    # Format all results.
    formatted = []
    if isinstance(location, str):
        location = f" in {location}"
    else:
        location = ""
    # If no keywords, get the top news or location news if the location was passed.
    if keywords is None:
        google_news = GNews(language=language, country=country, max_results=max_results,
                            exclude_websites=exclude_websites)
        if location != "":
            results = google_news.get_news(location)
        else:
            results = google_news.get_top_news()
        for result in results:
            result = get_article(result, driver, trusted, forecasting, location, attempts, delay, model, hugging_chat)
            if result is not None:
                formatted.append(result)
        if delay > 0:
            time.sleep(delay)
    # Otherwise, search by the keywords.
    else:
        google_news = GNews(language=language, country=country, start_date=start_date, end_date=end_date,
                            max_results=max_results, exclude_websites=exclude_websites)
        # Get the keywords as a list so each can be searched.
        if isinstance(keywords, str):
            keywords = keywords.split()
        # Store all results.
        results = []
        # Search for every keyword.
        for keyword in keywords:
            keyword_results = google_news.get_news(keyword)
            # Check all results for the keyword.
            for result in keyword_results:
                # If this URL has already been visited, skip it.
                match = False
                for existing in results:
                    if existing["url"] == result["url"]:
                        match = True
                        break
                # If this is a new result, format and append it.
                if not match:
                    result = get_article(result, driver, trusted, forecasting, location, attempts, delay, model,
                                         hugging_chat)
                    if result is not None:
                        formatted.append(result)
            if delay > 0:
                time.sleep(delay)
    # Close the web driver.
    if handle_driver:
        driver.quit()
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
        days = f" from the past {days} day{'s' if days > 1 else ''}" if days > 0 else ""
        single = len(formatted) == 1
        count = "a" if single else f"{len(formatted)}"
        s += (f"Below {'is' if single else 'are'} {count} news article{'' if single else 's'}{days} to help guide you "
              f"in making your decision. Articles that are from known reputable sources have been flagged with "
              f"\"Trusted: True\".")
        # Add every result.
        for i in range(len(formatted)):
            result = formatted[i]
            date = datetime.datetime(result["Year"], result["Month"], result["Day"])
            difference = end_date - date
            days = difference.days
            if days < 1:
                posted = "Today"
            else:
                posted = f"{days} day{'s' if days > 1 else ''} ago"
            s += "\n\n"
            if len(formatted) > 1:
                s += f"Article {i + 1} of {len(formatted)}\n"
            s += (f"Title: {result['Title']}\nPublisher: {result['Publisher']}\nTrusted: {result['Trusted']}"
                  f"\nPosted: {posted}")
            if result["Summary"] is not None:
                s += f"\n{result['Summary']}"
    # Save the data.
    if not os.path.exists("Data"):
        os.mkdir("Data")
    if os.path.exists("Data"):
        path = os.path.join("Data", "Articles")
        if not os.path.exists(path):
            os.mkdir(path)
        if os.path.exists(path):
            path = os.path.join(path, folder)
            if not os.path.exists(path):
                os.mkdir(path)
            if os.path.exists(path):
                # Ignore writing errors.
                f = open(os.path.join(path, filename), "w", errors="ignore")
                # noinspection PyBroadException
                try:
                    f.write(valid_encoding(s))
                    f.close()
                except:
                    # noinspection PyBroadException
                    try:
                        f.close()
                    except:
                        pass
                    os.remove(os.path.join(path, filename))
    return s


def llm_predict(keywords: str or list or None = "COVID-19", max_results: int = 10, language: str = "en",
                country: str = "CA", location: str or None = "Ontario, Canada",
                end_date: tuple or datetime.datetime or None = None, days: int = 7,
                exclude_websites: list or None = None, trusted: list or None = None,
                model: str or list = "Meta-Llama-3.1-405B", attempts: int = 10, delay: float = 0,
                forecasting: str = "COVID-19 hospitalizations", folder: str = "COVID Ontario", units: str = "weeks",
                periods: int = 1, previous: list or None = None, prediction: int or None = None,
                hugging_chat: hugchat.ChatBot or None = None) -> int:
    """
    Make a prediction with a LLM.
    :param keywords: The keywords to search for.
    :param max_results: The maximum number of results to return.
    :param language: The language to search in.
    :param country: The Country to search in.
    :param location: Keyword location to search with.
    :param end_date: The latest date that news results can be from.
    :param days: How many days prior to the end date to search from.
    :param exclude_websites: Websites to exclude.
    :param trusted: What websites should be labelled as trusted.
    :param model: Which model to use for LLM summaries.
    :param attempts: The number of times to attempt LLM summarization.
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :param forecasting: What is being forecast.
    :param folder: The name of the file to save the results.
    :param units: The units of predictions.
    :param periods: The number of periods to predict.
    :param previous: Previous values to help predict.
    :param prediction: A guide to help predict.
    :param hugging_chat: HuggingChat instance to use.
    :return: The news articles.
    """
    # Get the articles for the given period.
    articles = search_news(keywords, max_results, language, country, location, end_date, days, exclude_websites,
                           trusted, model, attempts, delay, forecasting, folder)
    # Format the location if one was given.
    if isinstance(location, str):
        location = f" in {location}"
    else:
        location = ""
    if periods < 1:
        periods = 1
    # If only one period, ensure there is no plural.
    if periods == 1:
        if units.endswith("s"):
            forecast_units = units[:-1]
        else:
            forecast_units = units
    else:
        forecast_units = f"{periods} {units}"
    # Begin building the prompt.
    s = (f"You are tasked with forecasting {forecasting} you predict will occur over the next {forecast_units}"
         f"{location}. You are to respond with a single integer and nothing else.")
    # Add previous values.
    if isinstance(previous, list) and len(previous) > 0:
        if len(previous) == 1:
            if units.endswith("s"):
                past_units = units[:-1]
            else:
                past_units = units
        else:
            past_units = f"{units} from oldest to newest"
        s += (f" Here {'are' if len(previous) > 1 else 'is'} the number of {forecasting}{location} over the past "
              f"{len(previous)} {past_units}: {previous[0]}")
        for i in range(1, len(previous)):
            s += f", {previous[i]}"
        s += "."
    # Add a prediction to help guide the LLM.
    if isinstance(prediction, int):
        s += (f" An analytical forecasting model has predicted that over the next {forecast_units}, there will be "
              f"{prediction} {forecasting}{location}. Using your best judgement, you may choose to keep this value or "
              f"adjust it.")
    # Get the result from the LLM.
    s = chat(f"{s} {articles}", hugging_chat, model, attempts, delay)
    if delay > 0:
        time.sleep(delay)
    # In case multiple values or non-numeric results were given, split and check all values.
    s = s.split()
    predictions = []
    # Try and parse every value into an integer.
    for p in s:
        # noinspection PyBroadException
        try:
            parsed = int(p)
            predictions.append(parsed)
        except:
            # noinspection PyBroadException
            try:
                parsed = math.ceil(float(p))
                predictions.append(parsed)
            except:
                pass
    # If no predictions were returned, determine what this method should return.
    if len(predictions) < 1:
        # If there was no prediction, return the most recent past result if some were passed or zero otherwise.
        if prediction is None:
            if isinstance(previous, list) and len(previous) > 0:
                return previous[-1]
            return 0
        # Return the initial prediction if there was one.
        else:
            return prediction
    # Return the greatest prediction if multiple were made.
    return max(predictions)


def parse_dates(file: str) -> list:
    """
    Parse all dates from a file.
    :param file: The file to get the dates from.
    :return: The parsed dates.
    """
    # Ensure the file exists.
    if not os.path.exists(file):
        return []
    # Read the file.
    with open(file, "r") as file:
        s = file.read()
    # Get every line.
    s = s.split("\n")
    dates = []
    # Convert every line into a date.
    for line in s:
        line = line.split("-")
        if len(line) < 3:
            continue
        dates.append(datetime.datetime(int(line[0]), int(line[1]), int(line[2])))
    # All dates.
    return dates


def prepare_articles(file: str, keywords: str or list or None = "COVID-19", max_results: int = 10,
                     language: str = "en", country: str = "CA", location: str or None = "Ontario, Canada",
                     days: int = 7, exclude_websites: list or None = None, trusted: list or None = None,
                     model: str or list = "Meta-Llama-3.1-405B", attempts: int = 10, delay: float = 0,
                     forecasting: str = "COVID-19 hospitalizations") -> None:
    """
    Prepare all article summarizations ahead of time.
    :param file: The file to get the dates from.
    :param keywords: The keywords to search for.
    :param max_results: The maximum number of results to return.
    :param language: The language to search in.
    :param country: The Country to search in.
    :param location: Keyword location to search with.
    :param days: How many days prior to the end date to search from.
    :param exclude_websites: Websites to exclude.
    :param trusted: What websites should be labelled as trusted.
    :param model: Which model to use for LLM summaries.
    :param attempts: The number of times to attempt LLM summarization.
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :param forecasting: What is being forecast.
    :return: Nothing.
    """
    # Get the dates.
    dates = parse_dates(file)
    # Determine the folder and ensure it exists.
    folder = os.path.splitext(os.path.basename(file))[0]
    if not os.path.exists("Data"):
        os.mkdir("Data")
    if not os.path.exists("Data"):
        return None
    path = os.path.join("Data", "Articles")
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path):
        return None
    path = os.path.join(path, folder)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path):
        return None
    # Apply any trusted publisher updates to existing results.
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path, file)
        if not os.path.isfile(file):
            continue
        f = open(file, "r")
        s = f.read()
        f.close()
        for publisher in trusted:
            core = f"Publisher: {publisher}\nTrusted: "
            s = s.replace(f"{core}False", f"{core}True")
        f = open(file, "w")
        f.write(s)
        f.close()
    # Load a HuggingChat instance.
    hugging_chat = hugging_face()
    # Load the Firefox webdriver.
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
    # Do all news summarizations.
    for i in range(len(dates)):
        print(f"Preparing articles for time period {i + 1} of {len(dates)}.")
        search_news(keywords, max_results, language, country, location, dates[i], days, exclude_websites, trusted,
                    model, attempts, delay, forecasting, folder, driver, hugging_chat)
    # Clean up the webdriver.
    driver.quit()
    # Clean all articles if anything was missed during the summarizing process.
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path, file)
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
        file = os.path.join(path, file)
        if not os.path.isfile(file):
            continue
        f = open(file, "r")
        s = f.read()
        f.close()
        # Give a warning if a file is too long.
        if len(s) > 16000:
            file = os.path.splitext(os.path.basename(file))[0]
            print(f"{file} too long.")
    for file in files:
        file = os.path.join(path, file)
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
    # Get and output all untrusted sources in case any should be changed to trusted.
    untrusted = []
    for file in files:
        file = os.path.join(path, file)
        if not os.path.isfile(file):
            continue
        f = open(file, "r")
        s = f.read()
        f.close()
        lines = s.split("\n")
        for i in range(len(lines)):
            if not lines[i].startswith("Publisher: ") or i + 1 >= len(lines) or lines[i + 1] == "Trusted: True":
                continue
            publisher = lines[i].split()
            if len(publisher) < 2:
                continue
            trimmed = publisher[1]
            for j in range(2, len(publisher)):
                trimmed += f" {publisher[j]}"
            if trimmed in trusted or trimmed in untrusted:
                continue
            untrusted.append(trimmed)
    untrusted.sort()
    print("Untrusted publishers:")
    for publisher in untrusted:
        print(publisher)


if __name__ == '__main__':
    t = ["CDC", "Canada.ca", "Statistique Canada", "AFP Factcheck", "World Health Organization (WHO)", "BC Gov News"
         "Doctors Without Borders", "Government of Nova Scotia", "Middlesex-London Health Unit", "GOV.UK", "FDA.gov",
         "Government of B.C.", "McGill University", "Wilfrid Laurier University", "Government of Ontario News",
         "Ontario COVID-19 Science Advisory Table", "CMAJ", "The White House", "University of Toronto", "Doctors of BC",
         "Instituts de recherche en santé du Canada", "National Institutes of Health (NIH) (.gov)", "Boston.gov",
         "Mental Health Commission of Canada", "Alberta Health Services", "Algonquin College", "Boston University",
         "BC Centre for Disease Control", "Bureau of Labor Statistics", "CAMH", "CMAJ Open", "Carleton Newsroom",
         "Canadian Medical Association", "Canadian Red Cross", "City of Calgary Newsroom", "Cornell Chronicle",
         "Faculty of Health Sciences | Queen's University", "Federal Reserve", "Folio - University of Alberta",
         "Gouvernement du Québec", "Government of Newfoundland and Labrador", "Government of Northwest Territories"
         "Government of Saskatchewan", "Government of Yukon", "Hamilton Health Sciences", "Harvard Gazette",
         "Harvard Health", "Heart and Stroke Foundation of Canada", "Institut national de santé publique du Québec",
         "JEMS (Journal of Emergency Medical Services)", "Johns Hopkins Bloomberg School of Public Health", "MIT News",
         "Johns Hopkins Medicine", "King's University College", "Liberal Party of Canada", "McGill Newsroom",
         "McGill Reporter", "McGill University Health Centre", "McMaster COVID-19", "Niagara Health", "UBC News",
         "McMaster Faculty of Health Sciences", "National Cancer Institute (.gov)", "Pan American Health Organization",
         "National Heart, Lung, and Blood Institute", "National Human Genome Research Institute", "Science", "UC Davis",
         "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)", "Prime Minister of Canada",
         "Queen's University", "Scientific American", "Senate of Canada", "Stanford Medical Center Report", "Statista",
         "U.S. Census Bureau", "UBC Faculty of Medicine", "UBC Okanagan News", "UC Davis Health", "UC San Diego Health",
         "UC San Francisco", "University of Alberta", "University of Calgary", "University of Guelph News", "The BMJ",
         "University of Kansas Medical Center", "University of Minnesota Twin Cities", "University of Utah Health Care",
         "University of Victoria", "University of Waterloo", "University of Winnipeg News", "Université de Montréal",
         "Washington University School of Medicine in St. Louis", "Western News", "Yale Medicine", "news.gov.mb.ca",
         "Wexner Medical Center - The Ohio State University", "Yale School of Medicine", "hss.gov.nt.ca",
         "Thunder Bay District Health Unit", "FactCheck.org", "BMC Public Health", "CDC Emergency Preparedness",
         "Simon Fraser University News"]
    prepare_articles("Data/Dates/COVID Ontario.txt", keywords=["COVID-19 Hospitalizations Ontario"], trusted=t, delay=5)
