import datetime
import math
import os
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
    message = message.replace("\u2011", '-')
    return message


def hugging_face() -> hugchat.ChatBot or None:
    """
    Try to use HuggingChat.
    :return: The hugging Chat interface, otherwise nothing.
    """
    if os.path.exists("hugging_face.txt"):
        f = open("hugging_face.txt", "r")
        s = f.read()
        f.close()
        s = s.split()
        if len(s) > 1:
            # noinspection PyBroadException
            try:
                sign = Login(s[0], s[1])
                cookies = sign.login(cookie_dir_path="./cookies/", save_cookies=True)
                return hugchat.ChatBot(cookies=cookies.get_dict())
            except:
                pass
    return None


def chat(prompt: str, hugging_chat: hugchat.ChatBot or None = None, model: str = "gpt-3.5",
         max_length: int = 16000) -> str:
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    if hugging_chat is None:
        hugging_chat = hugging_face()
    message = None
    if hugging_chat is None:
        hugging_chat = hugging_face()
    if hugging_chat is not None:
        # noinspection PyBroadException
        try:
            hugging_chat.new_conversation(switch_to=True)
            result = hugging_chat.chat(prompt)
            message = result.wait_until_done()
            hugging_chat.delete_all_conversations()
        except:
            message = None
    if message is None:
        if model is None or model not in ["gpt-3.5", "claude-3-haiku", "llama-3-70b", "mixtral-8x7b"]:
            model = "gpt-3.5"
        # noinspection PyBroadException
        try:
            message = DDGS().chat(prompt, model=model)
        except:
            message = None
    if message is None:
        return ""
    message = message.replace("\r", " ")
    message = message.replace("\n", " ")
    message = message.replace("\t", " ")
    message = message.replace("*", "")
    message = message.replace("#", "")
    message = message.replace("- ", "")
    message = message.replace("> ", "")
    message = message.replace("`", "")
    message = ''.join(c for c in message if unicodedata.category(c) in {"Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Nl", "No",
                                                                        "Zs", "Zl", "Zp", "Pc", "Pd", "Ps", "Pe", "Pi",
                                                                        "Pf", "Po", "Sm", "Sc", "Sk", "So"})
    message = valid_encoding(message)
    while message.__contains__("  "):
        message = message.replace("  ", " ")
    return message.strip()


def get_article(result: dict, driver, trusted: list, forecasting: str = "COVID-19 hospitalizations",
                delay: float = 0, summarize: bool = True, model: str = "gpt-3.5",
                hugging_chat: hugchat.ChatBot or None = None) -> dict or None:
    publisher = result["publisher"]["title"]
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
        summary = article.text.replace("\r", "\n")
        summary = summary.replace("\t", " ")
        while summary.__contains__("\n\n"):
            summary = summary.replace("\n\n", "\n")
        while summary.__contains__("  "):
            summary = summary.replace("  ", " ")
        summary = summary.strip()
        if delay > 0:
            time.sleep(delay)
        # Summarize the summary with an LLM if requested to.
        if summarize and summary != "":
            summary = chat(f"Summarize this article, including any important facts to help forecast "
                           f"{forecasting}: {summary}", hugging_chat, model)
            if delay > 0:
                time.sleep(delay)
    # If the full article cannot be downloaded or the summarization fails, use the initial news info.
    except:
        title = result["title"].replace(publisher, "").strip().strip("-").strip()
        summary = result["description"].replace(publisher, "").strip().strip("-").strip()
    # No point in having the summary if it is just equal to the title.
    if summary is not None:
        if (title == summary or summary == "" or summary.isspace() or summary.startswith("I'm sorry, ") or
                summary.startswith("I apologize, ")):
            summary = None
    title = valid_encoding(title)
    if title is None or title == "":
        return None
    if summary is not None:
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
                exclude_websites: list or None = None, trusted: list or None = None, model: str or None = None,
                delay: float = 0, summarize: bool = True, forecasting: str = "COVID-19 hospitalizations",
                folder: str = "COVID Ontario") -> str:
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
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :param summarize: Whether to summarize the results.
    :param forecasting: What is being forecast.
    :param folder: The name of the file to save the results.
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
    filename = f"{end_date.year}-{end_date.month}-{end_date.day}.txt"
    full = os.path.join("Data", "Articles", folder, filename)
    if os.path.exists(full):
        f = open(full, "r")
        s = f.read()
        f.close()
        return s
    # If there are no trusted sites, make an empty list for comparisons.
    if trusted is None:
        trusted = []
    hugging_chat = hugging_face()
    # Set up Firefox web driver to handle Google News URL redirects.
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service)
    # Format all results.
    formatted = []
    # If no keywords, get the top news or location news if the location was passed.
    if keywords is None:
        google_news = GNews(language=language, country=country, max_results=max_results,
                            exclude_websites=exclude_websites)
        if isinstance(location, str):
            results = google_news.get_news(location)
        else:
            results = google_news.get_top_news()
        for result in results:
            result = get_article(result, driver, trusted, forecasting, delay, summarize, model, hugging_chat)
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
                    result = get_article(result, driver, trusted, forecasting, delay, summarize, model, hugging_chat)
                    if result is not None:
                        formatted.append(result)
            if delay > 0:
                time.sleep(delay)
    # Close the web driver.
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
    if len(formatted) > 0:
        days = f" from the past {days} day{'s' if days > 1 else ''}" if days > 0 else ""
        if keywords is None:
            words = ""
        else:
            words = f" regarding {keywords[0]}"
            for i in range(1, len(keywords)):
                if i == len(keywords) - 1:
                    words += f"{',' if len(keywords) > 2 else ''} or {keywords[i]}"
                else:
                    words += f", {keywords[i]}"
        single = len(formatted) == 1
        if isinstance(location, str):
            location = f" in {location}"
        else:
            location = ""
        s += (f"Below {'is' if single else 'are'} {len(formatted)} news article{'' if single else 's'}{days}{words} to "
              f"help guide you in making your decision. Using your best judgement, take into consideration only the "
              f"articles that are most relevant for forecasting {forecasting}{location}. Articles that are from know "
              f"reputable sources have been flagged with \"Trusted: True\".")
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
            s += (f"\n\nArticle {i + 1} of {len(formatted)}\nTitle: {result['Title']}\nPublisher: {result['Publisher']}"
                  f"\nTrusted: {result['Trusted']}\nPosted: {posted}")
            if result["Summary"] is not None:
                s += f"\n{result['Summary']}"
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
                exclude_websites: list or None = None, trusted: list or None = None, model: str or None = None,
                delay: float = 0, summarize: bool = True, forecasting: str = "COVID-19 hospitalizations",
                folder: str = "COVID Ontario", units: str = "weeks", periods: int = 1, previous: list or None = None,
                prediction: int or None = None) -> int:
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
    :param delay: How much to delay web queries by to ensure we do not hit limits.
    :param summarize: Whether to summarize the results.
    :param forecasting: What is being forecast.
    :param folder: The name of the file to save the results.
    :param units: The units of predictions.
    :param periods: The number of periods to predict.
    :param previous: Previous values to help predict.
    :param prediction: A guide to help predict.
    :return: The news articles.
    """
    articles = search_news(keywords, max_results, language, country, location, end_date, days, exclude_websites,
                           trusted, model, delay, summarize, forecasting, folder)
    if isinstance(location, str):
        location = f" in {location}"
    else:
        location = ""
    if periods < 1:
        periods = 1
    if periods == 1:
        if units.endswith("s"):
            forecast_units = units[:-1]
        else:
            forecast_units = units
    else:
        forecast_units = f"{periods} {units}"
    s = (f"You are tasked with forecasting {forecasting} you predict will occur over the next {forecast_units}"
         f"{location}. You are to respond with a single integer and nothing else.")
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
    if isinstance(prediction, int):
        s += (f" An analytical forecasting model has predicted that over the next {forecast_units}, there will be "
              f"{prediction} {forecasting}{location}. Using your best judgement, you may choose to keep this value or "
              f"adjust it.")
    s = chat(f"{s} {articles}", model=model)
    s = s.split()
    predictions = []
    for p in s:
        # noinspection PyBroadException
        try:
            p = int(p)
            predictions.append(p)
        except:
            # noinspection PyBroadException
            try:
                p = math.ceil(float(p))
                predictions.append(p)
            except:
                pass
    if len(predictions) < 1:
        if prediction is None:
            if isinstance(previous, list) and len(previous) > 0:
                return previous[-1]
            return 0
        else:
            return prediction
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
                     model: str or None = None, delay: float = 0, summarize: bool = True,
                     forecasting: str = "COVID-19 hospitalizations", max_length: int = 16000) -> None:
    dates = parse_dates(file)
    folder = os.path.splitext(os.path.basename(file))[0]
    path = os.path.join("Data", "Articles", folder)
    if not os.path.exists(path):
        return None
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
    for i in range(len(dates)):
        print(f"Preparing articles for time period {i + 1} of {len(dates)}.")
        search_news(keywords, max_results, language, country, location, dates[i], days, exclude_websites, trusted,
                    model, delay, summarize, forecasting, folder)
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path, file)
        if not os.path.isfile(file):
            continue
        f = open(file, "r")
        s = f.read()
        f.close()
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
        if len(s) > max_length:
            file = os.path.splitext(os.path.basename(file))[0]
            print(f"{file} --> {len(s)} > {max_length}.")
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
         "Wexner Medical Center - The Ohio State University", "Yale School of Medicine", "hss.gov.nt.ca"]
    prepare_articles("Data/Dates/COVID Ontario.txt", trusted=t, delay=5, max_length=25000)
