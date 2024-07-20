import datetime
import os
import time

from duckduckgo_search import DDGS
from gnews import GNews
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


def search_news(keywords: str or list or None = None, max_results: int = 100, language: str = "en", country: str = "CA",
                location: str or None = None, end_date: tuple or datetime.datetime or None = None, days: int = 7,
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
    # Configure the news search.
    google_news = GNews(language=language, country=country, start_date=start_date, end_date=end_date,
                        max_results=max_results, exclude_websites=exclude_websites)
    # If no keywords, get the top news or location news if the location was passed.
    if keywords is None:
        if isinstance(location, str):
            results = google_news.get_news(location)
        else:
            results = google_news.get_top_news()
        if delay > 0:
            time.sleep(delay)
    # Otherwise, search by the keywords.
    else:
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
                # If this is a new result, append it.
                match = False
                for existing in results:
                    if existing["url"] == result["url"]:
                        match = True
                        break
                if not match:
                    results.append(result)
            if delay > 0:
                time.sleep(delay)
    # If there are no trusted sites, make an empty list for comparisons.
    if trusted is None:
        trusted = []
    # Ensure the LLM for summarizations is valid.
    if model is None or model not in ["gpt-3.5", "claude-3-haiku", "llama-3-70b", "mixtral-8x7b"]:
        model = "gpt-3.5"
    # Set up Firefox web driver to handle Google News URL redirects.
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service)
    # Store where the web driver has been.
    initial_url = "about:blank"
    visited_url = initial_url
    # Format all results.
    formatted = []
    for result in results:
        publisher = result["publisher"]["title"]
        # Try to get the full article and then summarize it with an LLM.
        # noinspection PyBroadException
        try:
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
                # Ensure we are not at the initial URL when loading the web driver.
                if current_url == initial_url:
                    continue
                # Ensure we are not at the previous URL from the last article.
                if current_url == visited_url:
                    continue
                # Ensure we are not still at the redirect URL.
                if current_url == redirect_url:
                    continue
                # Update the previously visited URL for future redirect handling.
                visited_url = current_url
                break
            # Download the final article.
            article = Article(visited_url)
            # noinspection PyBroadException
            try:
                article.download()
                article.parse()
                title = article.title
                # Clean the summary.
                summary = article.text.strip().replace("\r", "\n")
                while summary.__contains__("\n\n"):
                    summary = summary.replace("\n\n", "\n")
                if delay > 0:
                    time.sleep(delay)
                # Summarize the summary with an LLM if requested to.
                if summarize:
                    summary = DDGS().chat(f"Summarize this article: {summary}", model=model)
                    summary = summary.replace("\r", "\n")
                    while summary.__contains__("\n\n"):
                        summary = summary.replace("\n\n", "\n")
                    summary = summary.replace("\n", " ")
                    if delay > 0:
                        time.sleep(delay)
            except:
                # The article could not be downloaded, so skip it.
                continue
        # If the full article cannot be downloaded or the summarization fails, use the initial news info.
        except:
            title = result["title"].replace(publisher, "").strip().strip("-").strip()
            summary = result["Summary"].replace(publisher, "").strip().strip("-").strip()
        # No point in having the summary if it is just equal to the title.
        if summary is not None and (title == summary or summary == "" or summary.isspace()
                                    or summary == "I'm sorry, but I cannot access external content, including articles."
                                                  " If you provide me with the main points or key information from the "
                                                  "article, I'd be happy to help summarize it for you."):
            summary = None
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
        formatted.append({"Title": title, "Summary": summary, "Publisher": publisher,
                          "Year": int(published_date[3]), "Month": published_date[2],"Day": int(published_date[1]),
                          "Hour": int(published_time[0]), "Minute": int(published_time[1]),
                          "Second": int(published_time[2]), "Trusted": publisher in trusted})
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
        single = len(results) == 1
        s += (f"Below {'is' if single else 'are'} {len(results)} news article{'' if single else 's'}{days}{words} to "
              f"help guide you in making your decision. Using your best judgement, take into consideration only the "
              f"articles that are most relevant for forecasting {forecasting}{location}. Articles that are from know "
              f"reputable sources have been flagged with \"Trusted: True\".")
        # Add every result.
        for i in range(len(formatted)):
            result = formatted[i]
            date = datetime.datetime(result["Year"], result["Month"], result["Day"])
            difference = end_date - date
            days = difference.days
            if days < 0:
                days = 0
            s += (f"\n\nArticle {i + 1} of {len(formatted)}\nTitle: {result['Title']}\nPublisher: {result['Publisher']}"
                  f"\nPosted: {days} day{'s' if days > 1 else ''} ago\nTrusted: {result['Trusted']}")
            if result["Summary"] is not None:
                s += f"\nSummary: {result['Summary']}"
    # Simply output to the console for now.
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
                f = open(os.path.join(path, filename), "w")
                f.write(s)
                f.close()
    return s


def build_prompt(keywords: str or list or None = None, max_results: int = 100, language: str = "en",
                 country: str = "CA", location: str or None = None, end_date: tuple or datetime.datetime or None = None,
                 days: int = 7, exclude_websites: list or None = None, trusted: list or None = None,
                 model: str or None = None, delay: float = 0, summarize: bool = True,
                 forecasting: str = "COVID-19 hospitalizations", folder: str = "COVID Ontario", units: str = "weeks",
                 periods: int = 1, previous: list or None = None, prediction: int or None = None):
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
    return f"{s} {articles}"


def parse_dates(file: str) -> list:
    """
    Parse all dates from a file.
    :param file: The file to parse dates from.
    :return: The parsed dates.
    """
    # Ensure the file exists.
    path = os.path.join("Data", "Dates", file)
    if not os.path.exists(path):
        return []
    # Read the file.
    with open(path, "r") as file:
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


if __name__ == '__main__':
    #weeks = parse_dates("COVID Ontario.txt")
    #search_news(keywords="COVID-19", location="Ontario, Canada", trusted=["CDC", "Canada.ca", "Statistique Canada", "AFP Factcheck"], max_results=1, end_date=(2022, 11, 30), delay=0, summarize=True)
    print(build_prompt(keywords="COVID-19", location="Ontario, Canada", trusted=["CDC", "Canada.ca", "Statistique Canada", "AFP Factcheck"], max_results=1, end_date=(2022, 11, 30), delay=0, summarize=True, previous=[5, 6, 12, 20], prediction=25))
