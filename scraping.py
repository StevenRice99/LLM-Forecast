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
                delay: float = 0, summarize: bool = True) -> None:
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
    :return: Nothing.
    """
    # Configure the time period if one should be used.
    if end_date is None or days < 0:
        end_date = None
        start_date = None
    else:
        if isinstance(end_date, tuple):
            end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])
        start_date = end_date - datetime.timedelta(days=days)
    # Configure the news search.
    google_news = GNews(language=language, country=country, start_date=start_date, end_date=end_date,
                        max_results=max_results, exclude_websites=exclude_websites)
    # Trim the location if it has been set.
    if isinstance(location, str):
        location = location.strip()
        while location.__contains__("  "):
            location = location.replace("  ", " ")
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
        # Append the location if there is one so set it.
        if isinstance(location, str):
            location = f" {location}"
        else:
            location = ""
        # Store all results.
        results = []
        # Search for every keyword.
        for keyword in keywords:
            keyword_results = google_news.get_news(f"{keyword}{location}")
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
    # Format initial output.
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
    s = f"Here {'is' if single else 'are'} {len(results)} new{'i' if single else 's'} articles{days}{words}:"
    # Add every result.
    for result in formatted:
        s += f"\n\nTitle: {result['Title']}"
        s += f"\nPublisher: {result['Publisher']}"
        s += f"\nTrusted: {result['Trusted']}"
        if result["Summary"] is not None:
            s += f"\nSummary: {result['Summary']}"
    # Simply output to the console for now.
    print(s)


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
    search_news(keywords="COVID-19", location="Ontario, Canada",
                trusted=["CDC", "Canada.ca", "Statistique Canada", "AFP Factcheck"], max_results=100,
                end_date=(2022, 11, 30), delay=0, summarize=True)
