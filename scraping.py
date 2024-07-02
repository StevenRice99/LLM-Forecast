import datetime
import os

from duckduckgo_search import DDGS
from gnews import GNews
from newspaper import Article


def search_news(keywords: str or list or None = None, max_results: int = 100, language: str = "en", country: str = "CA",
                end_date: tuple or datetime.datetime or None = None, days: int = 7,
                exclude_websites: list or None = None, trusted: list or None = None, model: str or None = None) -> None:
    """
    Search the web for news.
    :param keywords: The keywords to search for.
    :param max_results: The maximum number of results to return.
    :param language: The language to search in.
    :param country: The Country to search in.
    :param end_date: The latest date that news results can be from.
    :param days: How many days prior to the end date to search from.
    :param exclude_websites: Websites to exclude.
    :param trusted: What websites should be labelled as trusted.
    :param model: Which model to use for LLM summaries.
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
    # If no keywords, get the top news.
    if keywords is None:
        results = google_news.get_top_news()
    # Otherwise, search by the keywords.
    else:
        if isinstance(keywords, list):
            keywords = " ".join(keywords)
        keywords = keywords.strip()
        while keywords.__contains__("  "):
            keywords = keywords.replace("  ", " ")
        results = google_news.get_news(keywords)
    # If there are no trusted sites, make an empty list for comparisons.
    if trusted is None:
        trusted = []
    # Ensure the LLM for summarizations is valid.
    if model is None or model not in ["gpt-3.5", "claude-3-haiku", "llama-3-70b", "mixtral-8x7b"]:
        model = "gpt-3.5"
    # Format all results.
    formatted = []
    for result in results:
        publisher = result["publisher"]["title"]
        # Try to get the full article and then summarize it with an LLM.
        # noinspection PyBroadException
        try:
            # Get the full article.
            article = Article(result["url"])
            article.download()
            article.parse()
            title = article.title
            # Clean the description.
            description = article.text.strip().replace("\r", "\n")
            while description.__contains__("\n\n"):
                description = description.replace("\n\n", "\n")
            # Summarize the description with an LLM.
            description = DDGS().chat(f"Summarize this article: {description}", model=model)
        # If the full article cannot be downloaded or the summarization fails, use the initial news info.
        except:
            title = result["title"].replace(publisher, "").strip().strip("-").strip()
            description = result["description"].replace(publisher, "").strip().strip("-").strip()
        # No point in having the description if it is just equal to the title.
        if title == description:
            description = None
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
        formatted.append({"Title": title, "Description": description, "Publisher": publisher,
                          "Year": int(published_date[3]), "Month": published_date[2],"Day": int(published_date[1]),
                          "Hour": int(published_time[0]), "Minute": int(published_time[1]),
                          "Second": int(published_time[2]), "Trusted": publisher in trusted})
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
        keywords = ""
    else:
        split = keywords.split(" ")
        plural = "s" if len(split) > 1 else ""
        keywords = f" containing keyword{plural} {split[0]}"
        for i in range(1, len(split)):
            if i == len(split) - 1:
                keywords += f", or {split[i]}"
            else:
                keywords += f", {split[i]}"
    s = f"Here are {len(results)} news articles{days}{keywords}:"
    # Add every result.
    for result in formatted:
        s += f"\n\nTitle: {result['Title']}"
        s += f"\nPublisher: {result['Publisher']}"
        s += f"\nTrusted: {result['Trusted']}"
        if result["Description"] is not None:
            s += f"\nDescription: {result['Description']}"
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
    search_news(keywords="COVID-19 Ontario Canada Pandemic Infectious Disease Hospitalizations Healthcare",
                trusted=["CDC", "Canada.ca"], max_results=5, end_date=(2024, 7, 1))
