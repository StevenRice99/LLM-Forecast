import datetime

from gnews import GNews


def search_news(keywords: str or list or None = None, max_results: int = 10, language: str = "en", country: str = "CA",
                end_date: tuple or None = None, days: int = 7, exclude_websites: list or None = None,
                trusted: list or None = None):
    if end_date is None or days < 0:
        end_date = None
        start_date = None
    else:
        end_date = datetime.datetime.strptime(*end_date)
        start_date = end_date - datetime.timedelta(days=days)
    google_news = GNews(language=language, country=country, start_date=start_date, end_date=end_date,
                        max_results=max_results, exclude_websites=exclude_websites)
    if keywords is None:
        results = google_news.get_top_news()
    else:
        if isinstance(keywords, list):
            keywords = " ".join(keywords)
        results = google_news.get_news(keywords)
    s = f"Here are the top {len(results)} news articles:"
    if trusted is None:
        trusted = []
    for result in results:
        title = result["title"]
        description = result["description"]
        publisher = result["publisher"]["title"]
        title = title.replace(publisher, "").strip().strip("-").strip()
        description = description.replace(publisher, "").strip().strip("-").strip()
        s += f"\n\nTitle: {title}"
        if title != description:
            s += f"\nDescription: {description}"
        s += f"\nPublisher: {publisher}"
        s += f"\nTrusted: {publisher in trusted}"
    print(s)


if __name__ == '__main__':
    search_news(keywords="COVID-19 Ontario Canada Pandemic Infectious Disease Hospitalizations Healthcare",
                trusted=["CDC", "Canada.ca"])
