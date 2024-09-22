# Hybridized Pandemic Forecasting Utilizing Large Language Models and Analytical Methods

## Usage

1. This project requires [Ollama](https://ollama.com "Ollama"), so ensure it is installed and running.
2. For web scraping, you need to have [Mozilla Firefox](https://www.mozilla.org/en-CA/firefox "Mozilla Firefox") installed.
3. Install requirements by running ``pip install -r requirements.txt`` in the directory of this project. It is recommended you do this in a [virtual environment](https://docs.python.org/3/tutorial/venv.html "Python Virtual Environments and Packages").
4. To replicate our tests, first delete the ``Data``, ``Results``, and ``Responses`` folders. The web scraping is the most time-consuming part, so if you wish to keep the existing news summaries and simply replicate our results from that, keep the ``Data`` and only delete the ``Results`` and ``Responses`` folders.
5. Run ``main.py`` to perform the experiments. Sometimes, during the web scraping [Selenium](https://www.selenium.dev "Selenium") using [Mozilla Firefox](https://www.mozilla.org/en-CA/firefox "Mozilla Firefox") may hang, and not move onto the next news article. In this case, simply restart the script, and it will continue from where it left off.
6. Under the ``Data`` folder, you will see the summaries of the news articles produced by the large language model. Under the ``Results`` folder, you will see the results charts and plots.
   1. ``Actual.csv`` - The actual COVID-19 hospitalizations to occur over the next given weeks from a given week.
   2. ``Baseline.csv`` - The baseline ARIMA model predictions of how many COVID-19 hospitalizations to occur over the next given weeks from a given week.
   3. ``LLM.csv`` - The large language model predictions of how many COVID-19 hospitalizations to occur over the next given weeks from a given week.

## Data

The data used for this experiment is taken from [Covid Timeline Canada](https://github.com/ccodwg/CovidTimelineCanada/blob/main/data/pt/hosp_admissions_pt.csv "Covid Timeline Canada GitHub").