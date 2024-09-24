# A Novel Hybridized Forecasting Technique Utilizing ARIMA and Large Language Models

## Usage

1. This project requires [Ollama](https://ollama.com "Ollama"), so ensure it is installed and running.
2. For web scraping, you need to have [Mozilla Firefox](https://www.mozilla.org/en-CA/firefox "Mozilla Firefox") installed.
3. Install requirements by running ``pip install -r requirements.txt`` in the directory of this project. It is recommended you do this in a [virtual environment](https://docs.python.org/3/tutorial/venv.html "Python Virtual Environments and Packages").
4. To replicate our tests, first delete the ``Data``, ``Results``, and ``Responses`` folders.
5. Run ``main.py`` to perform the experiments. Sometimes, during the web scraping [Selenium](https://www.selenium.dev "Selenium") using [Mozilla Firefox](https://www.mozilla.org/en-CA/firefox "Mozilla Firefox") may hang, and not move onto the next news article. In this case, simply restart the script, and it will continue from where it left off.
   1. ``-f`` or ``--forecast`` - Number of weeks to forecast. Defaults to twelve.
   2. ``-w`` or ``--width`` - The width of the figures. Defaults to five.
   3. ``-l`` or ``--height`` - The height of the figures. Defaults to four.
   4. ``-d`` or ``--decimals`` - The number of decimal spaces. Defaults to two.
6. Under the ``Data`` folder, you will see the summaries of the news articles produced by the large language model. Under the ``Results`` folder, you will see the results charts and plots.
   1. ``Actual.csv`` - The actual COVID-19 hospitalizations to occur over the next given weeks from a given week.
   2. ``Baseline.csv`` and ``Full Model.csv`` - The baseline model and full model predictions of how many COVID-19 hospitalizations to occur over the next given weeks from a given week.
   3. ``Difference Baseline.csv`` and ``Difference Full Model.csv`` - The difference between the actual results and each of the model results.
   4. ``Success Rate.csv`` - The success rate of each model, where success was determined if a forecast met or exceeded the actual amounts of hospitalizations to occur over a period.
   5. ``Average Difference.csv`` - The average difference each model had from the actual amounts of hospitalizations to occur over a given period.
   6. ``Total Failures.csv`` - The total failures which occurred for each model over a forecasting period.
   7. ``Total Excess.csv`` - The total excess which occurred for each model over a forecasting period.

## Data

The data used for this experiment is taken from [Covid Timeline Canada](https://github.com/ccodwg/CovidTimelineCanada/blob/main/data/pt/hosp_admissions_pt.csv "Covid Timeline Canada GitHub").