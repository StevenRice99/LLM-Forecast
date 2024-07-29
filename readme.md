# Hybridized Pandemic Forecasting Utilizing Large Language Models and Analytical Methods

## Code

- ``main.py`` contains the core forecasting logic. The ``predict`` method is the forecasting model itself, the ``test`` method performs the testing for the forecasting model, and the ``auto`` method is a helper method to test a lot of model configurations.
- ``evaluate.py`` contains the method for generating tables and plots.
- ``covid.py`` contains a method for downloading and parsing COVID-19 data for Ontario, Canada. The data is from [Covid Timeline Canada](https://github.com/ccodwg/CovidTimelineCanada/blob/main/data/pt/hosp_admissions_pt.csv "Covid Timeline Canada GitHub").
- ``scraping.py`` includes methods for web scraping and interacting with large language models.
- ``initial_tests.py`` contains the initial hyperparameter searching for polynomial regression, ARIMA, and SVR methods.
- ``llm_tests.py`` contains the tests for trying the best hyperparameters with and without a large language model.

## Usage

1. Install requirements by running ``pip install -r requirements.txt`` in the directory of this project.
2. For web scraping, you need to have [Mozilla Firefox](https://www.mozilla.org/en-CA/firefox "Mozilla Firefox") installed.
3. If you wish to use [HuggingFace](https://huggingface.co "HuggingFace") models, create a file named ``hugging_face.txt`` in the root of this project. On the first line, input your email/username and on the second line input your password. **This will not work if you have two-factor authentication enabled.** If you choose to not do this step, your large language models will be run through [DuckDuckGo AI Chat](https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat "DuckDuckGo AI Chat").
4. Run ``covid.py`` to download and convert Ontario COVID-19 data into a format for this repository.
5. To duplicate our tests, first delete the ``DATA`` and ``RESULTS`` folders.
6. In this implementation, web scraping is preferably done ahead of time. Note that this process takes a long time due to needing to use [Selenium](https://www.selenium.dev "") and wait to query the free large language models to not get rate limited. To do this, run ``scraping.py``.
7. Running ``initial_tests.py`` will perform the initial hyperparameter searching for polynomial regression, ARIMA, and SVR methods. The results for these tests can be found in the ``RESULTS`` folder under the ``ARIMA``, ``POLYNOMIAL``, and ``SVR`` subfolders.
8. Running ``llm_tests.py`` will perform the tests for trying the best hyperparameters with and without a large language model. The results can be found under ``RESULTS/FINAL``.