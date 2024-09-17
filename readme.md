# Hybridized Pandemic Forecasting Utilizing Large Language Models and Analytical Methods

## Code

- ``main.py`` contains the core forecasting logic. The ``predict`` method is the forecasting model itself, the ``test`` method performs the testing for the forecasting model, and the ``auto`` method is a helper method to test a lot of model configurations.
- ``evaluate.py`` contains the method for generating tables and plots.
- ``covid.py`` contains a method for downloading and parsing COVID-19 data for Ontario, Canada. The data is from [Covid Timeline Canada](https://github.com/ccodwg/CovidTimelineCanada/blob/main/data/pt/hosp_admissions_pt.csv "Covid Timeline Canada GitHub").
- ``scraping.py`` includes methods for web scraping.
- ``llm.py`` contains methods for messaging the [Ollama](https://ollama.com "Ollama") large language model.
- ``initial_tests.py`` contains the initial hyperparameter searching for the baseline model.
- ``llm_tests.py`` contains the tests for trying the best hyperparameters with the full model utilizing large language models.

## Usage

1. This project requires [Ollama](https://ollama.com "Ollama"), so ensure it is installed and running.
2. Install requirements by running ``pip install -r requirements.txt`` in the directory of this project.
3. For web scraping, you need to have [Mozilla Firefox](https://www.mozilla.org/en-CA/firefox "Mozilla Firefox") installed.
4. Run ``covid.py`` to download and convert Ontario COVID-19 data into a format for this repository.
5. To replicate our tests, first delete the ``DATA`` and ``RESULTS`` folders.
6. Web scraping is preferably done ahead of time. Note that this process takes a long time due to needing to use [Selenium](https://www.selenium.dev "") if Google starts to mask URLs. To do this, run ``scraping.py``.
7. Running ``initial_tests.py`` will perform the initial hyperparameter searching for the baseline model, with the results being under the ``RESULTS`` folder in the ``INITIAL`` subfolders.
8. Running ``llm_tests.py`` will perform the tests for trying the best hyperparameters with the full model. The results can be found under the ``RESULTS`` folder in the ``FINAL`` subfolders.