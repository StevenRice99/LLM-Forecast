# Pandemic Surge Forecasting utilizing Multiple Predictions for Improved Accuracy

Testing methods to forecast orders for handling supply chain logistics focusing on COVID-19.

## Code

- ``main.py`` contains the core forecasting logic. The ``predict`` method is the forecasting model itself, the ``test`` method performs the testing for the forecasting model, and the ``auto`` method is a helper method to test a lot of model configurations.
- ``evaluate.py`` contains the method for generating tables and plots.
- ``covid.py`` contains a method for downloading and parsing COVID-19 data for Ontario, Canada.

## Usage

1. Install requirements by running ``pip install -r requirements.txt`` in the directory of this project.
2. Before your first time running the remaining code for tests, run ``covid.py`` to download and convert Ontario COVID-19 data into a format for this repository.
3. Running ``demo.py`` provides an easy way to run a single model and generate a chart and plot. After running it, the ``Results`` directory will contain a folder for the model which inside its sub folders will contain a plot of the run, along with a CSV file (which only has one entry, being this model if it is your first run) in the root of the ``Results`` folder with the results of the run.
4. To duplicate our tests, run ``first.py`` or ``second.py`` respectfully. After either run, move the results into a subfolder of ``Results`` being either ``FIRST`` or ``SECOND`` respectfully. Then, run ``evaluate.py`` to generate the charts and plots for these tests.