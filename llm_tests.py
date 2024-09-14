from main import auto
from evaluate import evaluate

from configuration import start, buffer, capacity, chart_x, chart_y


def llm_test(lead: int = 0, power: int = 0, top: int = 1, memory: int = 1) -> None:
    """
    Handle testing parameters with and without a large language model.
    :param memory: How much past orders should be given to the forecasting model.
    :param lead: How long is the lead time before orders arrive.
    :param power: The polynomial up which to compute predictions with.
    :param top: The number of top parameters to take into account for averaging the prediction.
    :return: Nothing.
    """
    auto("Data/COVID Ontario.csv", start, memory, lead, lead - 1, buffer, capacity, power, top, model=True, verbose=True, output=f"FINAL")


if __name__ == '__main__':
    # Assign parameters for the various lead times as found in the initial tests.
    llm_test(1, 5, 1, 15)
    llm_test(2, 5, 1, 10)
    llm_test(3, 5, 10, 15)
    llm_test(4, 4, 10, 15)
    evaluate(f"Results/FINAL", ["Available", "Needed"], True, chart_x, chart_y, False)
