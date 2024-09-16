from main import auto
from evaluate import evaluate

from configuration import start, buffer, capacity, chart_x, chart_y

if __name__ == '__main__':
    # The parameters to try.
    memory = [5, 10, 15]
    top = [1, 3, 5, 10]
    powers = [1, 2, 3, 4, 5]
    # The various lead times to forecast for.
    for lead in [1, 2, 3, 4]:
        auto("Data/COVID Ontario.csv", start, memory, lead, lead - 1, buffer, capacity, powers, top,
             output=f"INITIAL {lead}")
        evaluate(f"Results/INITIAL {lead}", ["Available", "Needed", "Arrived"], True, chart_x, chart_y, False)
