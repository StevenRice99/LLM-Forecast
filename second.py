from main import auto

if __name__ == '__main__':
    # Second trial to try and improve the best results from the first trial.
    auto("Data/COVID Ontario.csv", 1, 8, 2, [3, 4], 500, 0, [2, 3], [1, 5, 10], None, None, False)
