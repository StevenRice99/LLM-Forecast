from main import test

if __name__ == '__main__':
    # Second trial to try and improve the best results from the first trial.
    test("Data/COVID Ontario.csv", 1, 8, 2, 4, 500, 0, 3, 5, None, None, True, model="gpt-3.5", output="THIRD")
