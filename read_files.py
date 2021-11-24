import pandas as pd
import glob


def read_files(path):
    all_files = glob.glob(path + "/*.psv")
    training_set = pd.DataFrame()
    for file in all_files:
        patient = pd.read_csv(file, sep='|')
        print(patient['Age'])
        training_set = training_set.append(patient, ignore_index=True)
    return training_set


if __name__ == '__main__':
    path = "//data/test"
    training_set = read_files(path)

    print(training_set)
    training_set.to_csv('training_test', index=False)
