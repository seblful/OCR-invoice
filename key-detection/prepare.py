from preparers import DatasetCreator

import os


# Setup hyperparameters
TRAIN_SPLIT = 0.8

HOME = os.getcwd()
RAW_DATA_PATH = os.path.join(HOME, 'raw-data')  # data from label studio
DATASET_PATH = os.path.join(HOME, 'dataset')  # dataset for training


def main():
    # Initializing dataset creator and process data (create dataset)
    dataset_creator = DatasetCreator(raw_data_path=RAW_DATA_PATH,
                                     dataset_path=DATASET_PATH,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.process()


if __name__ == '__main__':
    main()
