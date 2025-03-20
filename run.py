from typing import Any
from project1 import unzip, split_data, cnn, dataset

if __name__ == "__main__":
    file_list = unzip.create_all_data(100)
    # datasets = split_data.create_datasets(file_list)
    # train_ds, val_ds = dataset.create_dataset()
    cnn.run()
