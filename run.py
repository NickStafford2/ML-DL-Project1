# import test from src
from project1 import unzip, split_data

if __name__ == "__main__":
    file_list = unzip.create_file_list()
    datasets = split_data.create_datasets(file_list)
