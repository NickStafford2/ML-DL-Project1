from project1 import unzip, cnn

if __name__ == "__main__":
    file_list = unzip.create_all_data(100)
    cnn.run()
