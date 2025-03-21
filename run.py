from project1 import unzip, cnn

if __name__ == "__main__":
    file_list = unzip.generate_data_from_zip()
    cnn.run()
