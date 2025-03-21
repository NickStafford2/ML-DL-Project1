from project1 import unzip, cnn, constants

if __name__ == "__main__":
    model_name = "part1"
    paths = constants.get_data_folder(model_name)
    cnn.run(model_name, training_folder_path=paths.training_folder_path, num_classes=7)
