from project1 import setup_part1_data, cnn, constants

if __name__ == "__main__":
    model_name = "part1"
    paths = constants.get_data_folder(model_name)
    file_list = setup_part1_data.generate_data_from_zip(paths)
    cnn.run(model_name, training_folder_path=paths.training_folder_path, num_classes=3)
