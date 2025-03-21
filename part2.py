from project1 import setup_part2_data, cnn, constants

if __name__ == "__main__":
    model_name = "part2"
    paths = constants.get_data_folder(model_name)
    file_list = setup_part2_data.generate_data_from_zip(paths)
    cnn.run(
        model_name,
        training_folder_path=paths.training_folder_path,
        image_size=(224, 224),
        num_classes=3,
    )
