import argparse
from project1 import setup_part1_data, cnn, constants, utils


if __name__ == "__main__":
    args = utils.parse_args()
    model_name = "part1"
    paths = constants.get_data_folder(model_name)
    file_list = setup_part1_data.generate_data_from_zip(paths)
    cnn.run(
        model_name,
        input_channels=1,
        training_folder_path=paths.training_folder_path,
        image_size=(48, 48),
        num_classes=7,
        use_cache=args.use_cache,
    )
