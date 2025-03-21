import os
import shutil
import math
from zipfile import ZipFile

from project1 import utils
from project1.constants import FolderPaths


_zip_folder_path = "./MLDL_Data_Face"
_class_names = ["Class 1", "Class 2", "Class 3"]
_temp_folder_path = "./temp"


def _does_data_exist(paths: FolderPaths) -> bool:
    for class_name in _class_names:
        class_dir = f"{paths.training_folder_path}/{class_name}"
        if not os.path.exists(class_dir):
            return False
        if len(os.listdir(class_dir)) == 0:
            return False

    for class_name in _class_names:
        class_dir = f"{paths.test_folder_path}/{class_name}"
        if not os.path.exists(class_dir):
            return False
        if len(os.listdir(class_dir)) == 0:
            return False
    return True


def _setup_folders(paths: FolderPaths):
    _delete_temp()
    if not os.path.exists(_temp_folder_path):
        os.makedirs(_temp_folder_path)

    for class_name in _class_names:
        if not os.path.exists(f"{_temp_folder_path}/{class_name}"):
            os.makedirs(f"{_temp_folder_path}/{class_name}")

        class_folder = f"{paths.training_folder_path}/{class_name}"
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        class_folder = f"{paths.test_folder_path}/{class_name}"
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)


def _delete_training_contents(paths: FolderPaths):
    for class_name in _class_names:
        class_dir = f"{paths.training_folder_path}/{class_name}"
        dir_contents = os.listdir(class_dir)
        for filename in dir_contents:
            file_path = os.path.join(class_dir, filename)
            os.remove(file_path)


def _delete_temp():
    if os.path.exists(_temp_folder_path):
        shutil.rmtree(_temp_folder_path)


def _unzip_to_temp():
    for class_name in _class_names:
        dir_contents = os.listdir(f"{_zip_folder_path}/{class_name}")
        for zippedFile in dir_contents:
            if zippedFile.__contains__(".zip"):
                zippedFileNameNoExtension = zippedFile[:2]
                with ZipFile(f"{_zip_folder_path}/{class_name}/{zippedFile}", "r") as z:
                    z.extractall(
                        f"{_temp_folder_path}/{class_name}/{zippedFileNameNoExtension}"
                    )


def _copy_to_training(paths: FolderPaths):
    for class_name in _class_names:
        short_class_name = class_name.replace(" ", "")
        dir_contents = os.listdir(f"{_temp_folder_path}/{class_name}")
        for folder in dir_contents:
            folder_short = folder.replace(".", "").zfill(3)
            folder_path = f"{_temp_folder_path}/{class_name}/{folder}"
            folder_contents = os.listdir(folder_path)
            for filename in folder_contents:
                file = filename.split(".")
                file[0] = file[0].zfill(4)
                filename_short = ".".join(file)

                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(
                    f"{paths.training_folder_path}/{class_name}",
                    f"{short_class_name}_{folder_short}_{filename_short}",
                )
                os.rename(old_path, new_path)


def _create_test_data(paths: FolderPaths):
    for class_name in _class_names:
        src_dir = f"{paths.training_folder_path}/{class_name}"
        dest_dir = f"{paths.test_folder_path}/{class_name}"
        utils.move_random_files(src_dir, dest_dir)


def generate_data_from_zip(paths: FolderPaths):
    if _does_data_exist(paths):
        return
    _setup_folders(paths)
    _delete_training_contents(paths)
    _unzip_to_temp()
    _copy_to_training(paths)
    _delete_temp()
    _create_test_data(paths)
