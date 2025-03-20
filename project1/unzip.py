import os
import glob
import skimage
import shutil
from skimage import io
import random
from zipfile import ZipFile
from .constants import (
    data_folder_path,
    class_names,
    temp_folder_path,
    zip_folder_path,
)


def _does_data_exist() -> bool:
    for class_name in class_names:
        class_dir = f"{data_folder_path}/{class_name}"
        if not os.path.exists(class_dir):
            return False
        if len(os.listdir(class_dir)) == 0:
            return False
    return True


def _setup_folders():
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    for class_name in class_names:
        if not os.path.exists(f"{temp_folder_path}/{class_name}"):
            os.makedirs(f"{temp_folder_path}/{class_name}")

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    for class_name in class_names:
        if not os.path.exists(f"{data_folder_path}/{class_name}"):
            os.makedirs(f"{data_folder_path}/{class_name}")


def _delete_data_contents():
    for class_name in class_names:
        class_dir = f"{data_folder_path}/{class_name}"
        dir_contents = os.listdir(class_dir)
        for filename in dir_contents:
            file_path = os.path.join(class_dir, filename)
            # print(file_path)
            os.remove(file_path)


def _delete_temp():
    for class_name in class_names:
        dir_contents = os.listdir(f"{temp_folder_path}/{class_name}")
        for folder in dir_contents:
            path = f"{temp_folder_path}/{class_name}/{folder}"
            # print(path)
            shutil.rmtree(path)


def _unzip_to_temp():
    for class_name in class_names:
        dir_contents = os.listdir(f"{zip_folder_path}/{class_name}")
        for zippedFile in dir_contents:
            if zippedFile.__contains__(".zip"):
                zippedFileNameNoExtension = zippedFile[:2]
                # print(zippedFileNameNoExtension)
                with ZipFile(f"{zip_folder_path}/{class_name}/{zippedFile}", "r") as z:
                    z.extractall(
                        f"{temp_folder_path}/{class_name}/{zippedFileNameNoExtension}"
                    )


def _copy_to_data():
    for class_name in class_names:
        short_class_name = class_name.replace(" ", "")
        dir_contents = os.listdir(f"{temp_folder_path}/{class_name}")
        for folder in dir_contents:
            folder_short = folder.replace(".", "").zfill(3)
            folder_path = f"{temp_folder_path}/{class_name}/{folder}"
            folder_contents = os.listdir(folder_path)
            for filename in folder_contents:
                file = filename.split(".")
                file[0] = file[0].zfill(4)
                filename_short = ".".join(file)

                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(
                    f"{data_folder_path}/{class_name}",
                    f"{short_class_name}_{folder_short}_{filename_short}",
                )
                os.rename(old_path, new_path)


def _format_data():
    _setup_folders()
    _delete_temp()
    _delete_data_contents()
    _unzip_to_temp()
    _copy_to_data()
    _delete_temp()


def create_file_list() -> list[tuple[str, str]]:
    if not _does_data_exist():
        _format_data()

    output = []
    for class_name in class_names:
        class_dir = f"{data_folder_path}/{class_name}"
        dir_contents = os.listdir(class_dir)
        for filename in dir_contents:
            file_path = os.path.join(class_dir, filename)
            output.append((file_path, class_name))
    return output
