import os
import shutil

from project1.constants import FolderPaths


_emotion_data_folder_path = "./emotion_data"
_class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _does_data_exist(paths: FolderPaths) -> bool:
    for class_name in _class_names:
        class_dir = f"{paths.training_folder_path}/{class_name}"
        if not os.path.exists(class_dir):
            return False
        if len(os.listdir(class_dir)) == 0:
            return False

    # for class_name in class_names:
    #     class_dir = f"{paths.test_folder_path}/{class_name}"
    #     if not os.path.exists(class_dir):
    #         return False
    #     if len(os.listdir(class_dir)) == 0:
    #         return False
    return True


def _delete_training_contents(paths: FolderPaths):
    for class_name in _class_names:
        class_dir = f"{paths.training_folder_path}/{class_name}"
        dir_contents = os.listdir(class_dir)
        for filename in dir_contents:
            file_path = os.path.join(class_dir, filename)
            os.remove(file_path)


def _copy_to_training(paths: FolderPaths):
    # def merge_directories(src_dir1, src_dir2, dest_dir):
    src_dir1 = f"{_emotion_data_folder_path}/train"
    src_dir2 = f"{_emotion_data_folder_path}/validation"
    dest_dir = f"{paths.training_folder_path}"
    # Make sure the destination directory exists; if not, create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy contents from the first directory to the destination
    for item in os.listdir(src_dir1):
        s = os.path.join(src_dir1, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            # If it's a directory, recursively copy it
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            # If it's a file, copy it
            shutil.copy2(s, d)

    # Copy contents from the second directory to the destination
    for item in os.listdir(src_dir2):
        s = os.path.join(src_dir2, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            # If it's a directory, recursively copy it
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            # If it's a file, copy it
            shutil.copy2(s, d)


def _setup_folders(paths: FolderPaths):
    for class_name in _class_names:
        class_folder = f"{paths.training_folder_path}/{class_name}"
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        class_folder = f"{paths.test_folder_path}/{class_name}"
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)


def generate_data_from_zip(paths: FolderPaths):
    if _does_data_exist(paths):
        return
    _setup_folders(paths)
    _delete_training_contents(paths)
    _copy_to_training(paths)
