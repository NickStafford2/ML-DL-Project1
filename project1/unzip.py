import os
import glob
import skimage
import shutil
from skimage import io
import random
from zipfile import ZipFile

base_zip_folder_location = './MLDL_Data_Face'
temp_extraction_location = './temp'
final_extraction_location = "./data"

class_names = ['Class 1', 'Class 2', 'Class 3']

def delete_data():
    for class_name in class_names:
        class_dir = f"{final_extraction_location}/{class_name}"
        dir_contents = os.listdir(class_dir)
        for filename in dir_contents:
            file_path = os.path.join(class_dir, filename)  
            # print(file_path)
            os.remove(file_path)  

def delete_temp():
    for class_name in class_names:
        dir_contents = os.listdir(f"{temp_extraction_location}/{class_name}")
        for folder in dir_contents:
            path = f"{temp_extraction_location}/{class_name}/{folder}"
            print(path)
            shutil.rmtree(path)

def unzip():
    for class_name in class_names:
        dir_contents = os.listdir(f"{base_zip_folder_location}/{class_name}")
        for zippedFile in dir_contents:
            if zippedFile.__contains__('.zip'):
                zippedFileNameNoExtension = zippedFile[:2]
                # print(zippedFileNameNoExtension)
                with ZipFile(f"{base_zip_folder_location}/{class_name}/{zippedFile}", 'r') as z:
                    z.extractall(f"{temp_extraction_location}/{class_name}/{zippedFileNameNoExtension}")
                # raise Exception("intentional error")

def copy_to_temp():
    # empty_dirs = []
    for class_name in class_names:
        short_class_name = class_name.replace(" ", "")
        dir_contents = os.listdir(f"{temp_extraction_location}/{class_name}")
        for folder in dir_contents:
            folder_path = f"{temp_extraction_location}/{class_name}/{folder}"
            folder_contents = os.listdir(folder_path)
            for filename in folder_contents:
                old_path= os.path.join(folder_path, filename)
                new_path= os.path.join(f"{final_extraction_location}/{class_name}", f"{short_class_name}_{folder}_{filename}")
                os.rename(old_path, new_path)
                # os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
                # print(f"{old_path} -> {new_path}")
        #     if len(folder_contents) == 0:
        #         empty_dirs.append(f"{class_name}/{folder}")
        #
        # print(empty_dirs)

def run():
    delete_temp()
    delete_data()
    unzip()
    copy_to_temp()
    
