import random
from dataclasses import dataclass
import numpy as np
import os
import math
from .constants import final_extraction_location, class_names, temp_extraction_location, base_zip_folder_location


@dataclass
class Datasets: 
    training: list[tuple[str, str]]
    validation: list[tuple[str, str]]
    test: list[tuple[str, str]]

    def __init__(self, files: list[str], training_count: int, validation_count: int, test_count: int):
        random.shuffle(files)
        self.class1 = files[:training_count]
        self.class2 = files[:training_count]


def run(training_percent: float, validation_percent: float, test_percent: float) -> dict[str, set[str]]:
    # training = ClassData(set([]), set([]), set([]))
    # valication = ClassData(set([]), set([]), set([]))
    # test = ClassData(set([]), set([]), set([]))

    if training_percent > 1 or validation_percent > 1 or test_percent > 1:
        raise Exception("invalid partitions of data. Must be < 1")
    for class_name in class_names:
        files = os.listdir(f"{final_extraction_location}/{class_name}")
        training_count = math.floor(len(files) * training_percent)
        validation_count = math.floor(len(files) * validation_percent)
        test_count = len(files) - training_count - validation_count


        set(random.sample(files, training_count))
