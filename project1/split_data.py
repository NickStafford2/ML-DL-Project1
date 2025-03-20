import random
from dataclasses import dataclass, field
import numpy as np
import os
import math
from .constants import data_folder_path, class_names, temp_folder_path, zip_folder_path


@dataclass
class Datasets:
    training: list[tuple[np.ndarray, str]] = field(default_factory=list)
    validation: list[tuple[np.ndarray, str]] = field(default_factory=list)
    test: list[tuple[np.ndarray, str]] = field(default_factory=list)


def create_datasets(file_paths: list[tuple[np.ndarray, str]]) -> Datasets:
    (training_count, validation_count, test_count) = _get_set_sizes(
        len(file_paths), 0.8, 0.1, 0.1
    )
    random.shuffle(file_paths)
    training = file_paths[:training_count]
    validation = file_paths[training_count : training_count + validation_count]
    test = file_paths[training_count + validation_count :]

    assert (
        len(training) == training_count
    ), f"Expected {training_count}, but got {len(training)}"
    assert (
        len(validation) == validation_count
    ), f"Expected {validation_count}, but got {len(validation)}"
    assert len(test) == test_count, f"Expected {test_count}, but got {len(test)}"

    print(
        f"created dataset with \n  {len(training)} training \n  {len(validation)} validation \n  {len(test)} test"
    )
    datasets = Datasets(training, validation, test)
    return datasets


def print_datasets(datasets: Datasets, limit: int = 10):
    for d in datasets.training[:limit]:
        print(f"{d[0]} - {d[1]}")
    for d in datasets.validation[:limit]:
        print(f"{d[0]} - {d[1]}")
    for d in datasets.test[:limit]:
        print(f"{d[0]} - {d[1]}")


def _get_set_sizes(
    total: int, training_percent: float, validation_percent: float, test_percent: float
) -> tuple[int, int, int]:
    training_count = -1
    validation_count = -1
    test_count = -1

    if training_percent > 1 or validation_percent > 1 or test_percent > 1:
        raise Exception("invalid partitions of data. Must be < 1")
    for class_name in class_names:
        training_count = math.floor(total * training_percent)
        validation_count = math.floor(total * validation_percent)
        test_count = total - training_count - validation_count

    return (training_count, validation_count, test_count)
    # set(random.sample(files, training_count))
