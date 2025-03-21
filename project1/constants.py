from dataclasses import dataclass
import os


@dataclass
class FolderPaths:
    training_folder_path: str
    test_folder_path: str
    problem_folder_path: str
    root_folder_path: str = "./data"

    def __init__(self, problem_name: str) -> None:
        self.problem_folder_path = f"{self.root_folder_path}/{problem_name}"
        self.training_folder_path = f"{self.problem_folder_path}/training"
        self.test_folder_path = f"{self.problem_folder_path}/test"
        self._setup_folders()

    def _setup_folders(self):
        if not os.path.exists(self.root_folder_path):
            os.makedirs(self.root_folder_path)
        if not os.path.exists(self.problem_folder_path):
            os.makedirs(self.problem_folder_path)
        if not os.path.exists(self.test_folder_path):
            os.makedirs(self.test_folder_path)
        if not os.path.exists(self.training_folder_path):
            os.makedirs(self.training_folder_path)


def get_data_folder(problem_name: str) -> FolderPaths:
    return FolderPaths(problem_name)
