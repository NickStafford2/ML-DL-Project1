import argparse
import os
import shutil
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CNN model with options.")

    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=False,
        help="Set to if you want to disable caching (default is False).",
    )
    return parser.parse_args()


def move_random_files(src_dir, dest_dir, percentage: int = 10):
    # Ensure the destination directory exists, create if not
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # Calculate how many files to move (10% of the total files)
    num_files_to_move = int(len(files) * (percentage / 100))

    # Randomly select files to move
    files_to_move = random.sample(files, num_files_to_move)

    # Move the selected files
    for file in files_to_move:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
        shutil.move(src_file, dest_file)

    print(f"Moved {num_files_to_move} files from {src_dir} to {dest_dir}")
