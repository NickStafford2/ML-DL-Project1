# ML-DL-Project1

## Machine Learning Deep Learning Project 1. Image Classification

## Steps to Test both Models

- The zip file accompanying this readme has several trained models already, along with hyperparameter tuning values cached. Additionally, the entire dataset processed is there. It is a big zip file, but I reccomend using that for performance.

1. In the root directory do `poetry install`.
2.

- for full training part 1 with hyperparmeters: `poetry run python part1.py`
- for full training part 1 with hyperparmeters: `poetry run python part1.py`
- for full training part 2 with hyperparmeters: `poetry run python part1.py`
- for full training part 2 with hyperparmeters: `poetry run python part1.py`
- for full training part 2 with hyperparmeters: `poetry run python part2.py`
- for full training part 2 with cached hyperparmeters and model: `poetry run python part2.py --use_cache --use_hp_cache`

3. Clone the repository on your machine.
4. Add the best_model_part1.keras or best_model_part2.keras file to the root directory. This is the trained model that would be predicting your images.
5. Add the test image file you want to run to the root directory of the repo in 'workspaces/ML-DL-Project1'
6. run the file test_part1.py or test_part2.py in project1 directory with `python test_part1.py` or `python test_part2.py `.
7. The program will prompt you to enter the name of the image file you uploaded with its extension.
8. The output will be the class that the model predicts your image to be.
