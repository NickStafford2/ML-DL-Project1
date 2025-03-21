# import keras_tuner
#
#
# def create():
#     hp = keras_tuner.HyperParameters()
#     tuner = keras_tuner.RandomSearch(
#         hypermodel=build_model,
#         objective="val_accuracy",
#         max_trials=3,
#         executions_per_trial=2,
#         overwrite=True,
#         directory="my_dir",
#         project_name="helloworld",
#     )
#     tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
