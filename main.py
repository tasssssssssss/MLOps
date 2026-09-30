"""Entry point: run pipeline steps from the command line."""

import argparse

import model_pipeline as mp


def step_prepare():
    return mp.prepare_data()


def step_train(x_train, y_train):
    return mp.train_model(x_train, y_train)


def step_save(model):
    mp.save_model(model)


def step_load():
    return mp.load_model()


def step_evaluate(model, x_test, y_test):
    return mp.evaluate_model(model, x_test, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step", choices=["train", "evaluate", "predict", "all"], default="all"
    )
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = step_prepare()

    if args.step in ("train", "all"):
        model = step_train(x_train, y_train)
        step_save(model)
    if args.step in ("evaluate", "all"):
        step_evaluate(step_load(), x_test, y_test)
    if args.step == "predict":
        sample = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]
        print("Prediction:", mp.predict(step_load(), sample))


if __name__ == "__main__":
    main()
