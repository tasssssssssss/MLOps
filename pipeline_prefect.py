"""Prefect pipeline for the churn project."""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task

import main as steps

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd):
    subprocess.run(cmd, check=True)  # nosec B603


# ---------- Tasks ----------
@task
def install_task():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task
def format_task():
    run([sys.executable, "-m", "black", *FILES])


@task
def quality_task():
    run([sys.executable, "-m", "flake8", "--max-line-length", "88", *FILES])


@task
def security_task():
    run([sys.executable, "-m", "bandit", "-q", *FILES])


@task
def unit_test_task():
    run([sys.executable, "-m", "pytest", "tests", "-q"])


@task
def prepare_task():
    return steps.step_prepare()


@task
def train_task(x_train, y_train):
    return steps.step_train(x_train, y_train)


@task
def save_task(model):
    steps.step_save(model)


@task
def load_task():
    return steps.step_load()


@task
def evaluate_task(model, x_test, y_test):
    return steps.step_evaluate(model, x_test, y_test)


# ---------- Flows ----------
@flow(name="install")
def install_flow():
    install_task()


@flow(name="code")
def code_flow():
    format_task()
    quality_task()
    security_task()
    unit_test_task()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, x_test, y_test)


@flow(name="all")
def all_flow():
    code_flow()
    x_train, x_test, y_train, y_test = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)
    evaluate_task(model, x_test, y_test)


FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "all": all_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    FLOWS[parser.parse_args().flow]()
