from prefect import serve

from pipeline_prefect import (
    all_flow, code_flow, evaluate_flow, install_flow, train_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all", cron="0 2 * * *", tags=["full-pipeline", "mlops"]
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["mlops"]),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", tags=["mlops"]),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["mlops"]),
    )
