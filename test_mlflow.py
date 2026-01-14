import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("claim_denial_test")

with mlflow.start_run(run_name="sanity_run"):
    mlflow.log_metric("accuracy", 0.88)

print("MLflow run logged")