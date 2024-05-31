import mlflow
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import yaml

def load_yaml_as_dict(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# This automatically reads in the configuration
@hydra.main(config_path='configs', config_name='config', version_base='1.1')
def go(config: DictConfig):

    # Print the current working directory and original working directory
    print("Current working directory:", os.getcwd())  
    print("Original working directory:", get_original_cwd())  

    # Use the original working directory to form the correct path
    config_path = os.path.join(get_original_cwd(), 'configs', 'config.yaml')
    config = load_yaml_as_dict(config_path)
    print(OmegaConf.to_yaml(config))  # Debugging line to print the configuration

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        ## YOUR CODE HERE: call the preprocess step
        _ =  mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters = {
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Data with preprocessing applied"
            },
        )

    if "check_data" in steps_to_execute:

        ## YOUR CODE HERE: call the check_data step
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            },
        )

    if "segregate" in steps_to_execute:

        ## YOUR CODE HERE: call the segregate step
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact":"preprocessed_data.csv:latest",
                "artifact_root":"data",
                "artifact_type":"segregated_data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            },
        )

    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        ## YOUR CODE HERE: call the random_forest step
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config['random_forest_pipeline']["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

        ## YOUR CODE HERE: call the evaluate step
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest"
            },
        )


if __name__ == "__main__":
    go()
