from clearml import Task
from clearml.automation import PipelineController
import logging
# import os
# os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY")
# os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY")
# os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration - using same queue for everything
EXECUTION_QUEUE = "pipeline"

def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    logger.info(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    logger.info("Completed Task id={}".format(a_node.executed))
    # Log the parameters of the completed task
    task = Task.get_task(task_id=a_node.executed)
    logger.info(f"Task parameters: {task.get_parameters()}")
    return


def run_pipeline():
    # Connecting ClearML with the current pipeline
    pipe = PipelineController(
        name="AI_Studio_Pipeline_Demo", 
        project="AI_Studio_Demo", 
        version="0.0.1", 
        add_pipeline_tags=False
    )

    # Set default queue for pipeline control
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")

    # Add dataset creation step
    pipe.add_step(
        name="stage_data",
        base_task_project="AI_Studio_Demo",
        base_task_name="Pipeline step 1 dataset artifact",
        execution_queue=EXECUTION_QUEUE
    )

    # Add dataset processing step
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="AI_Studio_Demo",
        base_task_name="Pipeline step 2 process dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.25,
            "General/random_state": 42
        }
    )

    # Add HPO step
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_process"],
        base_task_project="AI_Studio_Demo",
        base_task_name="HPO: Train Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/test_queue": EXECUTION_QUEUE
        }
    )

    # Add training step
    pipe.add_step(
        name="stage_train",
        parents=["stage_process", "stage_hpo"],
        base_task_project="AI_Studio_Demo",
        base_task_name="Pipeline step 3 train model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/hpo_task_id": "${stage_hpo.id}",
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_epochs": 20,
            "General/batch_size": 16,
            "General/learning_rate": 1e-3,
            "General/weight_decay": 1e-5
        }
    )

    # Set callbacks for better logging
    pipe.set_pre_step_callback(pre_execute_callback_example)
    pipe.set_post_step_callback(post_execute_callback_example)

    # Start the pipeline locally but tasks will run on queue
    logger.info("Starting pipeline locally with tasks on queue: %s", EXECUTION_QUEUE)
    pipe.start_locally()
    logger.info("Pipeline started successfully")


if __name__ == "__main__":
    run_pipeline()
