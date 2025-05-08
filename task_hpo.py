from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, DiscreteParameterRange
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name='AI_Studio_Demo',
    task_name='HPO: Train Model',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Get the actual training model task
try:
    BASE_TRAIN_TASK_ID = Task.get_task(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model").id
    logger.info(f"Found base training task with ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task: {e}")
    raise

# Connect parameters
args = {
    'base_train_task_id': BASE_TRAIN_TASK_ID,
    'num_trials': 10,
    'time_limit_minutes': 60,
    'run_as_service': False,
    'test_queue': 'default',  # Queue for test tasks
    'dataset_task_id': ''  # Will be set from pipeline
}
args = task.connect(args)

# Execute the task remotely
task.execute_remotely()

# Get the dataset task ID from pipeline parameters
dataset_task_id = task.get_parameter('General/dataset_task_id')
logger.info(f"Using dataset task ID: {dataset_task_id}")

# Define hyperparameters to optimize
hyper_parameters = [
    UniformIntegerParameterRange(
        name="General/batch_size",
        min_value=16,
        max_value=64,
        step_size=16
    ),
    DiscreteParameterRange(
        name="General/learning_rate",
        values=[0.0001, 0.0003, 0.001, 0.003, 0.01]
    ),
    DiscreteParameterRange(
        name="General/weight_decay",
        values=[1e-6, 1e-5, 1e-4, 1e-3]
    )
]

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    logger.info(f'Job completed! ID: {job_id}, Value: {objective_value}, Iteration: {objective_iteration}')
    logger.info(f'Parameters: {job_parameters}')
    if job_id == top_performance_job_id:
        logger.info(f'New best performance! Objective reached {objective_value}')

# Create the optimizer
try:
    optimizer = HyperParameterOptimizer(
        base_task_id=BASE_TRAIN_TASK_ID,
        execution_queue="pipeline",
        hyper_parameters=hyper_parameters,
        objective_metric_title="validation",
        objective_metric_series="accuracy",
        objective_metric_sign="max",
        max_number_of_concurrent_tasks=2,
        max_iteration_per_job=1,
        total_max_jobs=args['num_trials'],
        project_name="AI_Studio_Demo",
        task_name="HPO: Train Model",
        time_limit_per_job=10,
        pool_period_min=0.1,
        execution_queue_override="default",
        base_task_parameters={
            'General/dataset_task_id': dataset_task_id  # Use the dataset task ID from pipeline
        }
    )
    logger.info("Successfully created optimizer")
except Exception as e:
    logger.error(f"Failed to create optimizer: {e}")
    raise

# Set report period
optimizer.set_report_period(0.2)  # Report every 12 seconds

# Start the HPO process
try:
    logger.info("Starting HPO process...")
    optimizer.start(job_complete_callback=job_complete_callback)
    logger.info("HPO process started successfully")
except Exception as e:
    logger.error(f"Failed to start HPO process: {e}")
    raise

# Set time limit for the entire HPO process
optimizer.set_time_limit(in_minutes=args['time_limit_minutes'])

# Wait until process is done
try:
    logger.info("Waiting for HPO process to complete...")
    optimizer.wait()
    logger.info("HPO process completed")
except Exception as e:
    logger.error(f"Error while waiting for HPO process: {e}")
    raise

# Get the top performing experiments
try:
    top_exp = optimizer.get_top_experiments(top_k=3)
    logger.info(f"Top experiments: {[t.id for t in top_exp]}")
except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    raise

# Make sure background optimization stopped
optimizer.stop()
logger.info("Optimizer stopped")

print('We are done, good bye')