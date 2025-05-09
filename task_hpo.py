from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
import logging
import time

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

# Connect parameters
args = {
    'base_train_task_id': 'd462790f2efa4daea96ed0b7e1a1003c',  # Will be set from pipeline
    'num_trials': 10,
    'time_limit_minutes': 60,
    'run_as_service': False,
    'test_queue': 'pipeline',  # Queue for test tasks
    'processed_dataset_id': '99e286d358754697a37ad75c279a6f0a',  # Will be set from pipeline
    'num_epochs': 50,  # Maximum number of epochs for HPO trials
    'batch_size': 32,  # Default batch size
    'learning_rate': 1e-3,  # Default learning rate
    'weight_decay': 1e-5  # Default weight decay
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('General/processed_dataset_id')  # Get from General namespace
if not dataset_id:
    # Try getting from args as fallback
    dataset_id = args.get('processed_dataset_id')
    print(f"No dataset_id now get dataset ID from args: {dataset_id}")

if not dataset_id:
    # Use fixed dataset ID as last resort
    dataset_id = "99e286d358754697a37ad75c279a6f0a"
    print(f"Using fixed dataset ID: {dataset_id}")

logger.info(f"Received dataset ID from parameters: {dataset_id}")

if not dataset_id:
    logger.error("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")
    raise ValueError("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")

# Get the actual training model task
try:
    BASE_TRAIN_TASK_ID = Task.get_task(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model").id
    logger.info(f"Found base training task with ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task: {e}")
    raise

# Verify dataset exists
try:
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Successfully verified dataset: {dataset.name}")
except Exception as e:
    logger.error(f"Failed to verify dataset: {e}")
    raise

# Create the HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        UniformIntegerParameterRange('General/num_epochs', min_value=10, max_value=args['num_epochs']),
        UniformIntegerParameterRange('General/batch_size', min_value=8, max_value=args['batch_size']),
        UniformParameterRange('General/learning_rate', min_value=1e-4, max_value=args['learning_rate']),
        UniformParameterRange('General/weight_decay', min_value=1e-6, max_value=args['weight_decay'])
    ],
    objective_metric_title='validation',
    objective_metric_series='accuracy',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    compute_time_limit=None,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=args['num_epochs'],
    pool_period_min=2.0,
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=5,
    parameter_override={
        'General/processed_dataset_id': "99e286d358754697a37ad75c279a6f0a",  # Pass the dataset ID to test tasks
        'General/test_queue': args['test_queue'],  # Pass the test queue
        'General/num_epochs': args['num_epochs'],  # Pass default num_epochs
        'General/batch_size': args['batch_size'],  # Pass default batch_size
        'General/learning_rate': args['learning_rate'],  # Pass default learning_rate
        'General/weight_decay': args['weight_decay']  # Pass default weight_decay
    },
    base_task_name="Pipeline step 3 train model",  # Specify the base task name
    base_task_project="AI_Studio_Demo",  # Specify the base task project
    base_task_type=Task.TaskTypes.training  # Specify the base task type
)

# Start the HPO task
logger.info("Starting HPO task...")
hpo_task.start()

# Wait for some results
logger.info("Waiting for initial results...")
time.sleep(60)  # Wait 60 seconds for initial results

# Get the top performing experiments
try:
    top_exp = hpo_task.get_top_experiments(top_k=3)
    if top_exp:
        logger.info(f"Top experiments: {[t.id for t in top_exp]}")
        for exp in top_exp:
            # Get the last reported value for validation accuracy
            metrics = exp.get_last_scalar_metrics()
            if metrics and 'validation' in metrics and 'accuracy' in metrics['validation']:
                accuracy = metrics['validation']['accuracy']
                logger.info(f"Experiment {exp.id} accuracy: {accuracy}")
            else:
                logger.warning(f"Experiment {exp.id} has no validation accuracy metric")
    else:
        logger.warning("No experiments completed yet. This might be normal if the optimization just started.")
except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    raise

# Make sure background optimization stopped
hpo_task.stop()
logger.info("Optimizer stopped")

print('We are done, good bye')