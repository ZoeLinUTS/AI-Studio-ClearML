from clearml import Task
from clearml.automation.optuna import OptimizerOptuna

# Initialize the HPO task
task = Task.init(project_name="AI_Studio_Demo", task_name="HPO: Train Model")

# Get the actual training model task
BASE_TRAIN_TASK_ID = Task.get_task(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model").id

# Connect parameters
args = {
    'base_train_task_id': BASE_TRAIN_TASK_ID,
    'num_trials': 10,
    'time_limit_minutes': 60
}
task.connect(args)
task.execute_remotely()
# Configure the HPO process
optimizer = OptimizerOptuna(
    base_task_id=BASE_TRAIN_TASK_ID,  # Use the actual training model as base
    execution_queue="pipeline_controller",  # Specify the execution queue
    hyper_parameters=[
        {
            "name": "learning_rate",
            "type": "float",
            "min": 0.0001,
            "max": 0.01,
            "log": True
        },
        {
            "name": "weight_decay",
            "type": "float",
            "min": 1e-6,
            "max": 1e-3,
            "log": True
        },
        {
            "name": "batch_size",
            "type": "int",
            "min": 16,
            "max": 64,
            "step": 16
        }
    ],
    objective_metric='validation_accuracy',  # Metric to optimize
    objective_metric_goal='maximize',        # Try to maximize validation accuracy
    num_concurrent_workers=2,               # Run 2 trials in parallel
    max_iteration_per_job=1,                # Each trial runs once
    total_max_jobs=args['num_trials'],      # Total number of trials to run
    project_name="AI_Studio_Demo",
    task_name="HPO: Train Model"
)

# Set time limit for the entire HPO process
optimizer.set_time_limit(in_minutes=args['time_limit_minutes'])

# Start the HPO process
# This will:
# 1. Clone the template task 10 times
# 2. Run each clone with different hyperparameters
# 3. Track which combination gives the best validation accuracy
# 4. Save the best parameters in the HPO task
optimizer.start()