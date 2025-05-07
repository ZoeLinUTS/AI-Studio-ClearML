from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, DiscreteParameterRange
import logging

# Initialize the HPO task
task = Task.init(
    project_name='AI_Studio_Demo',
    task_name='HPO: Train Model',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Get the actual training model task
BASE_TRAIN_TASK_ID = Task.get_task(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model").id

# Connect parameters
args = {
    'base_train_task_id': BASE_TRAIN_TASK_ID,
    'num_trials': 10,
    'time_limit_minutes': 60,
    'run_as_service': False
}
args = task.connect(args)

# Execute the task remotely
task.execute_remotely()

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
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

# Create the optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,  # Use the actual training model as base
    execution_queue="pipeline_controller",  # Specify the execution queue
    hyper_parameters=hyper_parameters,
    objective_metric_title="validation",
    objective_metric_series="accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,  # Run 2 trials in parallel
    max_iteration_per_job=1,           # Each trial runs once
    total_max_jobs=args['num_trials'], # Total number of trials to run
    project_name="AI_Studio_Demo",
    task_name="HPO: Train Model",
    time_limit_per_job=10,             # Time limit per job in minutes
    pool_period_min=0.1                # Check tasks every 6 seconds
)

# Set report period
optimizer.set_report_period(0.2)  # Report every 12 seconds

# Start the HPO process
optimizer.start(job_complete_callback=job_complete_callback)

# Set time limit for the entire HPO process
optimizer.set_time_limit(in_minutes=args['time_limit_minutes'])

# Wait until process is done
optimizer.wait()

# Get the top performing experiments
top_exp = optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])

# Make sure background optimization stopped
optimizer.stop()

print('We are done, good bye')