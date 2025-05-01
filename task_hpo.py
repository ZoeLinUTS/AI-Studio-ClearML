from clearml.automation.optuna import OptimizerOptuna

optimizer = OptimizerOptuna(
    base_task_id='your_base_train_task_id',
    hyper_parameters=[
        {'name': 'Model/lr', 'type': 'float', 'min': 0.0001, 'max': 0.01},
        {'name': 'Model/dropout', 'type': 'float', 'min': 0.1, 'max': 0.5},
    ],
    objective_metric_title='val_accuracy',
    objective_metric_series='score',
    objective_metric_sign='max',
    max_iteration=20,
    execution_queue='pipeline',
    project_name='examples',
    task_name='HPO_Demo',
)
optimizer.set_time_limit(in_minutes=60)
optimizer.start()