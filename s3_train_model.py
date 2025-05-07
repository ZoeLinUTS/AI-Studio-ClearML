import matplotlib.pyplot as plt
from clearml import Task, Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# Connecting ClearML with the current process,
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model")
logger = Logger.current_logger()

args = {
    'dataset_task_id': '',  # Will be set from pipeline
    'num_epochs': 20,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
}

task.connect(args)

# only create the task, we will actually execute it later
# task.execute_remotely() # After passing local testing, you should uncomment this command to initial task to ClearML

# Retrieve the HPO task ID from the pipeline
hpo_task_id = task.get_parameter("General/hpo_task_id")
hpo_task = Task.get_task(task_id=hpo_task_id)

# Retrieve optimized parameters from the HPO task
optimized_params = hpo_task.get_parameters()
args['learning_rate'] = optimized_params.get('learning_rate', args['learning_rate'])
args['weight_decay'] = optimized_params.get('weight_decay', args['weight_decay'])
args['batch_size'] = optimized_params.get('batch_size', args['batch_size'])

print('Retrieving Iris dataset')
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()
print('Iris dataset loaded')


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
# Hyperparameters
# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size=X_train.shape[1], num_classes=len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=args['learning_rate'],
    weight_decay=args['weight_decay']
)

for epoch in tqdm(range(args['num_epochs']), desc='Training Epochs'):
    epoch_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累积 loss
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    logger.report_scalar(title='train', series='epoch_loss', value=avg_loss, iteration=epoch)

# Save model
model_path = 'assets/model.pkl'
torch.save(model.state_dict(), model_path)
task.upload_artifact(name='model', artifact_object=model_path)
print('Model saved and uploaded as artifact')

# Load model for evaluation
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean().item()
    logger.report_scalar("validation", "accuracy", value=accuracy, iteration=0)

print(f'Model trained & stored with accuracy: {accuracy:.4f}')


# Plotting confusion matrix
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
y_test_names = [species_mapping[label.item()] for label in y_test]
predicted_names = [species_mapping[label.item()] for label in predicted]

cm = confusion_matrix(y_test_names, predicted_names, labels=list(species_mapping.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(species_mapping.values()))
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.savefig('figs/confusion_matrix.png')

print('Confusion matrix plotted and saved as confusion_matrix.png')