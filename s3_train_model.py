import matplotlib.pyplot as plt
from clearml import Task, Logger, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import time
import os
import pandas as pd
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Connecting ClearML with the current process
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model")
clearml_logger = Logger.current_logger()

# Connect parameters
args = {
    'processed_dataset_id': '',  # Will be set from pipeline
    'num_epochs': 20,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
}

task.connect(args)
logger.info("Connected parameters: %s", args)

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('General/processed_dataset_id')
logger.info(f"Received dataset ID from parameters: {dataset_id}")

if not dataset_id:
    logger.error("Processed dataset ID is null or empty")
    raise ValueError("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")

print('Retrieving Iris dataset')

# Load the dataset from ClearML
dataset = Dataset.get(dataset_id=dataset_id)
print(f"Loaded dataset: {dataset.name}")

# Get the dataframes
dataset_path = dataset.get_mutable_local_copy("X_train.csv")
X_train = pd.read_csv(os.path.join(dataset_path, "X_train.csv")).values

dataset_path = dataset.get_mutable_local_copy("X_test.csv")
X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values

dataset_path = dataset.get_mutable_local_copy("y_train.csv")
y_train = pd.read_csv(os.path.join(dataset_path, "y_train.csv")).values.ravel()

dataset_path = dataset.get_mutable_local_copy("y_test.csv")
y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values.ravel()

# Clean up temporary files and directories
for file in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
            logger.info(f"Cleaned up temporary directory: {file}")
        else:
            os.remove(file)
            logger.info(f"Cleaned up temporary file: {file}")

print('Iris dataset loaded successfully')

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

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    clearml_logger.report_scalar(title='train', series='epoch_loss', value=avg_loss, iteration=epoch)

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
    clearml_logger.report_scalar("validation", "accuracy", value=accuracy, iteration=0)

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