# 🧠 AI-Studio-ClearML

This repository provides a minimal, reproducible example of how to use [ClearML](https://clear.ml) to build machine learning pipelines, track experiments, and manage datasets using both **task-based pipelines** and **function-based pipelines**.

---

## 📦 Project Structure

```
├── model_artifacts/                  # Example outputs or saved models
├── work_dataset/                     # Dataset samples and usage examples
├── demo_functions.py                 # Base Functions from ClearML 
├── demo_using_artifacts_example.py  # Demonstrates artifact loading
├── main.py                           # Entry point
├── pipeline_from_tasks.py           # Pipeline built from existing ClearML Tasks
├── step1_dataset_artifact.py        # Step 1: Upload dataset as artifact
├── step2_data_preprocessing.py      # Step 2: Preprocess dataset
├── step3_train_model.py             # Step 3: Train model using preprocessed data
```

---

## 🧪 Features

- ✅ Task-based pipeline using `PipelineController.add_step(...)`
- [TBD] Function-based pipeline using `PipelineController.add_function_step(...)`
- ✅ Reusable ClearML Task templates
- ✅ Dataset and model artifact management with ClearML
- ✅ End-to-end ML workflow: Dataset → Preprocessing → Training
- ✅ Fully compatible with ClearML Hosted and ClearML Server

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install clearml
```

### 2. Configure ClearML

Set up ClearML by running:

```bash
clearml-init
```

You will be prompted to enter:
- ClearML Credential

Use [https://app.clear.ml](https://app.clear.ml) to register for a free account if needed.

---

### 3. Create a ClearML Agent
- Install the ClearML agent on your machine or server.
```bash
pip install clearml-agent
```
---

## 🛠️ How to Use

### Using Colab: refer to ClearML_Pipeline_Demo.ipynb.

### 🔁 Option 1: Pipeline from Predefined ClearML Tasks

To use a task-based pipeline, follow these steps:

#### Step 1: Register the Base Tasks

**Before running the pipeline, execute the following scripts \*\*once\*\* to create reusable ClearML Tasks:**
> **Note:** When running for the first time, comment out \`task.execute_remotely()\` in the each .py file of the three tasks to successfully create a task template.

```bash
# Step 1: Upload dataset
python step1_dataset_artifact.py

# Step 2: Preprocess dataset
python step2_data_preprocessing.py

# Step 3: Train model
python step3_train_model.py
```

These will appear in your ClearML dashboard and serve as base tasks for the pipeline.

#### Step 1.5: Initial ClearML Queue
Create Queue with name as pipeline (or your customized one), ensure it is consistent in pipeline_from_tasks.py
```
pipe.start(queue="pipeline")
```
![image](https://github.com/user-attachments/assets/f650dc61-8f44-48a4-ac17-8b8bd1087133)

Run the agent for queue worker:
![image](https://github.com/user-attachments/assets/b7edfe7e-6c35-4155-a8cd-45217cf48399)


#### Step 2: Run the Pipeline

Once all base tasks are registered, run the pipeline:

```bash
python main.py # Where we execute the run_pipeline()
```

---

### 🔧 [TBD] Option 2: Pipeline from Local Python Functions 

This version demonstrates using `add_function_step(...)` to wrap Python logic as pipeline steps.

---

### 🧩 Run Individual Pipeline Steps

You can run each task separately as well:

> **Note:** When running for the first time, comment out `task.execute_remotely()` in the code file to successfully create a task template.
> 
```bash
# Step 1: Upload dataset
python step1_dataset_artifact.py

# Step 2: Preprocess data
python step2_data_preprocessing.py

# Step 3: Train model
python step3_train_model.py
```


---

## 📘 References

- [ClearML Documentation](https://clear.ml/docs)
- [ClearML Pipelines Guide](https://clear.ml/docs/latest/docs/getting_started/building_pipelines)
- [ClearML GitHub](https://github.com/allegroai/clearml)

---

## 🙌 Acknowledgments

This project is developed and maintained by:

- **Jacoo-Zhao** (GitHub: [@Jacoo-Zhao](https://github.com/Jacoo-Zhao))
- **Zoe Lin** (Github: [@Zoe Lin](https://github.com/ZoeLinUTS))

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
