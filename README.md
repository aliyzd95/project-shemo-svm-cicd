# 🗣️ CI/CD-Powered Speech Emotion Recognition (SVM-SER)

This project presents a **Dockerized Speech Emotion Recognition (SER) system with a full CI/CD pipeline** using GitHub Actions. It classifies emotions (Anger, Surprise, Happiness, Sadness, Neutral) from speech audio using an SVM model with Opensmile features (eGeMAPSv02). MLflow is integrated for experiment tracking.

This repository enhances the original SVM-SER project by adding robust containerization and automated build, test, and deployment workflows.

**Original Project:** [aliyzd95/SVM-SER](https://github.com/aliyzd95/SVM-SER)

**Published Docker Image:** [aliyzd95/svm-ser on Docker Hub](https://hub.docker.com/r/aliyzd95/svm-ser)

---

## ✨ Key Features

* Classifies 5 emotions from speech audio.
* Utilizes `opensmile` (eGeMAPSv02 feature set).
* SVM model with Bayesian hyperparameter optimization.
* MLflow integration for experiment tracking.
* **Fully Dockerized:** Easy setup and reproducible environments using Docker & Docker Compose.
* **Automated CI/CD Pipeline (GitHub Actions):**
    * Automated testing of the prediction script (`test.py`) using a pre-trained model.
    * Automated Docker image build and push to Docker Hub on changes to the `main` branch.
* Pre-trained model (`model.joblib`) included for immediate testing and inference.

---

## 🛠️ Tech Stack

* **Python 3.10**
* **Machine Learning:** `scikit-learn` (SVM), `skopt`
* **Audio Feature Extraction:** `opensmile`
* **Experiment Tracking:** `MLflow`
* **Containerization:** `Docker`, `Docker Compose`
* **CI/CD & Automation:** `GitHub Actions`

---

## 🔄 CI/CD with GitHub Actions

This project leverages GitHub Actions for a robust Continuous Integration and Continuous Deployment (CI/CD) pipeline, defined in `.github/workflows/cicd.yml`. The pipeline ensures code quality and automates deployment:

1.  **Trigger:** The workflow is triggered on every `push` to the `main` branch.

2.  **`build-and-test` Job:**
    * **Checkout Code:** Fetches the latest version of the repository.
    * **Set up Python:** Configures the Python environment.
    * **Install Dependencies:** Installs system dependencies (like `ffmpeg`, `libsndfile1` for Opensmile) and Python packages from `requirements.txt`.
    * **Run Test (`python test.py`):** Executes a crucial test script (`test.py`). This script loads the pre-trained `model.joblib` and performs a prediction on a sample `test.wav` file. This step verifies the integrity of the model and the prediction pipeline. A failure here indicates an issue with the model or the prediction environment.

3.  **`build-and-push` Job:**
    * **Dependency:** This job only runs if the `build-and-test` job completes successfully.
    * **Docker Login:** Securely logs into Docker Hub using secrets stored in GitHub.
    * **Build Docker Image:** Builds a new Docker image of the application tagged as `latest` and with the username (e.g., `aliyzd95/svm-ser:latest`).
    * **Push Docker Image:** Pushes the newly built and tested Docker image to Docker Hub, making it available for deployment.

This automated workflow ensures that every change integrated into the `main` branch is automatically tested, and a deployable Docker image is published, streamlining the development and deployment process.

---

## 🏁 Quick Start

### Prerequisites

* [Git](https://git-scm.com/downloads)
* [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
* **[modified-shemo](https://github.com/aliyzd95/modified-shemo) Audio Dataset:**
    * Place your audio files in a `./shemo/` directory at the project root.
    * Ensure `modified_shemo.json` correctly paths to these audio files.

### Running with Docker

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aliyzd95/project-shemo-svm-cicd.git
    cd project-shemo-svm-cicd
    ```

2.  **Prepare Dataset:**
    * Create a `shemo` directory in the project root and place your ShEMO audio files inside it.
    * Verify that `path` entries in `modified_shemo.json` point to correct locations (e.g., `shemo/your_audio_file.wav`).

3.  **Build and run services (Application & MLflow):**
    This will start the training process (if `main.py` is the default command in the Dockerfile) and the MLflow tracking server.
    ```bash
    docker-compose up --build
    ```
    To run in detached mode: `docker-compose up --build -d`

4.  **Access MLflow UI:**
    Open your browser and go to [http://localhost:5000](http://localhost:5000).

---

## 🚀 Usage

### Model Training (Optional, if re-training is needed)

* Training is typically initiated when the `svm-ser` service starts via `docker-compose up`.
* Experiments, parameters, metrics, and models are logged to MLflow.
* The final trained model is saved as `model.joblib` and as an MLflow artifact.
* To re-run training manually:
    ```bash
    docker-compose exec svm-ser python main.py
    ```

### Prediction (Using the pre-trained model)

* The `test.py` script uses the pre-trained `model.joblib` to predict emotion from `test.wav`. This is the same script used in the CI pipeline for testing.
* To run prediction via Docker:
    ```bash
    docker-compose exec svm-ser python test.py
    ```
