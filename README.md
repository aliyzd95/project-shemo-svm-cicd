# 🗣️ Speech Emotion Recognition (SVM-SER) with CI/CD pipeline

[![CI/CD Pipeline](https://github.com/aliyzd95/project-shemo-svm-cicd/actions/workflows/cicd.yml/badge.svg)](https://github.com/aliyzd95/project-shemo-svm-cicd/actions/workflows/cicd.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/aliyzd95/svm-ser.svg)](https://hub.docker.com/r/aliyzd95/svm-ser)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aliyzd95/project-shemo-svm-cicd/blob/main/LICENSE) A Speech Emotion Recognition (SER) system using SVM with Opensmile features (eGeMAPSv02) for emotion classification from audio. This project integrates MLflow for experiment tracking and is containerized using Docker for easy setup and deployment.

**Docker Hub Image:** [https://hub.docker.com/r/aliyzd95/svm-ser](https://hub.docker.com/r/aliyzd95/svm-ser)

---

## ✨ Features

* Classifies 5 emotions from speech audio.
* Utilizes `opensmile` (eGeMAPSv02 feature set).
* SVM model with Bayesian hyperparameter optimization.
* MLflow integration for experiment tracking.
* Dockerized for easy setup and reproducibility.
* CI/CD pipeline with GitHub Actions for automated testing and Docker image publishing.
* Pre-trained model (`model.joblib`) included.

---

## 🛠️ Tech Stack

* **Python 3.10**
* **Machine Learning:** `scikit-learn` (SVM), `skopt`
* **Audio Feature Extraction:** `opensmile`
* **Experiment Tracking:** `MLflow`
* **Containerization:** `Docker`, `Docker Compose`
* **CI/CD:** `GitHub Actions`

---

## 🏁 Quick Start

### Prerequisites

* [Git](https://git-scm.com/downloads)
* [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
* **ShEMO Audio Dataset:**
    * Place your audio files in a `./shemo/` directory at the project root.
    * Ensure `modified_shemo.json` correctly paths to these audio files.

### Running with Docker

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aliyzd95/project-shemo-svm-cicd.git](https://github.com/aliyzd95/project-shemo-svm-cicd.git)
    cd project-shemo-svm-cicd
    ```

2.  **Prepare Dataset:**
    * Create a `shemo` directory in the project root and place your ShEMO audio files inside it.
    * Verify that the `path` entries in `modified_shemo.json` point to the correct locations within the `shemo` directory (e.g., `shemo/your_audio_file.wav`).

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

### Model Training

* Training is typically initiated when the `svm-ser` service starts via `docker-compose up`.
* Experiments, parameters, metrics, and models are logged to MLflow.
* The final trained model is saved as `model.joblib` in the project root and as an MLflow artifact.
* To re-run training manually within the container:
    ```bash
    docker-compose exec svm-ser python main.py
    ```

### Prediction

* The `test.py` script uses the pre-trained `model.joblib` to predict emotion from `test.wav`.
* To run prediction using the Docker container:
    ```bash
    docker-compose exec svm-ser python test.py
    ```
    *(Ensure `test.wav` is present in the project root inside the container, which it should be by default.)*

---

## 🔄 CI/CD

A GitHub Actions workflow (`.github/workflows/cicd.yml`) is configured for:
* **Automated Testing:** Runs `python test.py` on pushes to the `main` branch.
* **Docker Image Publishing:** Builds and pushes the Docker image to [Docker Hub](https://hub.docker.com/r/aliyzd95/svm-ser) after successful tests on the `main` branch.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/aliyzd95/project-shemo-svm-cicd/blob/main/LICENSE) file for details.
