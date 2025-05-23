seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)

import opensmile
import json
from sklearn.multiclass import OneVsOneClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
import mlflow.sklearn
from joblib import dump

mlflow.set_tracking_uri('http://mlflow:5000')

label2id = {"anger": 0, "surprise": 1, "happiness": 2, "sadness": 3, "neutral": 4, "fear": 5}

features = []
emotions = []
with open('modified_shemo.json', encoding='utf-8') as ms:
    feature_extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True, num_workers=None,
        sampling_rate=16000, resample=True,
    )
    modified_shemo = json.loads(ms.read())
    for file in modified_shemo:
        path = modified_shemo[file]["path"]
        emotion = modified_shemo[file]["emotion"]
        if emotion != 'fear':
            df = feature_extractor.process_file(path)
            features.append(df)
            emotions.append(label2id[emotion])
X = np.array(features).squeeze()
y = np.array(emotions)

signature = mlflow.models.infer_signature(X, y)

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)

model = SVC()
ovo = OneVsOneClassifier(model)

space = dict()
space['estimator__C'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
space['estimator__gamma'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

search = BayesSearchCV(ovo, space, scoring='recall_macro', cv=cv_inner, n_jobs=-1, verbose=0)
pipeline = make_pipeline(StandardScaler(), search)

mlflow.set_experiment("svm-shemo-docker")
with mlflow.start_run(run_name="SVM-SER"):
    scores = {'test_accuracy': [], 'test_recall_macro': []}

    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)

        best_params = pipeline.named_steps['bayessearchcv'].best_params_
        best_score = pipeline.named_steps['bayessearchcv'].best_score_
        best_model = pipeline.named_steps['bayessearchcv'].best_estimator_

        y_pred = pipeline.predict(X_test)
        acc = np.mean(y_pred == y_test)
        recall = recall_score(y_test, y_pred, average='macro')

        with mlflow.start_run(run_name=f"Fold_{fold_idx + 1}", nested=True):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_inner_recall_score", best_score)
            mlflow.log_metric("best_outer_acc_score", acc)
            mlflow.log_metric("best_outer_recall_score", recall)
            mlflow.sklearn.log_model(best_model, "best_model", signature=signature)

        scores['test_accuracy'].append(acc)
        scores['test_recall_macro'].append(recall)

    pipeline.fit(X, y)
    final_model = pipeline.named_steps['bayessearchcv'].best_estimator_

    dump(final_model, "model.joblib")

    with mlflow.start_run(run_name="Final-Model", nested=True):
        mlflow.sklearn.log_model(final_model, "final_model", signature=signature)

    print('____________________ Support Vector Machine ____________________')
    print(f"Weighted Accuracy: {np.mean(scores['test_accuracy']) * 100:.2f}")
    print(f"Unweighted Accuracy: {np.mean(scores['test_recall_macro']) * 100:.2f}")
