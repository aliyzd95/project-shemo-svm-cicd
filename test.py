import json
import random
import opensmile
# import numpy as np
import joblib

label2id = {"anger": 0, "surprise": 1, "happiness": 2, "sadness": 3, "neutral": 4}
id2label = {v: k for k, v in label2id.items()}

# load model
model = joblib.load("model.joblib")

# load dataset
with open("modified_shemo.json", encoding="utf-8") as f:
    data = json.load(f)

# pick one sample randomly
sample = random.choice(list(data.values()))
file_path = sample["path"]
true_emotion = sample["emotion"]

# extract features
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    sampling_rate=16000, resample=True
)
features = smile.process_file(file_path).values.squeeze().reshape(1, -1)

# predict
predicted_label = model.predict(features)[0]
predicted_emotion = id2label[predicted_label]

print("True emotion:", true_emotion)
print("Predicted emotion:", predicted_emotion)

assert predicted_emotion in label2id, "Predicted emotion not valid"
