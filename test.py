import opensmile
import joblib

label2id = {"anger": 0, "surprise": 1, "happiness": 2, "sadness": 3, "neutral": 4}
id2label = {v: k for k, v in label2id.items()}

model = joblib.load("model.joblib")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    sampling_rate=16000, resample=True
)
features = smile.process_file('test.wav').values.squeeze().reshape(1, -1)

predicted_label = model.predict(features)[0]
predicted_emotion = id2label[predicted_label]

print("Predicted emotion:", predicted_emotion)

assert predicted_emotion in label2id, "Predicted emotion not valid"
