
# Indian Sign Language Interpretor

An AI powered system that can interpret Indian (or any) Sign Language.
Just train the model with any Sign Language that you want and it'll be able to interpret it.

## How does it work?

1. Two separate datasets are created.
2. Each dataset is divided into two parts, i.e. the training data set and the testing dataset.
3. The training data of each dataset is used to train different Machine Learning models and the accuracy of each model is shown here.
4. The webcam feed is given as input to the Mediapipe API and using it’s returned value, it is decided whether the user is showing gestures using one hand or two hands.
5. The data is sent to the corresponding trained Kernel SVM model which returns the predicted alphabets that are then displayed on the screen.

### Flowchart
![Webcam feed](https://github.com/Nanak360/ai_asl_interpreter/assets/36790357/fd9192eb-b9d2-413b-b475-c1d219c5a98e)

### Accuracy report
![Screenshot from 2023-07-08 22-13-29](https://github.com/Nanak360/ai_asl_interpreter/assets/36790357/a30ac6d5-f84c-4842-8076-8daf2043dca5)

### Screenshots
![Untitled design](https://github.com/Nanak360/ai_asl_interpreter/assets/36790357/96247e77-7079-4f20-ac7a-cc2b7687198a)

## Local installation and usage

1. Setup the app locally

```bash
  python3 -m venv env

  source env/bin/activate

  pip install -r requirements.txt
```
2. Collect dataset
```bash
python3 scripts/collect_data.py
```
3. Train the model with the dataset
```bash
python3 scripts/train.py
```
4. Train the model with the dataset
```bash
python3 scripts/predict.py
```