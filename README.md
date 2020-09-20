# Quick-Deployables
Presenting quick deployable models (that are trained using tweets collected from 20 different crisis events labeled for priority/urgency) to filter critical messages during a crisis response.

## Priority Classifier
A bert-based model that predicts binary labels ```0``` or ```1``` (```1``` = high priority/urgent; ```0``` = rest) for a given set of input sentences. 

```Input:``` A text file

```Output:``` Predictions as a list of ```0```s and ```1```s

To be specific, we use ```DistilBert``` for English and ```bert-base-multilingual-uncased``` for multilingual.

### Requirements
- Python3.6+ and ```pip install -r requirements.txt``` to install necessary packages.
- Download the pytorch model from our Google Drive [urgency_en.pt](https://drive.google.com/file/d/1a2xFP8RVF0QE4qk7sW5rOww5EWM9FkL-/view?usp=sharing) to the current folder.
- Input sentences or tweets should be in text format as shown in ```sample.txt```

### How to run
```python predictor.py 'sample.txt' 768 'en' 'urgency_en.pt'```

Sample Output: ```[0,1]```

corresponding to the following two sentences in ```sample.txt```:
```
Hello there.
Please help, I'm near the train station.
```

#### Calling from another code
If you need to call from another code to return predictions as a numpy array
```
from predictor import predict
x = predict('sample.txt' 768 'en' 'urgency_en.pt')
```

#### Results on Crisis Data
We train using tweets from a set of crisis events and test using a **fully unseen** crisis. For example, when the target crisis is ```Maria```, we train using tweets from rest of all crises and test on tweets from ```Maria```. Our dataset is collected from various sources such as [CitizenHelper](https://ist.gmu.edu/~hpurohit/informatics-lab/icwsm17-citizenhelper.html) and [TREC](http://dcs.gla.ac.uk/~richardm/TREC_IS/).

| Target Crisis  | Accuracy  | Recall |
 :-: |  :-: |  :-:
| Maria                   | 0.87 | 0.83 |
| Harvey                  | 0.75 | 0.72 |
| Florence                | 0.92 | 0.89 |
| Irma                    | 0.67 | 0.59 |
| Australia Bushfire      | 0.81 | 0.74 |
| Philipinnes Floods      | 0.71 | 0.68 |
| Alberta Floods          | 0.77 | 0.68 |
| Nepal Earthquake        | 0.78 | 0.80 |
| Typhoon Hagupit         | 0.82 | 0.82 |
| Chile Earthquake        | 0.74 | 0.64 |
| Joplin Tornado          | 0.74 | 0.71 |
| Typhoon Yolanda         | 0.87 | 0.86 |
| Queensland Floods       | 0.76 | 0.73 |
| Manila Floods           | 0.65 | 0.57 |
| Paris Attacks           | 0.87 | 0.84 |
| Italy Earthquakes       | 0.73 | 0.70 |
| Guatemala Earthquake    | 0.67 | 0.64 |
| Boston Bombings         | 0.82 | 0.82 |
| Florida School Shooting | 0.75 | 0.66 |
| Covid                   | 0.77 | 0.73 |
| **Average**		          | **0.77** | **0.73** |

#### Multilingual
Use [urgency_ml.pt]() instead.
```python predictor.py 'sample.txt' 105879 'ml' 'urgency_ml.pt'```

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
