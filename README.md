# Quick-Deployables
Presenting quick deployable models (trained using crisis tweets) to filter messages during a crisis response.

## Priority Classifier (Bert-based)
A model that predicts binary labels 0 or 1 (1 = high priority/urgent; 0 = rest) for a given set of input sentences.

```Input:``` A text file

```Output:``` Predictions as a list of ```0```s and ```1```s

### Requirements
- Python3.6+ and ```pip install -r requirements.txt``` to install necessary packages.
- Download the pytorch model from our Google Drive [urgency_en.pt]() to the current folder.
- Input sentences or tweets should be in text format as shown in ```sample.txt```

### How to run
```python predictor.py 'sample.txt'```

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
x = predict('sample.txt')
```

#### Results on Crisis Data
We train using tweets from a set of crisis events and test using an unseen crisis. Our dataset is collected from various sources such as [CitizenHelper](https://ist.gmu.edu/~hpurohit/informatics-lab/icwsm17-citizenhelper.html) and [TREC](http://dcs.gla.ac.uk/~richardm/TREC_IS/).

| Target Crisis  | Accuracy in %  | Recall in % |
 :-: |  :-: |  :-:
| Maria                   | 87.46 | 87.46 |
| Harvey                  | 86.08 | 87.46 |
| Florence                | 87.68 | 87.46 |
| Irma                    | 84.23 | 87.46 |
| Australia Bushfire      | 83.34 | 87.46 |
| Philipinnes Floods      | 89.22 | 87.46 |
| Alberta Floods          | 84.33 | 87.46 |
| Nepal Earthquake        | 91.05 | 87.46 |
| Typhoon Hagupit         | 82.81 | 87.46 |
| Chile Earthquake        | 88.74 | 87.46 |
| Joplin Tornado          | 86.21 | 87.46 |
| Typhoon Yolanda         | 87.37 | 87.46 |
| Queensland Floods       | 84.33 | 87.46 |
| Manila Floods           | 91.05 | 87.46 |
| Paris Attacks           | 82.81 | 87.46 |
| Italy Earthquakes       | 88.74 | 87.46 |
| Guatemala Earthquake    | 86.21 | 87.46 |
| Boston Bombings         | 87.37 | 87.46 |
| Florida School Shooting | 86.21 | 87.46 |
| Covid                   | 87.37 | 87.46 |
| **Average**		          | **86.54** | **86.54** |

### Multilingual Model 
Use [urgency_ml.pt]() instead and change line ```7``` and ```8``` in ```predictor.py``` to:
```
hidden_layers = 105879
FNAME = "urgency_ml.pt"
```

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
