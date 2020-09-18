# Quick-Deployables
Presenting quick deployable tweet-trained models to use for crisis response.

## Urgency/Priority Binary Classifier (Bert-based)

### Requirements
- Python3.6+ and ```pip install -r requirements.txt``` to install necessary packages.
- Download model from Google Drive [urgency.pt]()

#### How to run
```python predictor.py 'sample.txt'```
where ```sample.txt``` contains sentences.

#### Results on Crisis Data
We train using tweets from a set of crisis events and test using an unseen crisis.

| Target Crisis  | Accuracy in %  |
 :-: |  :-:
| Maria                   | 87.46 |
| Harvey                  | 86.08 |
| Florence                | 87.68 |
| Irma                    | 84.23 |
| Australia Bushfire      | 83.34 |
| Philipinnes Floods      | 89.22 |
| Alberta Floods          | 84.33 |
| Nepal Earthquake        | 91.05 |
| Typhoon Hagupit         | 82.81 |
| Chile Earthquake        | 88.74 |
| Joplin Tornado          | 86.21 |
| Typhoon Yolanda         | 87.37 |
| Queensland Floods       | 84.33 |
| Manila Floods           | 91.05 |
| Paris Attacks           | 82.81 |
| Italy Earthquakes       | 88.74 |
| Guatemala Earthquake    | 86.21 |
| Boston Bombings         | 87.37 |
| Florida School Shooting | 86.21 |
| Covid                   | 87.37 |
| **Average**		          | **86.54** |

### Multilingual Model (coming soon..)

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
