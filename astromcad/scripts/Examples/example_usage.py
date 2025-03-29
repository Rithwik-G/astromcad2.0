from astromcad import Detect
from example_data import lc, host
import numpy as np

Detect.pad(lc) # Requires type(lc) == list

host=np.array(host)
lc=np.array(lc)

print('-------------Using Contextual Info-------------')
Detect.init(host=True)

print("Classification Output: ", Detect.predict([lc, host]))
print("Anomaly Score: ", Detect.anomaly_score([lc, host]))
print("Individual Anomaly Scores: ", Detect.score_discrete([lc, host]))
print("Classes: ", Detect.classes)

print('-------------No Contextual Info-------------')
Detect.init(host=False)

print("Classification Output: ", Detect.predict(lc))
print("Anomaly Score: ", Detect.anomaly_score(lc))
print("Individual Anomaly Scores: ", Detect.score_discrete(lc))
print("Classes: ", Detect.classes)