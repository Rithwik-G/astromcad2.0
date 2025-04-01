from utils.utils import *
from utils.load_data import *
from models import *
from tensorflow.keras.models import load_model
import os

if (__name__ == '__main__'):
    
    X_train, X_val, X_test, host_gal_train, host_gal_val, host_gal_test, y_train, y_val, y_test, class_weights, ntimesteps, x_data_anom, host_gal_anom, y_data_anom = get_data()
    y_class = np.array([classes[np.argmax(i)] for i in y_test])

    Detect.init(host=True)
    os.makedirs("scores", exist_ok=True)


    path = "../scores/norm_scores.csv"
    Detect.generate_score_csv([X_test, host_gal_test], path, y_class)
    print("Normal Class Scores Generated")

    path = "../scores/anom_scores.csv"
    Detect.generate_score_csv([x_data_anom, host_gal_anom], path, y_data_anom)
    print("Anom Class Scores Generated")
    
    