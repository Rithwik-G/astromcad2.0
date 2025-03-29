from utils.utils import *
from utils.load_data import *
from astromcad import *
from tensorflow.keras.models import load_model
if (__name__ == '__main__'):
    
    X_train, X_val, X_test, host_gal_train, host_gal_val, host_gal_test, y_train, y_val, y_test, class_weights, ntimesteps = get_data()
    y_single = np.array([np.argmax(i) for i in y_train])

    model = load_model("../Models/NoHostClassifier.h5")

    anomaly_detector = mcad(model, 'latent', ['lc'])
    anomaly_detector.create_encoder()
    anomaly_detector.init_mcif(X_train, y_single)

    save('../Models/NoHostMCIF.pickle', anomaly_detector.mcif)


    model = load_model("../Models/HostClassifier.h5")

    anomaly_detector = mcad(model, 'latent', ['lc', 'host'])
    anomaly_detector.create_encoder()
    anomaly_detector.init_mcif([X_train, host_gal_train], y_single)

    save('../Models/HostMCIF.pickle', anomaly_detector.mcif)
    