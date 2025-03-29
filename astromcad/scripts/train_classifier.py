from utils.utils import *
from utils.load_data import *
from astromcad import *



if (__name__ == '__main__'):
    
    X_train, X_val, X_test, host_gal_train, host_gal_val, host_gal_test, y_train, y_val, y_test, class_weights, ntimesteps = get_data()
    print(list(X_train[0]))
    print(list(host_gal_train[0]))
    assert(False)
    print("Building Model")
    model = build_model(100, ntimesteps, y_train.shape[1], contextual=0, n_features = 4)
    # model = build_model(100, ntimesteps, y_train.shape[1], contextual=2, n_features = 4)
    print(model.summary())
    
    print("Training")
    history = train(model, X_train, y_train, X_val, y_val, class_weights, epochs=40)
    # history = train(model, X_train, host_gal_train, y_train, X_val, host_gal_val, y_val, class_weights, epochs=40)
    
    model.save("Models/NoHostClassifier.h5")
    # model.save("Models/HostClassifier.h5")
    
    print("Model Saved")


