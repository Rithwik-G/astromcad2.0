from astromcad import *
from astromcad.scripts.utils.figures import *
from astromcad.scripts.utils.utils import *
from astromcad.scripts.utils.load_data import *
import os
import random

script_dir = 'astromcad'

if __name__ == '__main__':

    # train_classifier.py
    X_train, X_val, X_test, host_gal_train, host_gal_val, host_gal_test, y_train, y_val, y_test, class_weights, ntimesteps, x_data_anom, host_gal_anom, y_data_anom = get_data()

    print("Building Model")
    # model = build_model(100, ntimesteps, y_train.shape[1], contextual=0, n_features = 4)
    model = build_model(100, ntimesteps, y_train.shape[1], contextual=2, n_features = 4)
    print(model.summary())
    
    print("Training")
    # history = train(model, X_train, y_train, X_val, y_val, class_weights, epochs=40)
    # history = train(model, X_train, host_gal_train, y_train, X_val, host_gal_val, y_val, class_weights, epochs=40)
    
    # model.save(os.path.join(script_dir, "Models/NoHostClassifier.h5"))
    # model.save(os.path.join(script_dir, "Models/HostClassifier.h5"))
    
    print("Model Saved")

    # train_mcif.py
    y_single = np.array([np.argmax(i) for i in y_train])

    anomaly_detector = mcad(model, 'latent', ['lc', 'host'])
    anomaly_detector.create_encoder()
    anomaly_detector.init_mcif([X_train, host_gal_train], y_single)

    # save(os.path.join(script_dir, 'Models/HostMCIF.pickle'), anomaly_detector.mcif)

    # generate_scores.py
    os.makedirs(os.path.join(script_dir, "scores"), exist_ok=True)
    y_single_test = np.array([np.argmax(i) for i in y_test])

    path = os.path.join(script_dir, "scores/norm_scores.csv")
    anomaly_detector.generate_score_csv([X_test, host_gal_test], path, y_single_test)
    print("Normal Class Scores Generated")

    path = os.path.join(script_dir, "scores/anom_scores.csv")
    anomaly_detector.generate_score_csv([x_data_anom, host_gal_anom], path, y_data_anom)
    print("Anom Class Scores Generated")

    # generate_plots.py

    norm_results = pd.read_csv(os.path.join(script_dir, "scores/norm_scores.csv"))
    anom_results = pd.read_csv(os.path.join(script_dir, "scores/anom_scores.csv"))

    plot_recall(list(norm_results['score']), random.sample(list(anom_results['score']), 100))
    plt.savefig(os.path.join(script_dir, "figures/recall.pdf"), bbox_inches='tight')
    plt.show()
    print("Generated Recall Plot")

    median_score(norm_results, anom_results)
    plt.savefig(os.path.join(script_dir, "figures/median_score.pdf"), bbox_inches='tight')
    plt.show()
    print("Generated Median Score Plot")

    distribution(norm_results, anom_results)
    plt.savefig(os.path.join(script_dir, "figures/distribution.pdf"), bbox_inches='tight')
    plt.show()
    print("Generated Anomaly Distribution Plot")
