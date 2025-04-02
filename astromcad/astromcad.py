import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GRU, Masking, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


SCRIPT_DIR =os.path.dirname(os.path.abspath(__file__))

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

class mcif:
    def __init__(self, n_estimators = 100):
        self.n_estimators=n_estimators
    
    def train(self, x_data, labels):

        self.classes = np.unique(labels)
        self.iforests = [IsolationForest(n_estimators=self.n_estimators) for i in self.classes]
        
        for ind, class_name in enumerate(self.classes):
            here = []
            for i in range(len(x_data)):
                if (class_name == labels[i]):
                    here.append(x_data[i])

            self.iforests[ind].fit(here)
            

    def score_discrete(self, data):
        scores = [-det.decision_function(data) for det in self.iforests]

        scores = np.array(scores)
        scores = scores.T

        return scores

    def score(self, data):
        return [np.min(i) for i in self.score_discrete(data)]

class mcad:
    def __init__(self, model, latent_name, input_names):
        self.model = model
        self.input_names = input_names
        self.latent_name = latent_name
        self.input_layers = [self.model.get_layer(i).input for i in self.input_names]
    
    def create_encoder(self):
        self.encoder = Model(inputs=self.input_layers, outputs=self.model.get_layer(self.latent_name).output)

    def predict(self, input_data):
        return self.model.predict(x = input_data)

    def encode(self, input_data):
        return self.encoder.predict(x = input_data)
    
    def init_mcif(self, input_data, y_data, n_estimators=100):
        self.mcif = mcif(n_estimators)
        self.mcif.train(self.encode(input_data), y_data)

    def score(self, input_data):
        return self.mcif.score(self.encode(input_data))

    def score_discrete(self, data):
        return self.mcif.score_discrete(self.encode(data))

    def generate_score_csv(self, data, file_path, y=None):
        print(f"Saving scores to: {file_path}")
        scores = self.score(data)
        print("Generated Scores")
        with open(file_path, 'w') as f:
            ids = range(len(scores))
            if y is not None:
                df = pd.DataFrame({'obj_id': ids, 'score': scores, 'class': y})
            else:
                df = pd.DataFrame({'obj_id': ids, 'score': scores})
            df.to_csv(f, index=False)

class Detect:
    ntimesteps=656
    classes = ['SNIa', 'SNIa-91bg', 'SNIax', 'SNIb', 'SNIc', 'SNIc-BL', 'SNII', 'SNIIn', 'SNIIb', 'TDE', 'SLSN-I', 'AGN']

    @classmethod
    def pad(cls, x_data):
        for ind in range(len(x_data)):
            x_data[ind] = np.pad(x_data[ind], ((0, cls.ntimesteps - len(x_data[ind])), (0, 0)))

    @classmethod
    def init(cls, host):
        print("Initialized Model...")
        if (host):
            model_path = os.path.join(SCRIPT_DIR, os.path.join("Models", "HostClassifier.h5"))
            model = keras.models.load_model(model_path)

            cls.anomaly_detector = mcad(model, 'latent', ['lc', 'host'])
            cls.anomaly_detector.create_encoder()
            cls.anomaly_detector.mcif = load(os.path.join(SCRIPT_DIR, os.path.join("Models", "HostMCIF.pickle")))
        else:
            model_path = os.path.join(SCRIPT_DIR, os.path.join("Models", "NoHostClassifier.h5"))
            model = keras.models.load_model(model_path)

            cls.anomaly_detector = mcad(model, 'latent', ['lc'])
            cls.anomaly_detector.create_encoder()
            cls.anomaly_detector.mcif = load(os.path.join(SCRIPT_DIR, os.path.join("Models", "NoHostMCIF.pickle")))


    @classmethod
    def predict(cls, input_data):
        return cls.anomaly_detector.predict(input_data)

    @classmethod
    def anomaly_score(cls, input_data):
        return cls.anomaly_detector.score(input_data)

    @classmethod
    def score_discrete(cls, data):
        discrete_scores = cls.anomaly_detector.score_discrete(data)
        return discrete_scores

    @classmethod
    def generate_score_csv(cls, data, file_path, y=None):
        cls.anomaly_detector.generate_score_csv(data, file_path, y)



def build_model(latent_size, ntimesteps, num_classes, contextual, n_features=4):
    input_1 = Input((ntimesteps, n_features), name='lc')  # X.shape = (Nobjects, Ntimesteps, 4) CHANGE
    masking_input1 = Masking(mask_value=0.)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh')(lstm1)

    dense1 = Dense(100, activation='tanh')(lstm2)

    if (contextual == 0):
        merge1 = dense1
    else:
        input_2 = Input(shape = (contextual, ), name='host') # CHANGE
        dense2 = Dense(10)(input_2)
        merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu')(merge1)

    dense4 = Dense(latent_size, activation='relu', name='latent')(dense3)

    output = Dense(num_classes, activation='softmax')(dense4)

    if (contextual == 0):
        model = keras.Model(inputs=input_1, outputs=output)
    else:
        model = keras.Model(inputs=[input_1, input_2], outputs=output)

    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    return model

def train_contextual(model, X_train, host_gal_train, y_train, X_val, host_gal_val, y_val, class_weights, epochs=40):
    early_stopping = EarlyStopping(patience=5, min_delta=0.001,monitor="val_loss",restore_best_weights=True)
    return model.fit(x = [X_train, host_gal_train], validation_data=([X_val, host_gal_val], y_val), y = y_train, epochs=epochs, batch_size = 128, class_weight = class_weights, callbacks=[early_stopping])
    
def train(model, X_train, y_train, X_val, y_val, class_weights, epochs=40):
    early_stopping = EarlyStopping(patience=5, min_delta=0.001,monitor="val_loss",restore_best_weights=True)
    return model.fit(x = X_train, validation_data=(X_val, y_val), y = y_train, epochs=epochs, batch_size = 128, class_weight = class_weights, callbacks=[early_stopping])


# Library code for figures

def median_score(maj_csv, anom_csv, title="Median Anomaly Score"):
    # Housekeeping from csv
    y_data_maj = maj_csv['class']
    scores_maj = maj_csv['score']
    y_data_anom = anom_csv['class']
    scores_anom = anom_csv['score']

    maj_classes = list(np.unique(y_data_maj))
    anom_classes = list(np.unique(y_data_anom))

    all_classes = maj_classes + anom_classes

    # Generate the median scores for each class
    
    score_dist = {i : [] for i in all_classes}

    for i in range(len(y_data_maj)):
        score_dist[y_data_maj[i]].append(scores_maj[i])
    for i in range(len(y_data_anom)):
        score_dist[y_data_anom[i]].append(scores_anom[i])

    for key in score_dist.keys():
        score_dist[key] = np.median(score_dist[key])

    # Make a pretty plot
    fig, ax = plt.subplots(figsize=(13, 13))
    
    averages = list(score_dist.values())

    cmap = matplotlib.cm.Blues(np.linspace(0,1,100))
    cmap = matplotlib.colors.ListedColormap(cmap[25:75,:-1])

    im = ax.imshow([averages], cmap=cmap)

    ax.set_yticks([])
    ax.set_xticks(range(len(averages)), list(score_dist.keys()), fontsize=15, rotation=45)
    for x in range(len(averages)):
      ax.annotate(str(round(averages[x], 2)), xy=(x, 0),
                  horizontalalignment='center',
                  verticalalignment='center', fontsize=15, fontweight = "bold" if (x > len(maj_classes)) else "normal")
    ax.set_title(title, fontsize=20)


def distribution(maj_csv, anom_csv, title='Anomaly Score Distribution'): # Same input as average/median_score
    # Housekeeping from csv
    y_data_maj = maj_csv['class']
    scores_maj = maj_csv['score']
    y_data_anom = anom_csv['class']
    scores_anom = anom_csv['score']

    maj_classes = list(np.unique(y_data_maj))
    anom_classes = list(np.unique(y_data_anom))

    all_classes = maj_classes + anom_classes

    # Generate distribution plot
    color = ['#ADD8E6'] * 12 + ['#FF6645'] * 5
    
    x=[]
    g=[]

    for i in range(len(scores_maj)):
        g.append(y_data_maj[i])
        x.append(scores_maj[i])

    for i in range(len(scores_anom)):
        g.append(y_data_anom[i])
        x.append(scores_anom[i])

    name_to_index = {name: i for i, name in enumerate(all_classes)}

    df = pd.DataFrame(dict(x=x, g=g))
    df.sort_values('g', inplace=True, key=lambda x: x.map(name_to_index))


    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=color)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping, but we do color = 'blue'
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlabel="Anomaly Score")

    g.despine(bottom=True, left=True)
    g.fig.suptitle(title, fontsize=23)



def plot_recall(maj_scores, anom_scores, title="Anomalies Detected by Index"):  
    
    # Generate the recall curve
    all_scores = []
    for i in maj_scores:
        all_scores.append((i, 0))
    for i in anom_scores:
        all_scores.append((i, 1))

    all_scores = sorted(all_scores, key=lambda x: x[0], reverse=True)
    all_scores = all_scores[:2000]
    prefix_sum = [0]

    for i in all_scores:
        prefix_sum.append(prefix_sum[-1] + i[1])
    
    fig, ax = plt.subplots()

    ax.set_xlim(0, 2000)
    ax.set_xlabel("Index (Top 2000 Scores)", fontsize=18)
    ax.set_ylabel("Recall", fontsize=18)

    ax.set_title(title, fontsize=21)

    ax2 = ax.twinx()

    ax2.set_ylabel('Detected Anomalies', fontsize=16)

    # Guessing line
    x = np.array(range(0,2000))
    y = len(minority) / (len(majority) + len(minority)) * x
    plt.plot(x, y, label='Guessing', linestyle='dashed', color='grey')

    ax.plot(range(0, 2000), [i / len(anom_scores) for i in prefix_sum[1:]], color='orange', label='Model')
    ax2.plot(range(0, 2000), [i for i in prefix_sum[1:]], color='orange')

    margin = 0.05
    ax.set_ylim(-margin, 1 + margin)
    ax2.set_ylim(len(anom_scores) * -margin, (1 + margin) * len(anom_scores))
    

    plt.legend(fontsize=14)
    plt.tight_layout()