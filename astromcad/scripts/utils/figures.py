import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib


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
