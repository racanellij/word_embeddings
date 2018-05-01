import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import plotly.graph_objs as go
import seaborn as sns
from matplotlib import colors
import plotly.plotly as py
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.metrics import confusion_matrix, classification_report,precision_score, precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
import keras
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical  
import keras.backend as K
from keras.callbacks import Callback

# personal preference
plt.style.use('seaborn')


def plot_weights_dnn(epochs,weights, X_tr, Y_tr, X_te, Y_te):
    accs = []
    recalls = []
    weights = np.array(weights)
    for weight in weights:
        history, recall, predictions = dnn_1024(X_over,Y_over,X_te,Y_te,25,weight, verbose = 0)
        accs.append(history.history['val_acc'][-1])
        recalls.append(recall.recalls[-1])
        
    plt.plot(weights,np.array(recalls),label='Recall Category 1')
    plt.plot(weights,np.array(accs),label = 'Accuracy')
    
    plt.xlabel('Weight')
    plt.ylabel('Score')
    plt.title('Model Statistics by Positive Ouput Weight')
    
    plt.legend()
    plt.show()
    plt.close()
    return
    

# Plots the validation and testing loss/accuracy for a given neural network
def dnn_plots(history, metrics = False):
    '''
        inputs:
            history: a fitted neural network
    '''
    history_dict = history.history
    losses = history_dict['loss']
    val_losses = history_dict['val_loss']
    val_recalls = metrics.recalls
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1,len(acc)+1)
    
    plt.plot(epochs, losses, label = 'Training Loss')
    plt.plot(epochs, val_losses, label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.plot(epochs, acc, label = 'Training Accuracy')
    plt.plot(epochs, val_acc, label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.clf()

    if metrics:
        plt.plot(epochs, val_acc, label = 'Validation Accuracy')
        plt.plot(epochs, val_recalls, label = 'Validation Recall')
        plt.title('Validation Recall and Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
        plt.clf()
    
    return

# Plots the precision, recall, and fscore for given probability weights
def plot_weights(clf, weights, X_tr, Y_tr, X_te, Y_te, recall = False, precision = False, fscore = False, title = False):
    '''
        inputs:
            clf: the classifier function 
            weights: the weights you want to test (list or numpy array)
            X_tr, Y_tr, X_te, Y_te: numpy arrays which contain the training and testing data
            recall: bool, will plot recall for each category
            precision: bool, will plot precision for each category
            fscore: bool, will plot fscore for each category
            if all are false will just plot all 3 for output category 1
    '''
    clf_placeholder = clf
    fscores_0 = []
    fscores_1 = []
    recalls_0 = []
    recalls_1 = []
    precisions_0 = []
    precisions_1 = []
    weights = np.array(weights)
    for weight in weights:
        clf.class_weight = {0:1., 1: weight}
        clf = clf.fit(X_tr,Y_tr)
        predictions = clf.predict(X_te)
        recalls,precisions,fscores,support = score(predictions, Y_te)
        precisions_0.append(precisions[0])
        precisions_1.append(precisions[1])
        recalls_0.append(recalls[0])
        recalls_1.append(recalls[1])
        fscores_0.append(fscores[0])
        fscores_1.append(fscores[1])
        clf = clf_placeholder
    if recall:
        plt.plot(weights,np.array(recalls_0),label='Recall Category 0')
        plt.plot(weights,np.array(recalls_1),label='Recall Category 1')
    elif precision:
        plt.plot(weights,np.array(precisions_0),label='Precision Category 0')
        plt.plot(weights,np.array(precisions_1),label='Precision Category 1')    
    elif fscore:
        plt.plot(weights,np.array(fscores_0),label='F-Score Category 0')
        plt.plot(weights,np.array(fscores_1),label='F-Score Category 1')        
    else:
        plt.plot(weights,np.array(recalls_1),label='Recall')
        plt.plot(weights,np.array(precisions_1),label='Precision')
        plt.plot(weights,np.array(fscores_1),label='F-Score')
    plt.xlabel('Weight')
    plt.ylabel('Score')
    if title:
        plt.title(title)
    else:
        plt.title('Model Statistics by Positive Ouput Weight')
    plt.legend()
    plt.show()
    plt.close()
    return


def plot_cs(clf, cs, X_tr, Y_tr, X_te, Y_te, recall = False, precision = False, fscore = False, acc = False):
    '''
        inputs:
            clf: the classifier function 
            cs: the cs you want to test (list or numpy array)
            X_tr, Y_tr, X_te, Y_te: numpy arrays which contain the training and testing data
            recall: bool, will plot recall for each category
            precision: bool, will plot precision for each category
            fscore: bool, will plot fscore for each category
            if all are false will just plot all 3 for output category 1
    '''
    clf_placeholder = clf
    fscores_0 = []
    fscores_1 = []
    recalls_0 = []
    recalls_1 = []
    precisions_0 = []
    precisions_1 = []
    accs = []
    cs = np.array(cs)
    for c in cs:
        clf.C = c
        clf = clf.fit(X_tr,Y_tr)
        predictions = clf.predict(X_te)
        recalls,precisions,fscores,support = score(predictions, Y_te)
        precisions_0.append(precisions[0])
        precisions_1.append(precisions[1])
        recalls_0.append(recalls[0])
        recalls_1.append(recalls[1])
        fscores_0.append(fscores[0])
        fscores_1.append(fscores[1])
        accs.append((predictions == Y_te).mean())
        clf = clf_placeholder
    if recall:
        plt.plot(cs,np.array(recalls_0),label='Recall Category 0')
        plt.plot(cs,np.array(recalls_1),label='Recall Category 1')
    elif precision:
        plt.plot(cs,np.array(precisions_0),label='Precision Category 0')
        plt.plot(cs,np.array(precisions_1),label='Precision Category 1')    
    elif fscore:
        plt.plot(cs,np.array(fscores_0),label='F-Score Category 0')
        plt.plot(cs,np.array(fscores_1),label='F-Score Category 1')    
    elif acc:
        plt.plot(cutoffs,np.array(recalls_1),label='Recall Category 1')
        plt.plot(cutoffs,np.array(accs),label='Model Accuracy')   
    else:
        plt.plot(cs,np.array(recalls),label='Recall')
        plt.plot(cs,np.array(precisions),label='Precision')
        plt.plot(cs,np.array(fscores),label='F-Score')
    plt.xlabel('Weight')
    plt.ylabel('Score')
    plt.title('Model Statistics by Positive Ouput Weight')
    plt.legend()
    plt.show()
    plt.close()
    return




# Plots the precision, recall, and fscore for given probability cutoffs
def plot_cutoffs(clf, cutoffs, X_tr, Y_tr, X_te, Y_te, recall = False, precision = False, fscore = False, acc = False):
    '''
        inputs:
            clf: the classifier function 
            cutoffs: the cutoffs you want to test (list or numpy array)
            X_tr, Y_tr, X_te, Y_te: numpy arrays which contain the training and testing data
            recall: bool, will plot recall for each category
            precision: bool, will plot precision for each category
            fscore: bool, will plot fscore for each category
            if all are false will just plot all 3 for output category 1
    '''
    fscores_0 = []
    fscores_1 = []
    recalls_0 = []
    recalls_1 = []
    accs = []
    precisions_0 = []
    precisions_1 = []
    cutoffs = np.array(cutoffs)
    clf = clf.fit(X_tr,Y_tr)
    predict_proba = clf.predict_proba(X_te)
    for cutoff in cutoffs:
        predictions = predict_proba[:,1] > cutoff
        recalls,precisions,fscores,support = score(predictions, Y_te)
        precisions_0.append(precisions[0])
        precisions_1.append(precisions[1])
        recalls_0.append(recalls[0])
        recalls_1.append(recalls[1])
        fscores_0.append(fscores[0])
        fscores_1.append(fscores[1])
        accs.append((predictions == Y_te).mean())
    if recall:
        plt.plot(cutoffs,np.array(recalls_0),label='Recall Category 0')
        plt.plot(cutoffs,np.array(recalls_1),label='Recall Category 1')
    elif precision:
        plt.plot(cutoffs,np.array(precisions_0),label='Precision Category 0')
        plt.plot(cutoffs,np.array(precisions_1),label='Precision Category 1')    
    elif fscore:
        plt.plot(cutoffs,np.array(fscores_0),label='F-Score Category 0')
        plt.plot(cutoffs,np.array(fscores_1),label='F-Score Category 1') 
    elif acc:
        plt.plot(cutoffs,np.array(recalls_1),label='Recall Category 1')
        plt.plot(cutoffs,np.array(accs),label='Model Accuracy')
    else:
        plt.plot(cutoffs,np.array(recalls_1),label='Recall')
        plt.plot(cutoffs,np.array(precisions_1),label='Precision')
        plt.plot(cutoffs,np.array(fscores_1),label='F-Score')
    plt.xlabel('Cutoff')
    plt.ylabel('Score')
    plt.title('Model Statistics by Probability Cutoff')
    plt.legend()
    plt.show()
    plt.close()

    return


# Prints a classification report and plots a confusion matrix
def plot_confusion_matrix(Y_test, predictions, classes):
    '''
        inputs:
            Y_test: test labels 
            predictions: predictions made by your model
            classes: [label of negative output, label of positive output]
    '''
    matrix = confusion_matrix(Y_test, predictions) 
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.OrRd)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = ",d"
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('\nClasification report:\n', classification_report(Y_test, predictions))

    return plt

# plots a histogram for categorical variables
def plot_categorical_hist(data, col, normalize = False, title = False):
    '''
        inputs:
            data: the pandas dataframe containing the data
            col: name of the column we want to plot
            normalize: whether we want frequencies or counts
            title: title of the plot
    '''
    if normalize:
        normalize = 1
        fmt = FormatStrFormatter('%.2f')
    else:
        fmt = plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    if title:
        title = title
    else:
        title = col
    bars = data[col].value_counts(normalize = normalize)
    bars = bars.plot.bar(title = title, color = ['gray'], alpha = 0.5)
    bars.set_xlabel('Category')
    bars.set_ylabel('Frequency')
    bars.get_yaxis().set_major_formatter(fmt)

    return plt

# Plots stacked bar charts representing observed category frequencies
# the positive output population compared to either the negative or total population
def stack_bars(data, col, output, output_label = False, total = False, title = False):
    '''
        inputs:
            data: the pandas dataframe containing the data
            col: name of the column we want to plot
            output: the name of the column containing the output data
            output_label: output label to be shown on plot
            total: boolean, if true compare positive output to total population, otherwise compare to negative output pop
            title: title of the plot
    '''
    if title:
        title = title
    else:
        title = col
    if output_label:
        output_label = output_label
    else:
        output_label = output
    if total:
        label_0 = 'Total Population'
        tot = 999
    else:
        label_0 = 'Non ' + output_label + ' Population'
        tot = 1
    label_1 = output_label + ' Population'    
    
    stacked_0 = pd.DataFrame(columns = data[data[output] != tot][col].value_counts(normalize=True).index)
    stacked_0.loc[0] = data[data[output] != tot][col].value_counts(normalize=True)
    
    stacked_1 = pd.DataFrame(columns = data[data[output] == 1][col].value_counts(normalize=True).index)
    stacked_1.loc[0] = data[data[output] == 1][col].value_counts(normalize=True)
    
    stacked_bars = pd.concat([stacked_0,stacked_1],join = 'outer')
    stacked_bars.index = [label_0, label_1]
    
    stacked_bars = stacked_bars.plot.bar(title = title, stacked=True, alpha = 0.55)
    stacked_bars.set_ylabel('Frequency')
    
    plt.ylim([0,1.1])
    plt.xticks(rotation = 0)

    return plt



# Plots bar graph to compare relative frequencies of the positive output population compared
# to either the negative or total population
def plot_bar(data, col, output, output_label = False,total = False, normalize = True, title = False):
    '''
        inputs:
            data: the pandas dataframe containing the data
            col: name of the column we want to plot
            output: the name of the column containing the output data
            output_label: output label to be shown on plot
            total: boolean, if true compare positive output to total population, otherwise compare to negative output pop
            normalize: boolean specifies whether we want normalized frequency distributions or distribution of counts
            title: title of the plot
    '''
    if output_label:
        output_label = output_label
    else:
        output_label = output
    label_1 = output_label + ' Population'   
    if total:
        other_pop = data[col].value_counts(normalize = normalize)
        label_other = 'Total Population'
        output_label = 'Total '
    else:
        other_pop = data[data[output] == 0][col].value_counts(normalize = normalize)
        label_other = 'Non ' + label_1
    positive_pop = data[data[output] == 1][col].value_counts(normalize = normalize)
    
    bars = pd.concat([positive_pop,other_pop],axis=1,join='outer')
    bars.columns = [label_1,label_other]
    bars = bars.plot.bar(title = col, color = ['red','gray'], alpha = 0.55)
    bars.set_xlabel('Category')
    bars.set_ylabel('Frequency')

    return plt


# Plots continuous histogram for positive versus negative populations 
def plot_cts_hist(data,col,output, bins, output_label = False, zoom = False, normalized = True, title = False):
    '''
        inputs:
            data: the pandas dataframe containing the data
            col: name of the column we want to plot
            output: the name of the column containing the output data
            bins: number of bins in histogram
            output_label: output label to be shown on plot
            zoom: [lower bound of range, upper bound of range]
            normalized: boolean specifies whether we want normalized frequency distributions or distribution of counts
            title: title of the plot
    '''
    if title:
        title = title
    else:
        title = col
    if output_label:
        output_label = output_label
    else:
        output_label = output
    label_1 = output_label + ' Population'
    other_label = 'Non ' + output_label + ' Population'
    plt.figure()
    sns.distplot(data[data[output]==1][col], norm_hist=normalized,color = 'red',bins=bins, label = label_1)
    sns.distplot(data[data[output]==0][col], norm_hist=normalized,bins=bins, label = other_label)
    plt.legend([label_1,other_label])
    if zoom:
        zoom = zoom
        plt.xlim(zoom[0],zoom[1])
    plt.title(title)
    plt.ylabel('Frequency')

    return plt


# Takes continuous 3d data (post PCA and with output column included for coloring) 
# and plots in 3-dimensional space 
# Also plots a 2d projection of the same data
def plot3d(scatter, output, output_label = False):
    '''
        inputs:
            scatter: dataframe with labeled continuous data for 3D (and 2D projection) plotting
            output: the name of the column containing the output data
            output_label: output label to be shown on plot

    '''
    if output_label:
        output_label = output_label
    else:
        output_label = output
    label_1 = output_label + ' Population'   
    label_0 = 'Non ' + output_label + ' Population' 
    
    scatter_0 = scatter[scatter[output] == 0].drop('default',axis=1)
    scatter_1 = scatter[scatter[output] == 1].drop('default',axis=1)

    x, y, z = scatter_0[0], scatter_0[1], scatter_0[2]
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.5
        ),
        name = label_0
    )

    x2, y2, z2 = scatter_1[0], scatter_1[1], scatter_1[2]
    trace2 = go.Scatter3d(
        x=x2,
        y=y2,
        z=z2,
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.5
        ),
        name = label_1
    )
    plotted = [trace1, trace2]
    layout = go.Layout(title = '3-D Projection of Data',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=plotted, layout=layout)
    plt.scatter(scatter[0],scatter[1], c=scatter[output],alpha=0.5)
    plt.title('2-D Projection of Data')
    plt.show()
    plt.close
    return py.iplot(fig,filename='3d-projection')




def rfc_n_estimators(estimators, X_tr, Y_tr, X_te, Y_te, recall = False, precision = False, fscore = False, title = False):
    '''
        inputs:
            clf: the classifier function 
            estimators: the estimators you want to test (list or numpy array)
            X_tr, Y_tr, X_te, Y_te: numpy arrays which contain the training and testing data
            recall: bool, will plot recall for each category
            precision: bool, will plot precision for each category
            fscore: bool, will plot fscore for each category
            if all are false will just plot all 3 for output category 1
    '''
    clf = RandomForestClassifier()
    clf_placeholder = clf
    fscores_0 = []
    fscores_1 = []
    recalls_0 = []
    recalls_1 = []
    precisions_0 = []
    precisions_1 = []
    estimators = np.array(estimators)
    for estimator in estimators:
        clf.n_estimators = estimator
        clf = clf.fit(X_tr,Y_tr)
        predictions = clf.predict(X_te)
        recalls,precisions,fscores,support = score(predictions, Y_te)
        precisions_0.append(precisions[0])
        precisions_1.append(precisions[1])
        recalls_0.append(recalls[0])
        recalls_1.append(recalls[1])
        fscores_0.append(fscores[0])
        fscores_1.append(fscores[1])
        clf = clf_placeholder
    if recall:
        plt.plot(estimators,np.array(recalls_0),label='Recall Category 0')
        plt.plot(estimators,np.array(recalls_1),label='Recall Category 1')
    elif precision:
        plt.plot(estimators,np.array(precisions_0),label='Precision Category 0')
        plt.plot(estimators,np.array(precisions_1),label='Precision Category 1')    
    elif fscore:
        plt.plot(estimators,np.array(fscores_0),label='F-Score Category 0')
        plt.plot(estimators,np.array(fscores_1),label='F-Score Category 1')        
    else:
        plt.plot(estimators,np.array(recalls_1),label='Recall')
        plt.plot(estimators,np.array(precisions_1),label='Precision')
        plt.plot(estimators,np.array(fscores_1),label='F-Score')
    plt.xlabel('Estimators')
    plt.ylabel('Score')
    if title:
        plt.title(title)
    else:
        plt.title('Model Statistics by Number of Estimators')
    plt.legend()
    plt.show()
    plt.close()
    return


    # Plots the precision, recall, and fscore for given probability hyperparameters
def rfc_min_samples_split(hyperparameters, X_tr, Y_tr, X_te, Y_te, recall = False, precision = False, fscore = False, title = False):
    '''
        inputs:
            clf: the classifier function 
            hyperparameters: the hyperparameters you want to test (list or numpy array)
            X_tr, Y_tr, X_te, Y_te: numpy arrays which contain the training and testing data
            recall: bool, will plot recall for each category
            precision: bool, will plot precision for each category
            fscore: bool, will plot fscore for each category
            if all are false will just plot all 3 for output category 1
    '''
    clf = RandomForestClassifier()
    clf_placeholder = clf
    fscores_0 = []
    fscores_1 = []
    recalls_0 = []
    recalls_1 = []
    precisions_0 = []
    precisions_1 = []
    hyperparameters = np.array(hyperparameters)
    for param in hyperparameters:
        clf.min_samples_split = param
        clf = clf.fit(X_tr,Y_tr)
        predictions = clf.predict(X_te)
        recalls,precisions,fscores,support = score(predictions, Y_te)
        precisions_0.append(precisions[0])
        precisions_1.append(precisions[1])
        recalls_0.append(recalls[0])
        recalls_1.append(recalls[1])
        fscores_0.append(fscores[0])
        fscores_1.append(fscores[1])
        clf = clf_placeholder
    if recall:
        plt.plot(hyperparameters,np.array(recalls_0),label='Recall Category 0')
        plt.plot(hyperparameters,np.array(recalls_1),label='Recall Category 1')
    elif precision:
        plt.plot(hyperparameters,np.array(precisions_0),label='Precision Category 0')
        plt.plot(hyperparameters,np.array(precisions_1),label='Precision Category 1')    
    elif fscore:
        plt.plot(hyperparameters,np.array(fscores_0),label='F-Score Category 0')
        plt.plot(hyperparameters,np.array(fscores_1),label='F-Score Category 1')        
    else:
        plt.plot(hyperparameters,np.array(recalls_1),label='Recall')
        plt.plot(hyperparameters,np.array(precisions_1),label='Precision')
        plt.plot(hyperparameters,np.array(fscores_1),label='F-Score')
    plt.xlabel('Split')
    plt.ylabel('Score')
    if title:
        plt.title(title)
    else:
        plt.title('Model Statistics by Minimum Samples to Split Branch')
    plt.legend()
    plt.show()
    plt.close()
    return


    # Plots the precision, recall, and fscore for given probability hyperparameters
def rfc_min_samples_leaf(hyperparameters, X_tr, Y_tr, X_te, Y_te, recall = False, precision = False, fscore = False, title = False):
    '''
        inputs:
            clf: the classifier function 
            hyperparameters: the hyperparameters you want to test (list or numpy array)
            X_tr, Y_tr, X_te, Y_te: numpy arrays which contain the training and testing data
            recall: bool, will plot recall for each category
            precision: bool, will plot precision for each category
            fscore: bool, will plot fscore for each category
            if all are false will just plot all 3 for output category 1
    '''
    clf = RandomForestClassifier()
    clf_placeholder = clf
    fscores_0 = []
    fscores_1 = []
    recalls_0 = []
    recalls_1 = []
    precisions_0 = []
    precisions_1 = []
    hyperparameters = np.array(hyperparameters).astype(int)
    for param in hyperparameters:
        clf.min_samples_leaf = param
        clf = clf.fit(X_tr,Y_tr)
        predictions = clf.predict(X_te)
        recalls,precisions,fscores,support = score(predictions, Y_te)
        precisions_0.append(precisions[0])
        precisions_1.append(precisions[1])
        recalls_0.append(recalls[0])
        recalls_1.append(recalls[1])
        fscores_0.append(fscores[0])
        fscores_1.append(fscores[1])
        clf = clf_placeholder
    if recall:
        plt.plot(hyperparameters,np.array(recalls_0),label='Recall Category 0')
        plt.plot(hyperparameters,np.array(recalls_1),label='Recall Category 1')
    elif precision:
        plt.plot(hyperparameters,np.array(precisions_0),label='Precision Category 0')
        plt.plot(hyperparameters,np.array(precisions_1),label='Precision Category 1')    
    elif fscore:
        plt.plot(hyperparameters,np.array(fscores_0),label='F-Score Category 0')
        plt.plot(hyperparameters,np.array(fscores_1),label='F-Score Category 1')        
    else:
        plt.plot(hyperparameters,np.array(recalls_1),label='Recall')
        plt.plot(hyperparameters,np.array(precisions_1),label='Precision')
        plt.plot(hyperparameters,np.array(fscores_1),label='F-Score')
    plt.xlabel('Samples')
    plt.ylabel('Score')
    if title:
        plt.title(title)
    else:
        plt.title('Model statistics by Minimum Samples per Leaf')
    plt.legend()
    plt.show()
    plt.close()
    return