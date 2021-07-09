import numpy as np
import argparse
import os
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from utils import * 

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp
import seaborn as sns

parser = argparse.ArgumentParser(description='Generate confusion matrix and classification report with dice for segmentation of gum, teeth, boundary', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--csv', type=str, help='csv file with columns surf,pred', required=True)
parser.add_argument('--point_features', type=str, help='Name of array in point data for the labels', default="RegionId")

args = parser.parse_args()

y_true_arr = [] 
y_pred_arr = []
dice_arr = []

df = pd.read_csv(args.csv)

for idx, row in df.iterrows():

  print("Reading:", row["surf"])
  surf = ReadSurf(row["surf"])
  print("Reading:", row["pred"])
  pred = ReadSurf(row["pred"])

  surf_features_np = vtk_to_numpy(surf.GetPointData().GetScalars(args.point_features))
  pred_features_np = vtk_to_numpy(pred.GetPointData().GetScalars(args.point_features))

  pred_features_np[pred_features_np==-1] = 1

  surf_features_np = np.reshape(surf_features_np, -1)
  pred_features_np = np.reshape(pred_features_np, -1)

  jaccard = jaccard_score(surf_features_np, pred_features_np, average=None)
  dice = 2.0*jaccard/(1.0 + jaccard)

  dice_arr.append(dice)
  y_true_arr.extend(surf_features_np)
  y_pred_arr.extend(pred_features_np)

dice_arr = np.array(dice_arr)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.3f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

  return cm


cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Gum", "Teeth", "Boundary"], title="Confusion Matrix Segmentation")
confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
fig.savefig(confusion_filename)

cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
print(cnf_matrix)
FP = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

F1 = 2 * (PPV * TPR)/(PPV + TPR)

print("True positive rate, sensitivity or recall:", TPR)
print("True negative rate or specificity:", TNR)
print("Positive predictive value or precision:", PPV)
print("Negative predictive value:", NPV)
print("False positive rate or fall out", FPR)
print("False negative rate:", FNR)
print("False discovery rate:", FDR)
print("Overall accuracy:", ACC)
print("F1 score:", F1)

print(classification_report(y_true_arr, y_pred_arr))

jaccard = jaccard_score(y_true_arr, y_pred_arr, average=None)
print("jaccard score:", jaccard)
print("dice:", 2.0*jaccard/(1.0 + jaccard))

# Plot normalized confusion matrix
fig2 = plt.figure()
cm = plot_confusion_matrix(cnf_matrix, classes=["Gum", "Teeth", "Boundary"], normalize=True, title="Confusion Matrix Segmentation - normalized")
norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
fig2.savefig(norm_confusion_filename)


fig3 = plt.figure() 
# Creating plot
s = sns.violinplot(data=dice_arr)
plt.xticks([0, 1, 2], ["Gum", "Teeth", "Boundary"])
s.set_title('Dice coefficients')
box_plot_filename = os.path.splitext(args.csv)[0] + "_violin_plot.png"
fig3.savefig(box_plot_filename)