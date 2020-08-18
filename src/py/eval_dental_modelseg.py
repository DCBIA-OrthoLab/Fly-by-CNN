
from __future__ import print_function
import numpy as np
import vtk
import argparse
import os
from datetime import datetime, time
import json
import glob
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp
import csv

parser = argparse.ArgumentParser(description='Evaluate dental model seg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--csv', type=str, help='CSV columns gt and prediction. Both VTK files with label array "RegionId"', required=True)
parser.add_argument('--out', type=str, help='Out filename for plots', default="./out")

args = parser.parse_args()

y_pred_arr = []
y_true_arr = []

fpr_arr = []
tpr_arr = []
roc_auc_arr = []
iou_arr = []

abs_diff_arr = []
mse_arr = []

eval_type = "class"
class_names = ["Gum", "Teeth", "Boundary"]

if(eval_type == "class"):

  with open(args.csv) as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:

      args = parser.parse_args()

      reader = vtk.vtkPolyDataReader()
      reader.SetFileName(row["gt"])
      reader.Update()

      clean = vtk.vtkCleanPolyData()
      clean.SetInputData(reader.GetOutput())
      clean.SetTolerance(0.0001)
      clean.Update()
      surf1 = clean.GetOutput()

      surf1_label = surf1.GetPointData().GetArray('RegionId')
      for pid in range(surf1_label.GetNumberOfTuples()):
        y_true_arr.append(surf1_label.GetTuple(pid)[0])

      reader = vtk.vtkPolyDataReader()
      reader.SetFileName(row["prediction"])
      reader.Update()

      clean = vtk.vtkCleanPolyData()
      clean.SetInputData(reader.GetOutput())
      clean.SetTolerance(0.0001)
      clean.Update()
      surf2 = clean.GetOutput()

      surf2_label = surf2.GetPointData().GetArray('RegionId')
      for pid in range(surf2_label.GetNumberOfTuples()):
        if surf2_label.GetTuple(pid)[0] == -1:
          y_pred_arr.append(0)
        else:
          y_pred_arr.append(surf2_label.GetTuple(pid)[0] - 1)

elif(eval_type == "segmentation"):
  fpr, tpr, _ = roc_curve(np.array(image_batch[1]).reshape(-1), np.array(y_pred).reshape(-1), pos_label=1)
  roc_auc = auc(fpr,tpr)

  fpr_arr.append(fpr)
  tpr_arr.append(tpr)
  roc_auc_arr.append(roc_auc)

  y_pred_flat = np.array(y_pred).reshape((len(y_pred), -1))
  labels_flat = np.array(image_batch[1]).reshape((len(y_pred), -1))

  for i in range(len(y_pred)):
    intersection = 2.0 * np.sum(y_pred_flat[i] * labels_flat[i]) + 1e-7
    union = np.sum(y_pred_flat[i]) + np.sum(labels_flat[i]) + 1e-7
    iou_arr.append(intersection/union)

elif(eval_type == "image" or eval_type == "numeric"):
  abs_diff_arr.extend(np.average(np.absolute(y_pred - image_batch[1]).reshape([1, -1]), axis=-1))
  mse_arr.extend(np.average(np.square(y_pred - image_batch[1]).reshape([1, -1]), axis=-1))


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

if(eval_type == "class"):
  # Compute confusion matrix

  cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
  np.set_printoptions(precision=3)

  # Plot non-normalized confusion matrix
  fig = plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
  confusion_filename = os.path.splitext(args.out)[0] + "_confusion.png"
  fig.savefig(confusion_filename)
  # Plot normalized confusion matrix
  fig2 = plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

  norm_confusion_filename = os.path.splitext(args.out)[0] + "_norm_confusion.png"
  fig2.savefig(norm_confusion_filename)

elif(eval_type == "segmentation"):

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_arr]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(len(fpr_arr)):
      mean_tpr += interp(all_fpr, fpr_arr[i], tpr_arr[i])

  mean_tpr /= len(fpr_arr)

  roc_auc = auc(all_fpr, mean_tpr)

  roc_fig = plt.figure()
  lw = 1
  plt.plot(all_fpr, mean_tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")

  roc_filename = os.path.splitext(json_tf_records)[0] + "_roc.png"
  roc_fig.savefig(roc_filename)

  iou_obj = {}
  iou_obj["iou"] = iou_arr

  iou_json = os.path.splitext(json_tf_records)[0] + "_iou_arr.json"

  with open(iou_json, "w") as f:
    f.write(json.dumps(iou_obj))

  iou_fig_polar = plt.figure()
  ax = iou_fig_polar.add_subplot(111, projection='polar')
  theta = 2 * np.pi * np.arange(len(iou_arr))/len(iou_arr)
  colors = iou_arr
  ax.scatter(theta, iou_arr, c=colors, cmap='autumn', alpha=0.75)
  ax.set_rlim(0,1)
  plt.title('Intersection over union')
  locs, labels = plt.xticks()
  plt.xticks(locs, np.arange(0, len(iou_arr), round(len(iou_arr)/len(locs))))

  iou_polar_filename = os.path.splitext(json_tf_records)[0] + "_iou_polar.png"
  iou_fig_polar.savefig(iou_polar_filename)

  iou_fig = plt.figure()
  x_samples = np.arange(len(iou_arr))
  plt.scatter(x_samples, iou_arr, c=colors, cmap='autumn', alpha=0.75)
  plt.title('Intersection over union')
  iou_mean = np.mean(iou_arr)
  plt.plot(x_samples,[iou_mean]*len(iou_arr), label='Mean', linestyle='--')
  plt.text(len(iou_arr) + 2,iou_mean, '%.3f'%iou_mean)
  iou_stdev = np.std(iou_arr)
  stdev_line = plt.plot(x_samples,iou_mean + [iou_stdev]*len(iou_arr), label='Stdev', linestyle=':', alpha=0.75)
  stdev_line = plt.plot(x_samples,iou_mean - [iou_stdev]*len(iou_arr), label='Stdev', linestyle=':', alpha=0.75)
  plt.text(len(iou_arr) + 2,iou_mean + iou_stdev, '%.3f'%iou_stdev, alpha=0.75, fontsize='x-small')
  iou_filename = os.path.splitext(json_tf_records)[0] + "_iou.png"
  iou_fig.savefig(iou_filename)

elif(eval_type == "image" or eval_type == "numeric"):
  abs_diff_arr = np.array(abs_diff_arr)
  abs_diff_fig = plt.figure()
  x_samples = np.arange(len(abs_diff_arr))

  plt.scatter(x_samples, abs_diff_arr, c=abs_diff_arr, cmap='cool', alpha=0.75, label='Mean absolute error')
  plt.xlabel('Samples')
  plt.ylabel('Absolute error')
  plt.title('Mean absolute error')
  
  abs_diff_mean = np.array([np.mean(abs_diff_arr)]*len(abs_diff_arr))
  mean_line = plt.plot(x_samples,abs_diff_mean, label='Mean', linestyle='--')
  abs_diff_stdev = np.array([np.std(abs_diff_mean)]*len(abs_diff_mean))
  stdev_line = plt.plot(x_samples, abs_diff_mean + abs_diff_stdev, label='Mean', linestyle=':', alpha=0.75)
  plt.text(len(abs_diff_mean), np.mean(abs_diff_mean), "{0:.3f}".format(np.mean(abs_diff_mean)))

  abs_filename = os.path.splitext(json_tf_records)[0] + "_abs_diff.png"
  abs_diff_fig.savefig(abs_filename)

  mse_arr = np.array(mse_arr)
  mse_fig = plt.figure()
  plt.scatter(x_samples, mse_arr, c=mse_arr, cmap='cool', alpha=0.75, label='MSE')
  plt.xlabel('Samples')
  plt.ylabel('MSE')
  plt.title('Mean squared error')

  mse_mean = np.array([np.mean(mse_arr)]*len(mse_arr))
  mse_line = plt.plot(x_samples,mse_mean, label='Mean', linestyle='--')
  mse_stdev = np.array([np.std(mse_arr)]*len(mse_arr))
  stdev_line = plt.plot(x_samples, mse_mean + mse_stdev, label='Mean', linestyle=':', alpha=0.75)
  plt.text(len(mse_mean), np.mean(mse_mean), "{0:.3f}".format(np.mean(mse_mean)))

  mse_filename = os.path.splitext(json_tf_records)[0] + "_mse.png"
  mse_fig.savefig(mse_filename)

  print("mae:", abs_diff_mean[0], "mse:", mse_mean[0])