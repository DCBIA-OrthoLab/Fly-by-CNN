import os
import sys
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import csv
from pandas import read_csv
import itk
import time
import math
import webbrowser

#
# prediction
#

class prediction(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "prediction" 
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Mathieu Leclercq"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#prediction">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""


#
# predictionWidget
#

class predictionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.fileName = ""
    self.surfaceFile = ""
    self.outputFile  = ""
    self.lArrays = []
    self.model = "" 
    self.resolution = 256
    self.predictedId = ""
    self.rotation = None




  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/prediction.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)


    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = predictionLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # UI elements
      
    # Inputs
    self.ui.applyChangesButton.connect('clicked(bool)',self.onApplyChangesButton)
    self.ui.rotationSpinBox.valueChanged.connect(self.onRotationSpinbox)
    self.ui.rotationSlider.valueChanged.connect(self.onRotationSlider)
    self.ui.browseSurfaceButton.connect('clicked(bool)',self.onBrowseSurfaceButton)
    self.ui.browseModelButton.connect('clicked(bool)',self.onBrowseModelButton)
    self.ui.surfaceLineEdit.textChanged.connect(self.onEditSurfaceLine)
    self.ui.modelLineEdit.textChanged.connect(self.onEditModelLine)

    


    # Advanced 
    self.ui.predictedIdLineEdit.textChanged.connect(self.onEditPredictedIdLine)
    self.ui.resolutionComboBox.currentTextChanged.connect(self.onResolutionChanged)

    # Outputs 
    self.ui.browseOutputButton.connect('clicked(bool)',self.onBrowseOutputButton)
    self.ui.outputLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.outputFileLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.openOutButton.connect('clicked(bool)',self.onOutButton)

    self.ui.resetButton.connect('clicked(bool)',self.onReset)
    self.ui.cancelButton.connect('clicked(bool)', self.onCancel)

    self.ui.progressLabel.setHidden(True)
    self.ui.openOutButton.setHidden(True)
    self.ui.cancelButton.setHidden(True)
    #self.ui.propertyComboBox.setHidden(True)

    #initialize variables
    self.model = self.ui.modelLineEdit.text
    self.surfaceFile = self.ui.surfaceLineEdit.text
    self.outputFile = self.ui.outputLineEdit.text + self.ui.outputFileLineEdit.text
    self.predictedId = self.ui.predictedIdLineEdit.text
    self.resolution = self.ui.resolutionComboBox.currentText
    self.rotation = self.ui.rotationSlider.value


    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    # if inputParameterNode:
    #   self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True


    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch


    self._parameterNode.EndModify(wasModified)


  def onResolutionChanged(self):
    self.resolution = int(self.ui.resolutionComboBox.currentText)

  def updateProgressBar(self,caller,event):
    """
    self.logic.progress = self.logic.GetProgress()
    if self.logic.progress == 100:
      self.onProcessDone()
    self.ui.progressBar.setValue(self.logic.progress) 
    """
    pass

  def onProcessDone(self):
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)

    self.ui.progressLabel.setHidden(False)
    self.ui.openOutButton.setHidden(False)    
    self.ui.cancelButton.setEnabled(False)
    self.ui.progressBar.setEnabled(False)


  def onOutButton(self):
    webbrowser.open(self.outputFile)




  def onPropertyChanged(self):
    self.property = self.ui.propertyComboBox.currentText
    print(self.property)

  def onApplyChangesButton(self):

    #if os.path.isdir(self.surfaceFile) and os.path.isdir(self.outputFile) and self.property != '':
    if self.model != '':
      self.ui.applyChangesButton.setEnabled(False)
      self.ui.progressBar.setEnabled(True)
      self.logic = predictionLogic(self.surfaceFile,self.outputFile,self.resolution, self.ui.rotationSpinBox.value,self.model, self.predictedId)
      self.logic.process()
      self.logic.cliNode.AddObserver('ModifiedEvent', self.updateProgressBar)
      self.ui.cancelButton.setHidden(False)
      self.ui.cancelButton.setEnabled(True)
      self.ui.resetButton.setEnabled(False)
      self.ui.progressBar.setRange(0,0)
      self.ui.progressBar.setEnabled(True)
      self.ui.progressBar.setTextVisible(True)
      self.ui.progressLabel.setHidden(False)

    else:
      print('error')
      msg = qt.QMessageBox()
      if not(os.path.isdir(self.surfaceFile)):        
        msg.setText("Surface directory : \nIncorrect path.")
        print('Error: Incorrect path for surface directory.')
        self.ui.surfaceLineEdit.setText('')
        print(f'surface folder : {self.surfaceFile}')
     
      elif not(os.path.isdir(self.outputFile)):
        msg.setText("Output directory : \nIncorrect path.")
        print('Error: Incorrect path for output directory.')
        self.ui.outputLineEdit.setText('')
        print(f'output folder : {self.surfaceFile}')


      msg.setWindowTitle("Error")
      msg.exec_()

      return

  def onReset(self):
    self.ui.outputLineEdit.setText("")
    self.ui.surfaceLineEdit.setText("")
    self.ui.rotationSpinBox.value = 50
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.progressLabel.setHidden(True)
    self.ui.openOutButton.setHidden(True)
    self.ui.progressBar.setValue(0)

  def onCancel(self):
    self.logic.cliNode.Cancel()
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.ui.progressBar.setEnabled(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressLabel.setHidden(True)
    self.ui.cancelButton.setEnabled(False)


    
    print("Process successfully cancelled.")


  def onBrowseSurfaceButton(self):
    newsurfaceFile = qt.QFileDialog.getOpenFileName(self.parent, "Select a surface")
    if newsurfaceFile != '':
      self.surfaceFile = newsurfaceFile
      self.ui.surfaceLineEdit.setText(self.surfaceFile)
    #print(f'Surface directory : {self.surfaceFile}')



  def onBrowseModelButton(self):
    newModel = qt.QFileDialog.getOpenFileName(self.parent, "Select a model")
    if newModel != '':
      self.model = newModel
      self.ui.modelLineEdit.setText(self.model)
    #print(f'Surface directory : {self.surfaceFile}')



  def onBrowseOutputButton(self):
    newoutputFile = qt.QFileDialog.getExistingDirectory(self.parent, "Select a directory")
    if newoutputFile != '':
      self.outputFile = newoutputFile
      self.ui.outputLineEdit.setText(self.outputFile)
    #print(f'Output directory : {self.outputFile}')      




  def onEditModelLine(self):
    self.model = self.ui.modelLineEdit.text



  def onEditPredictedIdLine(self):
    self.predictedId = self.ui.predictedIdLineEdit.text


  def onEditSurfaceLine(self):
    self.surfaceFile = self.ui.surfaceLineEdit.text
    


  def onEditOutputLine(self):
    self.outputFile = self.ui.outputLineEdit.text + self.ui.outputFileLineEdit.text





  def onRotationSlider(self):
    self.ui.rotationSpinBox.value = self.ui.rotationSlider.value
    self.rotation = self.ui.rotationSlider.value


  def onRotationSpinbox(self):
    self.ui.rotationSlider.value = self.ui.rotationSpinBox.value
    self.rotation = self.ui.rotationSlider.value

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    return

  def ReadCsv(self,fileName):
    fileList = []
    try:
      with open(fileName, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
          fileList.append(row)
      """
      for item in fileList:
        print (item)      
      """
      print (".csv file opened successfully")
      return fileList
    except IOError:  # incorrect path name
      msg = qt.QMessageBox()
      if self.fileName != '':
        msg.setText("Incorrect path.")
        print('Error: Incorrect path.')
      else:
        msg.setText("Please select a .csv file.")
        print("Error: Please select a .csv file.")
      msg.setWindowTitle("Error")
      msg.exec_()
      self.ui.lineEdit.setText('')
      print(f'fileName : {self.fileName}')
      
      return -1


  def ReadCsvPandas(self,fileName):

    try:
      file = read_csv(fileName)
      #print(file.to_string())
      for column in file:
        print(column)

    except:
      msg = qt.QMessageBox()
      if self.fileName != '':
        msg.setText("Incorrect path.")
        print('Error: Incorrect path.')
      else:
        msg.setText("Please select a .csv file.")
        print("Error: Please select a .csv file.")
      msg.setWindowTitle("Error")
      msg.exec_()
      self.ui.lineEdit.setText('')
      print(f'fileName : {self.fileName}')

#
# predictionLogic
#

class predictionLogic(ScriptedLoadableModuleLogic):
  """
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, surfaceFile= None,outputFile=None, resolution=None, rotation=None,model=None,predictedId=None):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.surfaceFile = surfaceFile
    self.outputFile = outputFile
    self.resolution = resolution
    self.rotation = rotation
    self.model = model
    self.predictedId = predictedId
    self.nbOperation = 0
    self.progress = 0
    self.cliNode = None
    print(f"model: {self.model}")
    print(f'surfaceFile : {self.surfaceFile}')
    print(f'outptutfile : {self.outputFile}')
    print(f'resolution : {self.resolution}')
    print(f'rotation : {self.rotation}')
    print(f'predictedId : {self.predictedId}')


  # def setDefaultParameters(self, parameterNode):
  #   """
  #   Initialize parameter node with default settings.
  #   """

  def process(self):
    print('process')
    parameters = {}
    parameters ["surfaceFile"] = self.surfaceFile
    parameters ["outputFile"] = self.outputFile
    parameters ["rotation"] = self.rotation
    parameters ['resolution'] = self.resolution
    parameters ['model'] = self.model
    parameters ['predictedId'] = self.predictedId
    #self.GetNbOperation()
    #print('nb operation : ', self.nbOperation)
    print ('parameters : ', parameters)
    flybyProcess = slicer.modules.predictioncli
    self.cliNode = slicer.cli.run(flybyProcess,None, parameters)    
    return flybyProcess


  def GetProgress(self):
    """
    filesList = os.listdir(self.outputFile)
    self.progress =  math.floor(100 * ((len(filesList)-self.initialNbFiles)/self.nbOperation))
    return self.progress
    """ 
    return None




#
# predictionTest
#

class predictionTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_prediction1()

  def test_prediction1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('prediction1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    # logic = predictionLogic()



    self.delayDisplay('Test passed')

def ReadSurf(fileName):   # Copied from utils.py (https://github.com/DCBIA-OrthoLab/fly-by-cnn/tree/master/src/py)

    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    if extension == ".vtk":
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()    
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".off":
        reader = OFFReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(fileName)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                obj_import.SetTexturePath(textures_path)
            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
            
            append.Update()
            surf = append.GetOutput()
            
        else:
            print('unknown extension')
            reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            surf = reader.GetOutput()

    return surf