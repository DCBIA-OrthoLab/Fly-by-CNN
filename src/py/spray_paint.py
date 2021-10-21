import os
import re
import numpy as np
import itk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import argparse
import glob

import LinearSubdivisionFilter as lsf
from utils import * 
from post_process import *


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None, surf_property="", neighbors=1):
        # self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        # self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("CharEvent", self.char_event)

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005);
        self.surf_property = surf_property
        self.value = 0
        self.neighbors = neighbors
        self.left_button_down = False

    # def middle_button_press_event(self, obj, event):
    #     print("Middle Button pressed")
    #     self.OnMiddleButtonDown()
    #     return

    # def middle_button_release_event(self, obj, event):
    #     print("Middle Button released")
    #     self.OnMiddleButtonUp()
    #     return

    def left_button_press_event(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer());
        cellId = self.picker.GetCellId()
        if cellId != -1:
            self.left_button_down = True
            self.paint_surface(cellId)
        else:
            self.OnLeftButtonDown()
        return

    def left_button_release_event(self, obj, event):
        self.left_button_down = False
        self.OnLeftButtonUp()
        return

    def mouse_move_event(self, obj, event):

        clickPos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer());
        cellId = self.picker.GetCellId()
        if cellId != -1 and self.left_button_down:
            self.paint_surface(cellId)
        else:
            self.OnMouseMove()

        return

    def char_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        print(key)
        if key.isnumeric():
            self.value = int(key)
            print("Label is now:", self.value)
        elif key == 'KP_Add':
            self.value += 1
            print("Label is now:", self.value)
        elif key == 'KP_Subtract':
            self.value -= 1
            print("Label is now:", self.value)
        else:
            self.OnChar()

    def paint_surface(self, cellId):
        actor = self.picker.GetActor()
        surf = actor.GetMapper().GetInput()

        pointIds = vtk.vtkIdList()
        surf.GetCellPoints(cellId, pointIds)

        pids = []
        for i in range(pointIds.GetNumberOfIds()):
            pids.append(pointIds.GetId(i))
        
        all_pids = pids
        for l in range(self.neighbors):
            pids = GetAllNeighbors(surf, pids)
            all_pids = np.concatenate((all_pids, pids))

        all_pids = np.unique(all_pids)

        for pid in all_pids:
            surf.GetPointData().GetScalars().SetTuple1(pid, self.value)

        surf.GetPointData().GetScalars().Modified()
        self.GetDefaultRenderer().GetRenderWindow().GetInteractor().Render()


def main(args):
    surf = ReadSurf(args.surf)
    actor = GetColoredActor(surf, args.property)
    actor.GetProperty().SetInterpolationToFlat()

    colors = vtk.vtkNamedColors()

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('white'))
    renderer.AddActor(actor)

    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    renwin.SetSize(1080, 960)

    interactorstyle = MyInteractorStyle(surf_property=args.property, neighbors=args.neighbors)
    interactorstyle.SetDefaultRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(interactorstyle)
    interactor.SetRenderWindow(renwin)

    interactor.Initialize()
    renwin.Render()
    interactor.Start()

    WriteSurf(surf, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group('Input parameters')
    input_group.add_argument('--surf', type=str, help='Target surface/mesh', required=True)
    input_group.add_argument('--property', type=str, help='Input property file with same number of points as "surf"', default=None)
    input_group.add_argument('--neighbors', type=int, help='Neighborhood size', default=2)

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--out', type=str, help='Output surface/mesh', default='out.vtk')


    args = parser.parse_args()
    main(args)