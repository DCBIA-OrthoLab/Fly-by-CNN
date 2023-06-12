import os
import glob
import argparse
import torch
from tqdm import tqdm
import numpy as np
import random
import math
import utils

def main(args):

  surf = utils.ReadSurf(args.surf)

  surf = utils.GetUnitSurf(surf)

  utils.WriteSurf(surf, args.out)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Choose a .vtk file.')
  parser.add_argument('--surf',type=str, help='Input. Either .vtk file or folder containing vtk files.', required=True)
  parser.add_argument('--out',type=str,help ='Name of output file is input is a single file, or name of output folder if input is a folder.',required=True)
  args = parser.parse_args()
  main(args)
