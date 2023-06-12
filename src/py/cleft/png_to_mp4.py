import argparse

import math
import os
import numpy as np 
import cv2

from tqdm import tqdm
import glob

def main(args):

    fourcc = None
    out = None

    for filename in tqdm(glob.glob(args.dir + args.ext)):

        img = cv2.imread(filename)
        height, width, layers = img.shape

        if fourcc is None:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(args.out, fourcc, args.fps, (height, width))
        
        out.write(img)

    out.release()



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Create video from series of files')    
    parser.add_argument('--dir', help='Input directory', type=str, required=True)   
    parser.add_argument('--out', help='Output filename', type=str, required=True)
    parser.add_argument('--ext', help='Files extension', type=str, default="*.png")
    parser.add_argument('--fps', help='Frames per second', type=int, default=24)

    args = parser.parse_args()

    main(args)

