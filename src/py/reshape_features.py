import sys
import itk
import numpy as np
from utils import * 


img = ReadImage(sys.argv[1])
img_np = itk.GetArrayViewFromImage(img)

img_np = np.transpose(np.reshape(img_np, [s for s in img_np.shape if s != 1]))
img_np = img_np.reshape(list(img_np.shape) + [1])

out_img = GetImage(img_np)

writer = itk.ImageFileWriter.New(FileName=sys.argv[2], Input=out_img)
writer.UseCompressionOn()
writer.Update()

