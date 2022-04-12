import pandas as pd
import os 

df = pd.read_csv("/work/jprieto/data/fly_by_fibers/tracts_filtered_test.csv")

for idx, row in df.iterrows():
     out = os.path.join("fiber_test", str(idx) + "_fiber_test.pickle")
     out_rename = os.path.join("fiber_test", str(row["id"]) + "_" + str(row["class"]) + "_fiber_test.pickle")
     if os.path.exists(out):
             print("mv", out, out_rename)