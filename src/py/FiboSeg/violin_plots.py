import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0,'..')


# importing the required module
import seaborn
import argparse

parser = argparse.ArgumentParser(description='Choose a csv file')
parser.add_argument('--csv',type=str, help='csv file', required=True)
args = parser.parse_args()

# use to set style of background of plot
seaborn.set(style="whitegrid")
 
# loading data-set
data = pd.read_csv(args.csv)


 
seaborn.violinplot(x='crown', y='avr_dist',
                   data=data,cut=0, scale='count')

plt.title('Average distance to groundtruth for every crown')
plt.ylabel('Average distance (mm)')
ax = plt.gca()

ax.set_ylim([0, 4])
plt.show()