import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0,'..')


# importing the required module
import seaborn
import argparse
import plotly.express as px

def main(args):
     
    # loading data-set
    data = pd.read_csv(args.csv)
    fig = px.violin(data,y = 'abs-error',box=True, points = 'all')
    fig.update_layout(title_text="GA at birth prediction absolute error: <br>Template Space.",
                    )
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose a csv file')
    parser.add_argument('--csv',type=str, help='csv file', required=True)
    args = parser.parse_args()
    main(args)