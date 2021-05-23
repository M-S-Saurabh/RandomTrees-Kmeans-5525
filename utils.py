import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

DATA_FOLDER = './data/'

def read_data(dataset, test_size=0.3, include_ids=False):
    index_col = None if include_ids else 0
    df = pd.read_csv(DATA_FOLDER+dataset, delimiter=',', index_col=index_col, header=None)

    df_X = df.iloc[:, :-1]
    # Replacing missing values with mode
    for col in df_X.columns:
        mode = df_X[[col]].mode().values[0,0]
        df_X[[col]] = df_X[[col]].replace('?', mode).astype(int)
    df_X = df_X.to_numpy()

    df_y = df.iloc[:, -1].to_numpy()
    # Convert labels to +/-1
    df_y[df_y == 2] = -1
    df_y[df_y == 4] = 1

    return train_test_split(df_X, df_y, test_size=test_size, random_state=0)

def entropy(y):
    if len(y) == 0: return 0
    p_0 = np.count_nonzero(y == -1) / len(y)
    p_1 = 1 - p_0
    eps = 1e-10
    return (-p_0 * np.log2(p_0 +eps) - p_1 * np.log2(p_1 +eps))

def plot_losses(losses, filename):
    plt.figure()
    plt.plot(losses, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.savefig(filename)

def plot_results(error_rates, name='adaboost', xlabel='num of weak learners'):
    plt.figure()
    plt.plot([e[0] for e in error_rates],
             [e[1] for e in error_rates], 'b-', label='train error')
    plt.plot([e[0] for e in error_rates],
             [e[2] for e in error_rates], 'r-', label='test error')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Error rate')
    plt.savefig('ErrorRate_vs_N-{}.png'.format(name))


def display_results(error_rates, name='adaboost', xlabel='num of weak learners'):
    plot_results(error_rates, name, xlabel=xlabel)
    print('--------------------'+name+'----------------------')
    print('Error rates vs.'+xlabel+':')
    step = 1 if len(error_rates) < 21 else 5 if len(error_rates) < 51 else 10
    row_format ="{:>15}" * (3)
    print(row_format.format(xlabel, "train_error", "test_error"))
    for i in range(0, len(error_rates), step):
        print( row_format.format( *np.round(error_rates[i],3) ) )
