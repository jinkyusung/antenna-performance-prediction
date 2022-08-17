import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_plot(df, mean=True, median=False, horizons=[], \
                color='gray', mean_color='blueviolet', median_color='b', hcolor='r'):
    n = len(df.columns)
    fig = plt.figure(figsize=(30, 5 * n))
    
    for i in range(1, n + 1):
        ax = fig.add_subplot(n, 1, i)
        ax.set_title(f'{df.columns[i-1]}', fontdict={'fontsize': 16,'fontweight':'bold'})
        ax.scatter(df.index.to_list(), df.iloc[:, i-1], s=0.5, c=color)
        
        if horizons:
            tmp = df[df.iloc[:, i-1] == horizons[i-1]]
            ax.scatter(tmp.index.to_list(), tmp.iloc[:, i-1], s=0.5, c=hcolor)
        if mean:
            ax.hlines(df.iloc[:, i-1].mean(), -700, len(df)+700, color=mean_color, linestyle='dashed', label='Mean')
            ax.legend(loc='best', shadow=True, fontsize='small')
        if median:
            ax.hlines(df.iloc[:, i-1].median(), -700, len(df)+700, color=median_color, linestyle='dashed', label='Median')
            ax.legend(loc='best', shadow=True, fontsize='small')
    plt.show()


def show_pred(train, pred):
    pred_t = list(zip(*pred))   # Transposed matrix of pred.
    n_feature = len(pred_t)     # The number of feature
    n_pred = len(pred)          # The number of prediction
    
    train_idx = train.index.to_list()
    pred_idx = [k + train_idx[-1] for k in range(1, n_pred + 1)]

    fig = plt.figure(figsize=(30, 5 * n_feature))
    
    for i in range(n_feature):
        ax = fig.add_subplot(n_feature, 1, i+1)
        ax.set_title(f'{train.columns[i]}', fontdict={'fontsize': 16,'fontweight':'bold'})
        ax.scatter(train_idx, train.iloc[:, i], s=0.5, c='gray')
        ax.scatter(pred_idx, pred_t[i], s=0.5, c='coral')
    plt.show()


def show_pred_df(train, submission):
    n_feature = len(submission.columns)     # The number of feature
    n_pred = len(submission)          # The number of prediction
    
    train_idx = train.index.to_list()
    pred_idx = [k + train_idx[-1] for k in range(1, n_pred + 1)]

    fig = plt.figure(figsize=(30, 5 * n_feature))
    
    for i in range(n_feature):
        ax = fig.add_subplot(n_feature, 1, i+1)
        ax.set_title(f'{train.columns[i]}', fontdict={'fontsize': 16,'fontweight':'bold'})
        ax.scatter(train_idx, train.iloc[:, i], s=0.5, c='gray')
        ax.scatter(pred_idx, submission.iloc[:, i+1], s=0.5, c='coral')
    plt.show()



    
    
def print_namegroup(namegroup):
    for feature_name in namegroup:
        print(f'{feature_name}: ', end='')
        print(*namegroup[feature_name], sep='  ')


def corr_heatmap(df):
    n = len(df.columns)
    c = df.corr()
    plt.figure(figsize=(n+1, n))
    sns.heatmap(c, annot=True)


def corr_heatmap_with_y(x_df):

    y1 = pd.read_csv('../dataset/y_feature/신호대 잡음비 (각도n).csv')
    y2 = pd.read_csv('../dataset/y_feature/안테나 Gain 평균 (각도n).csv')
    y3 = pd.read_csv('../dataset/y_feature/안테나 n Gain 편차.csv')
    y4 = pd.read_csv('../dataset/y_feature/평균 신호대 잡음비.csv')

    for y_df in [y1, y2, y3, y4]:
        tmp = pd.concat([x_df, y_df], axis=1)
        correlation = tmp.corr()
        correlation = tmp.corr().filter(regex='Y').filter(regex='X', axis=0)

        n = len(correlation.index)
        m = len(correlation.columns)

        plt.figure(figsize=(m, n))
        sns.heatmap(correlation, annot=True)


def x_to_y_visualize(x_column, y_df):
    col = len(y_df.columns)
    fig = plt.figure(figsize=(30, 10 * col))
    
    for i in range(col):
        ax = fig.add_subplot(col, (col+1) // 2, i+1)
        ax.set_title(f'{y_df.columns[i]}', fontdict={'fontsize': 16,'fontweight':'bold'})
        ax.scatter(x_column, y_df.iloc[:, i], s=0.5, c='gray')
    plt.show()