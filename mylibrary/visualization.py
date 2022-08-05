import matplotlib.pyplot as plt


def pred_visualize(train, pred):
    
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
    
    
def print_namegroup(namegroup):
    
    for feature_name in namegroup:
        print(f'{feature_name}: ', end='')
        print(*namegroup[feature_name], sep='  ')