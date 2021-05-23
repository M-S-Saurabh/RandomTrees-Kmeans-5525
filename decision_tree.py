import numpy as np
from sklearn.metrics import accuracy_score

from utils import entropy

def get_rf_learner(X, y, num_attrs):
    N, D = X.shape
    # Generating a new dataset by random sampling with replacement
    indices = np.random.choice(N, N)
    X_train = X[indices]
    y_train = y[indices]
    # Train stump on the new dataset
    stump = get_stump(X_train, y_train, num_attrs=num_attrs)
    return stump

def get_adaboost_learner(X, y, weights):
    N, D = X.shape
    # by sampling with weight probability
    indices = np.random.choice(N, N, p=weights)
    X_train = X[indices]
    y_train = y[indices]

    # Train stump on the new dataset
    stump = get_stump(X_train, y_train)

    neg_indices = (X[:, stump['attribute']] < stump['threshold'])
    y_pred = np.ones(y.shape)
    y_pred[neg_indices] = -1
    y_pred *= stump['flip']

    # Update weights
    weights *= np.exp(-stump['alpha_t'] * y_pred * y)
    weights /= np.sum(weights) # normalize
    return stump, weights

def get_stump(X_train, y_train, num_attrs=None):
    N, D = X_train.shape
    max_gain = -np.float('inf')
    stump = {}
    if num_attrs is None: attrs = range(D)
    else: attrs = np.random.choice(D, num_attrs, replace=False)
    for i in attrs:
        column = X_train[:, i]
        values = np.unique(column)
        for thres in values:
            neg_indices = (column < thres)
            pos_indices = (column >= thres)
            
            # Calculate information gain
            p0 = np.count_nonzero(neg_indices) / len(column)
            p1 = np.count_nonzero(pos_indices) / len(column)
            curr_entropy = p0 * entropy(y_train[neg_indices]) + p1 * entropy(y_train[pos_indices])
            info_gain = 1 - curr_entropy

            # If info_gain is the highest so far, save this stump
            if max_gain < info_gain:
                max_gain = info_gain
                stump['attribute'] = i
                stump['threshold'] = thres
                # Determine flip based on accuracy
                y_pred = np.ones(y_train.shape)
                y_pred[neg_indices] = -1
                err_rate = 1 - accuracy_score(y_train, y_pred)
                flip = 1.0
                if err_rate > 0.5:
                    err_rate = 1.0 - err_rate
                    flip = -1.0
                stump['flip'] = flip
                stump['alpha_t'] = 0.5 * np.log( (1-err_rate)/(err_rate+1e-10) )
    return stump

if __name__ == "__main__":
    # Testing decision stump
    print('Testing decision stump')
    # from sklearn.datasets import load_iris
    # X, y = load_iris(return_X_y=True)

    # from sklearn.tree import DecisionTreeClassifier
    # model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    # model.fit(X,y)
    # print(model.feature_importances_)
    # print(model.tree_.feature)
    # print(model.tree_.threshold)
    
    X = np.array([[1,1,1],[1,1,0],[0,0,1],[1,0,0]])
    y = np.array([1,1,-1,-1])
    weights = np.full(4, 1/4)
    for i in range(4):
        clf, weights = get_learner(X, y, weights=weights)

    from adaboost import adaboost_predict
    preds = adaboost_predict([clf], X)
    print(preds)
