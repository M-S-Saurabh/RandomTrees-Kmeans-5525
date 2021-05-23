import numpy as np

from sklearn.metrics import accuracy_score

from utils import display_results, read_data, entropy
from decision_tree import get_rf_learner

def random_forest_fit(X_train, y_train, num_learners, m=3):
    N, D = X_train.shape
    learners = []
    for i in range(num_learners):
        stump = get_rf_learner(X_train, y_train, m)
        learners.append(stump)
    return learners

def random_forest_predict(model, X_test):
    N, D = X_test.shape
    g = np.zeros(N)
    for learner in model:
        attr_idx = learner['attribute']
        thres = learner['threshold']
        column = X_test[:, attr_idx]
        y_pred = np.ones(N)
        y_pred[column < thres] = -1
        y_pred *= learner['flip']
        g += y_pred
    return np.sign(g)

def rf(dataset: str)-> None:
    X_train, X_test, y_train, y_test = read_data(dataset)

    # Sub-problem (i)
    m = 3
    max_learners = 100
    model = random_forest_fit(X_train, y_train, max_learners, m=m)
    error_rates = []
    for num_learners in range(0, max_learners+1, 10):
        if num_learners == 0: num_learners = 1
        y_train_pred = random_forest_predict(model[:num_learners], X_train)
        y_test_pred = random_forest_predict(model[:num_learners], X_test)
        err_rate = ( num_learners,
                     1-accuracy_score(y_train, y_train_pred),
                     1-accuracy_score(y_test, y_test_pred) )
        error_rates.append(err_rate)
    display_results(error_rates, name='random-forest')

    # Sub-problem (ii)
    X_train, X_test, y_train, y_test = read_data(dataset, include_ids=True)
    np.random.seed(42)
    max_learners = 100
    error_rates = []
    for m in range(2,11):
        model = random_forest_fit(X_train, y_train, max_learners, m=m)
        y_train_pred = random_forest_predict(model, X_train)
        y_test_pred = random_forest_predict(model, X_test)
        err_rate = ( m,
                     1-accuracy_score(y_train, y_train_pred),
                     1-accuracy_score(y_test, y_test_pred) )
        error_rates.append(err_rate)
    display_results(error_rates, name='random-forest-varying-m', xlabel='Num of random attrs(m)')

if __name__ == "__main__":
    np.random.seed(42)
    rf('breast-cancer-wisconsin.data')