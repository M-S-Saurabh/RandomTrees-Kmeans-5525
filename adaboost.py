import numpy as np

from sklearn.metrics import accuracy_score

from utils import display_results, read_data
from decision_tree import get_adaboost_learner

def adaboost_fit(X_train, y_train, num_learners):
    N, D = X_train.shape
    weights = np.full(N, 1/N)
    learners = []
    for i in range(num_learners):
        stump, weights = get_adaboost_learner(X_train, y_train, weights)
        learners.append(stump)
    return learners

def adaboost_predict(model, X_test):
    N, D = X_test.shape
    g = np.zeros(N)
    for stump in model:
        neg_indices = (X_test[:, stump['attribute']] < stump['threshold'])
        y_pred = np.ones(N)
        y_pred[neg_indices] = -1
        y_pred *= stump['flip']
        g += (stump['alpha_t'] * y_pred)
    return np.sign(g)

def adaboost(dataset: str) -> None:
    X_train, X_test, y_train, y_test = read_data(dataset, test_size=0.5)
    error_rates = []
    max_learners = 100
    model = adaboost_fit(X_train, y_train, max_learners)
    for num_learners in range(0, max_learners+1):
        if num_learners == 0: num_learners = 1
        y_train_pred = adaboost_predict(model[:num_learners], X_train)
        y_test_pred = adaboost_predict(model[:num_learners], X_test)
        err_rate = ( num_learners,
                     1-accuracy_score(y_train, y_train_pred),
                     1-accuracy_score(y_test, y_test_pred) )
        error_rates.append(err_rate)
    display_results(error_rates)

if __name__ == "__main__":
    np.random.seed(42)
    adaboost('breast-cancer-wisconsin.data')