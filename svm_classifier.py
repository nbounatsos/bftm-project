import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV


def main():
    # Import the data
    X = pd.read_csv('train_call.tsv', sep='\t')
    y = pd.read_csv('train_clinical.txt', sep='\t', index_col=0)

    # Transpose X and remove the first 4 rows
    X = X.transpose()
    X = X.tail(-4)

    # Split to train and test data, 80% to training and 20% to testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Selection on training data, using RFE with 5-fold CV
    svc = SVC(kernel='linear')
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='accuracy')
    rfecv.fit(X_train, y_train.values.ravel())
    print("Optimal number of features : %d" % rfecv.n_features_)

    features = list(X_train.columns[rfecv.support_])
    print('Best features :', features)

    # Remove rest of the features from both train and test set
    X_train, X_test = X_train.filter(features), X_test.filter(features)

    # Support Vector Machines
    SV = SVC(probability=True, random_state=42)
    SV.fit(X_train, y_train.values.ravel())
    print('SVM Classifier Training Accuracy:', SV.score(X_train, y_train.values.ravel()))

    # Split the data into training and validation sets (80/20 split)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    # SVM with Hyper Parameters Tuning
    SV_0 = SVC(probability=True, random_state=42)

    params = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'linear']}

    # Use the validation set for cross-validation during hyperparameter optimization
    SV_1 = GridSearchCV(SV_0, param_grid=params, n_jobs=-1,
                        cv=[(list(range(len(X_train_split))), list(range(len(X_train_split), len(X_train))))],
                        scoring='accuracy')
    SV_1.fit(X_train, y_train.values.ravel())

    print("Best Hyper Parameters:\n", SV_1.best_params_)

    # Calculate the training and validation accuracy
    print('SVM Classifier Training Accuracy:', SV_1.score(X_train_split, y_train_split))
    print('SVM Classifier Validation Accuracy:', SV_1.score(X_val, y_val))

    # Compare the training and validation accuracy
    if abs(SV_1.score(X_train_split, y_train_split) - SV_1.score(X_val, y_val)) < 0.05:
        print("The model is not overfitting.")
    else:
        print("The model might be overfitting.")


if __name__ == '__main__':
    main()

#%%
