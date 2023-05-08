import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2


def build_and_compile_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(20,), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(3, activation='softmax', kernel_regularizer=l2(0.001)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Load the dataset
    X = pd.read_csv('train_call.tsv', sep='\t')
    y = pd.read_csv('train_clinical.txt', sep='\t', index_col=0)

    # Transform it
    X = X.transpose()
    X = X.tail(-4)

    # Create a dictionary to map the string labels to integers
    class_mapping = {'HER2+': 1, 'HR+': 2, 'Triple Neg': 3}

    # Replace the string labels with integers using the mapping
    y = y.replace(class_mapping)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode the target variable
    y = to_categorical(y - 1, num_classes=3)

    # Number of folds
    k = 5

    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize variables to store the test accuracies and the trained models
    test_accuracies = []
    models = []

    # Convert the one-hot encoded labels back to integer class labels
    y_int = np.argmax(y, axis=1)

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(X, y_int):
        X_trainval, X_test = X[train_index], X[test_index]
        y_trainval, y_test = y[train_index], y[test_index]

        # Feature Selection using L1 regularization
        lasso = LassoCV(cv=5, random_state=42)
        selector = SelectFromModel(lasso, max_features=100).fit(X_trainval, y_trainval.argmax(axis=1))
        X_trainval_selected = selector.transform(X_trainval)
        X_test_selected = selector.transform(X_test)

        # Dimensionality Reduction using PCA
        pca = PCA(n_components=20, random_state=42)
        pca.fit(X_trainval_selected)
        X_trainval_pca = pca.transform(X_trainval_selected)
        X_test_pca = pca.transform(X_test_selected)

        # Build and compile the model
        model = build_and_compile_model()

        # Train the model
        model.fit(X_trainval_pca, y_trainval, epochs=100, batch_size=10, verbose=0)

        # Evaluate the model on the test set
        y_pred = np.argmax(model.predict(X_test_pca), axis=-1)
        y_test_int = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_int, y_pred)

        # Save the test accuracy and the trained model
        test_accuracies.append(accuracy)
        models.append(model)

    # Calculate the average test accuracy
    average_test_accuracy = np.mean(test_accuracies)
    print(f"Average Test Accuracy: {average_test_accuracy:.3f}")


if __name__ == '__main__':
    main()
