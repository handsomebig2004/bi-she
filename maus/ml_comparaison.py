import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import time


# fetch feature data
feat_inf_ecg = pd.read_pickle("modified_features/feat_inf_ecg.pkl")
feat_inf_ppg = pd.read_pickle("modified_features/feat_inf_ppg.pkl")
feat_pix_ppg = pd.read_pickle("modified_features/feat_pix_ppg.pkl")
label = pd.read_pickle("modified_features/label.pkl")
obj_position = pd.read_pickle("modified_features/obj_position.pkl")
all_features = np.array([list(feat_inf_ecg[i]) + list(feat_inf_ppg[i]) + list(feat_pix_ppg[i]) for i in  range(len(feat_pix_ppg))])

# simplify data for binary classification (for now)
label[label == 2] = 1
label[label == 3] = 1  


def compare_diff_class(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # linear SVC (with the same parameters as them)
    lin_svc = LinearSVC(C=20.001, class_weight={0:0.67, 1:0.33})
    lin_svc.fit(X_train, y_train)

    print("\nLinear SVC:")
    print(f"f1:  {f1_score(y_test, lin_svc.predict(X_test)):.4f}")
    print(f"acc: {accuracy_score(y_test, lin_svc.predict(X_test)):.4f}")


    # SVC
    tuned_parameters = {'degree': range(2,6), 'C': [0.001, 0.01, 0.1, 1]}
    my_kfold = KFold(n_splits=10)

    SVC_grid = GridSearchCV(SVC(kernel='poly'),
                            tuned_parameters,
                            cv=my_kfold,
                            n_jobs=-1)
    SVC_grid.fit(X_train, y_train)

    print("\nSVC:")
    print(f"f1:  {f1_score(y_test, SVC_grid.predict(X_test)):.4f}")
    print(f"acc: {accuracy_score(y_test, SVC_grid.predict(X_test)):.4f}")


    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=500)
    random_forest.fit(X_train, y_train)

    print("\nRandom Forest:")
    print(f"f1:  {f1_score(y_test, random_forest.predict(X_test)):.4f}")
    print(f"acc: {accuracy_score(y_test, random_forest.predict(X_test)):.4f}")


def compare_diff_class_mean(X, y):
    
    start = time.clock_gettime(0)
    
    lin_svc_f1 = 0
    lin_svc_acc = 0
    svc_f1 = 0
    svc_acc = 0
    svc_poly_f1 = 0
    svc_poly_acc = 0
    rd_forest_f1 = 0
    rd_forest_acc = 0
    
    for i in range(10):
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        # linear SVC (with the same parameters as them)
        lin_svc = LinearSVC(C=20.001, class_weight={0:0.67, 1:0.33})
        lin_svc.fit(X_train, y_train)

        lin_svc_f1 += f1_score(y_test, lin_svc.predict(X_test))
        lin_svc_acc += accuracy_score(y_test, lin_svc.predict(X_test))


        # SVC rbf
        tuned_parameters = {'degree': range(2, 5), 'gamma': [1e-7, 1e-6, 1e-5, 0.0001, 0.1], 'C': [0.001, 0.01, 0.1, 1]}
        my_kfold = KFold(n_splits=10)

        SVC_grid = GridSearchCV(SVC(kernel='rbf'),
                                tuned_parameters,
                                cv=my_kfold,
                                n_jobs=-1)
        SVC_grid.fit(X_train, y_train)

        svc_f1 += f1_score(y_test, SVC_grid.predict(X_test))
        svc_acc += accuracy_score(y_test, SVC_grid.predict(X_test))


        # SVC poly
        tuned_parameters = {'degree': range(2,6), 'C': [0.001, 0.01, 0.1, 1]}
        my_kfold = KFold(n_splits=10)

        SVC_poly_grid = GridSearchCV(SVC(kernel='poly'),
                                tuned_parameters,
                                cv=my_kfold,
                                n_jobs=-1)
        SVC_poly_grid.fit(X_train, y_train)

        svc_poly_f1 += f1_score(y_test, SVC_poly_grid.predict(X_test))
        svc_poly_acc += accuracy_score(y_test, SVC_poly_grid.predict(X_test))

        # Random Forest
        random_forest = RandomForestClassifier(n_estimators=500)
        random_forest.fit(X_train, y_train)

        rd_forest_f1 += f1_score(y_test, random_forest.predict(X_test))
        rd_forest_acc += accuracy_score(y_test, random_forest.predict(X_test))

    print("\n Linear SVC:")
    print(f"  f1:  {lin_svc_f1/10:.4f}")
    print(f"  acc: {lin_svc_acc/10:.4f}")

    print("\n SVC with rbf:")
    print(f"  f1:  {svc_f1/10:.4f}")
    print(f"  acc: {svc_acc/10:.4f}")

    print("\n SVC with poly:")
    print(f"  f1:  {svc_poly_f1/10:.4f}")
    print(f"  acc: {svc_poly_acc/10:.4f}")

    print("\n Random Forest:")
    print(f"  f1:  {rd_forest_f1/10:.4f}")
    print(f"  acc: {rd_forest_acc/10:.4f}")
    
    print(f"\nin {(time.clock_gettime(0) - start):.2f} seconds\n\n")


if __name__ == "__main__":
    
    print("inf_ecg:")
    compare_diff_class_mean(feat_inf_ecg, label)
    print("inf_ppg:")
    compare_diff_class_mean(feat_inf_ppg, label)
    print("pix_ppg")
    compare_diff_class_mean(feat_pix_ppg, label)
    print("all features")
    compare_diff_class_mean(all_features, label)
    

# normalisation z-score pour données cardiaques (pas avec l'EDA)
# ou min-max

