"""
run_solution.py by Ben Bay
"""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC, Precision, Recall

SEED = 42


def process_base(df):
    result = df[["id", "age", "sex", "married", "highBPDiagnosed"]]
    with pd.option_context("mode.chained_assignment", None):
        result["age"] = result["age"].replace(-1, result["age"].median())
    bad_vals = [
        "UNDER 16 - INAPPLICABLE",
        "Inapplicable",
        "not ascertained",
        "Refused",
        "DK",
    ]
    for val in bad_vals:
        result = result[result != val]
    result["highBPDiagnosed"] = result["highBPDiagnosed"].replace(
        {"Yes": 1}, regex=True
    )
    result["highBPDiagnosed"] = result["highBPDiagnosed"].replace({"No": 0}, regex=True)
    result["sex"] = result["sex"].replace({"Female": 0}, regex=True)
    result["sex"] = result["sex"].replace({"Male": 1}, regex=True)
    result = result.sort_values(by=["id"])
    result = result.dropna()
    result = result.drop_duplicates()
    married_hots = pd.get_dummies(result["married"])
    result = result.drop("married", axis=1)
    result = pd.concat((result, married_hots), axis=1)
    cols = list(result.columns)
    cols.remove("highBPDiagnosed")
    cols.append("highBPDiagnosed")
    result = result[cols]
    return result


def process_meds(df, base_ids, threshold=1000):
    result = df[["id", "rxName"]]
    result = result[result["id"].isin(base_ids)]
    result = result.groupby("rxName").filter(lambda x: len(x) > threshold)
    result = result.sort_values(by=["id"])
    result = result.dropna()
    result = result.drop_duplicates()
    return result


def multihot_encode_meds(df):
    result = df.pivot_table(
        index=["id"], columns=["rxName"], aggfunc=[len], fill_value=0
    )
    result.columns = [x[1] for x in result.columns]
    return result


def prepare_data(base_path, meds_path):
    base = pd.read_csv(base_path)
    meds = pd.read_csv(meds_path)

    base_df = process_base(base)
    base_ids = set(base_df["id"])

    meds_df = process_meds(meds, base_ids)
    meds_ids = set(meds_df["id"])

    base_df = base_df[base_df["id"].isin(meds_ids)]
    base_ids = set(base_df["id"])

    assert base_ids == meds_ids

    vec_df = multihot_encode_meds(meds_df)

    result = pd.merge(
        vec_df,
        base_df,
        how="inner",
        on="id",
    )

    result["age"] = (result["age"] - result["age"].min()) / (
        result["age"].max() - result["age"].min()
    )

    result = result.to_numpy()

    train, test = train_test_split(
        result, test_size=0.3, stratify=result[:, -1], random_state=SEED
    )
    X_train = train[:, 1:-1]
    y_train = train[:, -1]
    X_test = test[:, 1:-1]
    y_test = test[:, -1]
    return X_train, y_train, X_test, y_test


def evaluate(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, auc, f1


def dnn():
    tf.random.set_seed(SEED)
    input_dims = 214
    model = tf.keras.models.Sequential(
        [
            Dense(64, activation="selu", input_shape=(input_dims,)),
            Dense(64, activation="selu"),
            Dense(64, activation="selu"),
            Dense(64, activation="selu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tf.keras.optimizers.Nadam()
    loss_func = tf.keras.losses.BinaryCrossentropy()
    model.compile(
        loss=loss_func,
        optimizer=optimizer,
        metrics=[
            AUC(),
            Precision(),
            Recall(),
        ],
    )
    return model


def keras_dnn():
    return KerasClassifier(build_fn=dnn, epochs=1, batch_size=16, verbose=1)


def stacked_ensemble(seed, deep=False):
    estimators = []
    estimators.append(("lr", LogisticRegression(random_state=seed)))
    estimators.append(("lgbm", lgb.LGBMClassifier(random_state=seed)))
    estimators.append(("forest", RandomForestClassifier(random_state=seed)))
    if deep:
        estimators.append(("dnn", keras_dnn()))
    return StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(random_state=seed)
    )


def run(base, meds, stacked, deep):
    X_train, y_train, X_test, y_test = prepare_data(base, meds)

    table = []

    print(f"* Random baseline... ", end="")
    np.random.seed(seed=SEED)
    rnd_baseline_pred = np.random.randint(2, size=len(y_test))
    baseline = evaluate(rnd_baseline_pred, y_test)
    table.append(["Random baseline", *baseline])
    print(f"Done.")

    print(f"* Logistic regression... ", end="")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = y_pred.round()
    scores = evaluate(y_pred, y_test)
    table.append(["Logistic regression", *scores])
    print(f"Done.")

    print(f"* Gradient-boosted trees... ", end="")
    model = lgb.LGBMClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = y_pred.round()
    scores = evaluate(y_pred, y_test)
    table.append(["Gradient-boosted trees", *scores])
    print(f"Done.")

    print(f"* Random forest... ", end="")
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = y_pred.round()
    scores = evaluate(y_pred, y_test)
    table.append(["Random forest", *scores])
    print(f"Done.")

    if deep:
        print(f"* Deep neural network... ", end="")
        model = dnn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = y_pred.round()
        scores = evaluate(y_pred, y_test)
        table.append(["Deep neural network", *scores])
        print(f"Done.")

    if stacked:
        print(f"* Stacked ensemble... ", end="")
        model = stacked_ensemble(SEED, deep=deep)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = evaluate(y_pred, y_test)
        table.append(["Stacked ensemble", *scores])
        print(f"Done.")

    print(tabulate(table, headers=["Learner", "Accuracy", "AUC", "F1"]))


def process_args(args):
    """This function contains the logic for processing the argparser."""
    base = "meps_base_data.csv"
    meds = "meps_meds.csv"
    stacked = False
    deep = False
    if args.base:
        base = args.base
    if args.meds:
        meds = args.meds
    if args.stack:
        stacked = True
    if args.deep:
        deep = True
    run(base, meds, stacked, deep)
    return 0


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Build a model that predicts whether a disease is present in a patient, given patient and medication history data."
    )
    parser.add_argument("--base", action="store", type=str, help="Path of base csv.")
    parser.add_argument("--meds", action="store", type=str, help="Path of meds csv.")
    parser.add_argument(
        "--stack", action="store_true", help="Use stacking to combine weak learners."
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Train a deep neural network in addition to simpler ML models.",
    )
    parser.set_defaults(func=process_args)
    return parser


def main():
    parser = setup_argparse()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""
FUTURE WORK
* compress medication multi-hot encoding vector with PCA or an autoencoder.
* consider additional patient features.
* add more thorough testing.
"""
