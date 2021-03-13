import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from category_encoders import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Performance Report
def evaluate(y_test, y_pred):
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy : ")
    print(accuracy_score(y_test, y_pred) * 100)

    print("Report : ")
    report = classification_report(y_test, y_pred)
    print(report)


def main():
    # Preprocess the data
    # start your code here

    # Load data
    data = pd.read_csv("bank.csv")

    # Fix typo in column name
    data.rename(columns={"subcribed": "subscribed"}, inplace=True)

    # Encoding features
    data = data.replace({"yes": 1, "no": 0})
    ohe = OneHotEncoder(
        cols=["job", "marital", "education", "contact", "month", "poutcome"],
        use_cat_names=True,
        return_df=True,
    )
    data = ohe.fit_transform(data)

    # print(data.head())

    # Get features and target
    X = data.drop(columns=["subscribed"])
    y = data["subscribed"]

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100
    )

    # end your code here

    # print(
    #     "\n\nDecision Tree: -------------------------------------------------------------------------\n\n"
    # )
    # # start your code here

    # tree_classifier = DecisionTreeClassifier(
    #     max_depth=4,
    #     max_leaf_nodes=4,
    #     random_state=100,
    # )

    # tree_classifier.fit(X_train, y_train)
    # y_pred_tree = tree_classifier.predict(X_test)
    # evaluate(y_test, y_pred_tree)

    # # feature_imp_tree = pd.Series(
    # #     tree_classifier.feature_importances_, index=X_train.columns
    # # ).sort_values(ascending=False)[:10]
    # # print(feature_imp_tree)

    # # plt.figure(figsize=(20, 10))

    # # plot_tree(
    # #     tree_classifier,
    # #     feature_names=X_train.columns,
    # #     class_names=["no", "yes"],
    # #     rounded=True,
    # # )
    # # plt.savefig("decision_tree.svg", bbox_inches="tight")
    # # plt.show()

    # # end your code here

    # print(
    #     "\n\nRandom Forest: -------------------------------------------------------------------------\n\n"
    # )
    # # start your code here
    # rf_classifier = RandomForestClassifier(
    #     # bootstrap=False,
    #     criterion="entropy",
    #     max_depth=9,
    #     max_leaf_nodes=21,
    #     min_samples_leaf=5,
    #     random_state=100,
    # )

    # rf_classifier.fit(X_train, y_train)
    # y_pred_rf = rf_classifier.predict(X_test)
    # evaluate(y_test, y_pred_rf)

    # feature_imp_rf = pd.Series(
    #     rf_classifier.feature_importances_, index=X_train.columns
    # ).sort_values(ascending=False)[:10]
    # print(feature_imp_rf)
    # # end your code here

    print(
        "\n\nXGBoost: -------------------------------------------------------------------------\n\n"
    )
    # start your code here
    xgb_classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=5,
        use_label_encoder=False,
        colsample_bytree=0.3,
    )

    xgb_classifier.fit(X_train, y_train)
    y_pred_xgb = xgb_classifier.predict(X_test)
    evaluate(y_test, y_pred_xgb)

    # feature_imp_xgb = pd.Series(
    #     xgb_classifier.feature_importances_, index=X_train.columns
    # ).sort_values(ascending=False)[:10]
    # print(feature_imp_xgb)

    # plt.figure(figsize=(10, 5))

    # y_pos = np.arange(len(feature_imp_xgb))
    # plt.bar(y_pos, feature_imp_xgb, align="center", color="purple")
    # plt.xticks(y_pos, feature_imp_xgb.index, rotation=30, ha="right")
    # plt.xlabel("Features")
    # plt.title("Feature Importance", weight="bold", fontsize=18, pad=20)

    # plt.savefig("xgb_features.svg", bbox_inches="tight")
    # plt.show()
    # # end your code here

    # print(
    #     "\n\nVoting: -------------------------------------------------------------------------\n\n"
    # )
    # # start your code here
    # vt_classifier = VotingClassifier(
    #     estimators=[
    #         ("tree", tree_classifier),
    #         ("rf", rf_classifier),
    #         ("xgb", xgb_classifier),
    #     ]
    # )
    # vt_classifier.fit(X_train, y_train)
    # y_pred_vt = vt_classifier.predict(X_test)
    # evaluate(y_test, y_pred_vt)
    # # end your code here


if __name__ == "__main__":
    main()
