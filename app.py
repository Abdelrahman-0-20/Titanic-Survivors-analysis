

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="RMS Titanic - Passenger Manifest Analysis",
    layout="wide",
)

REQUIRED_TRAIN_COLS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Fare", "Embarked", "Cabin",
]
REQUIRED_TEST_COLS = [
    "PassengerId", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Fare", "Embarked", "Cabin",
]

TITLE_MAP = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
             "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


 
# Data loading & validation
 
@st.cache_data(show_spinner=False)
def load_csv(uploaded_bytes, fallback_name: str):
    if uploaded_bytes is not None:
        return pd.read_csv(uploaded_bytes)
    p = Path(fallback_name)
    if p.exists():
        return pd.read_csv(p)
    return None


def missing_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


@st.cache_data(show_spinner=False)
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-engineer a raw Titanic-schema dataframe. Safe to call on
    train or test (works whether or not 'Survived' is present)."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    df["Title"] = df["Title"].map(TITLE_MAP).fillna("Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    return df


FEATURE_COLS = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
NUM_COLS = ["Age", "Fare", "FamilySize"]

LABEL_MAP = {
    "Pclass": "Ticket class",
    "Age": "Age",
    "Fare": "Fare paid",
    "FamilySize": "Family size aboard",
    "IsAlone": "Traveling alone",
    "Sex_male": "Sex: male",
    "Embarked_Q": "Boarded Queenstown",
    "Embarked_S": "Boarded Southampton",
    "Title_Master": "Title: Master",
    "Title_Miss": "Title: Miss",
    "Title_Mr": "Title: Mr",
    "Title_Mrs": "Title: Mrs",
    "Title_Rare": "Title: uncommon",
}


def vectorize(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.get_dummies(df[FEATURE_COLS], columns=["Sex", "Embarked", "Title"], drop_first=True)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def group_rate(df: pd.DataFrame, col: str, order: list[str] | None = None) -> pd.DataFrame:
    g = df.groupby(col)["Survived"].agg(["mean", "count"]).reset_index()
    g.columns = ["name", "rate", "count"]
    g["name"] = g["name"].astype(str)
    if order:
        g["__ord"] = g["name"].apply(lambda x: order.index(x) if x in order else 999)
        g = g.sort_values("__ord").drop(columns="__ord")
    return g.reset_index(drop=True)


 
# Modeling
 
@st.cache_resource(show_spinner="Training and cross-validating the models...")
def train_models(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    X = vectorize(df)
    y = df["Survived"]
    columns = list(X.columns)

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    logreg = LogisticRegression(max_iter=1000)
    lr_cv_acc = cross_val_score(logreg, X_scaled, y, cv=skf, scoring="accuracy")
    lr_oof_pred = cross_val_predict(logreg, X_scaled, y, cv=skf, method="predict")
    lr_oof_proba = cross_val_predict(logreg, X_scaled, y, cv=skf, method="predict_proba")[:, 1]
    logreg.fit(X_scaled, y)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed)
    rf_cv_acc = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
    rf_oof_pred = cross_val_predict(rf, X, y, cv=skf, method="predict")
    rf_oof_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    rf.fit(X, y)

    return {
        "logreg": logreg,
        "rf": rf,
        "scaler": scaler,
        "columns": columns,
        "y": y,
        "lr_cv_acc": float(lr_cv_acc.mean()),
        "rf_cv_acc": float(rf_cv_acc.mean()),
        "lr_oof_pred": lr_oof_pred,
        "lr_oof_proba": lr_oof_proba,
        "rf_oof_pred": rf_oof_pred,
        "rf_oof_proba": rf_oof_proba,
    }


def scale_numeric(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X_scaled = X.copy()
    X_scaled[NUM_COLS] = scaler.transform(X[NUM_COLS])
    return X_scaled


def predict_single(bundle: dict, model_choice: str, pclass, sex, age, fare,
                   sibsp, parch, embarked, title):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    row = pd.DataFrame([{
        "Pclass": pclass, "Sex": sex, "Age": age, "Fare": fare,
        "Embarked": embarked, "FamilySize": family_size, "IsAlone": is_alone,
        "Title": title,
    }])
    X_row = vectorize(row, columns=bundle["columns"])

    if model_choice == "Logistic Regression":
        X_scaled = scale_numeric(X_row, bundle["scaler"])
        model = bundle["logreg"]
        prob = model.predict_proba(X_scaled)[0][1]
        contributions = []
        for col, coef in zip(bundle["columns"], model.coef_[0]):
            contrib = coef * X_scaled.iloc[0][col]
            if abs(contrib) > 0.001:
                contributions.append((LABEL_MAP.get(col, col), contrib))
        contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    else:
        model = bundle["rf"]
        prob = model.predict_proba(X_row)[0][1]
        contributions = sorted(
            ((LABEL_MAP.get(c, c), imp) for c, imp in zip(bundle["columns"], model.feature_importances_)),
            key=lambda c: abs(c[1]), reverse=True,
        )
    return prob, contributions, family_size, is_alone


def predict_batch(bundle: dict, model_choice: str, engineered_df: pd.DataFrame) -> np.ndarray:
    X = vectorize(engineered_df, columns=bundle["columns"])
    if model_choice == "Logistic Regression":
        X = scale_numeric(X, bundle["scaler"])
        return bundle["logreg"].predict(X)
    return bundle["rf"].predict(X)


 
# Charts 
 
def rate_bar_chart(data_df: pd.DataFrame, title: str, overall_rate: float, note: str | None = None, height: int = 280) -> None:
    fig = go.Figure()
    fig.add_bar(
        x=data_df["name"],
        y=(data_df["rate"] * 100).round(1),
        customdata=data_df["count"],
        hovertemplate="%{x}<br>%{y:.1f}% survived (n=%{customdata})<extra></extra>",
    )
    fig.add_hline(
        y=overall_rate * 100,
        line_dash="dash",
        annotation_text=f"ship avg {overall_rate * 100:.0f}%",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    if note:
        st.caption(note)


def class_sex_chart(train: pd.DataFrame) -> None:
    pcs = train.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
    classes_present = sorted(train["Pclass"].unique())
    class_labels = [f"Class {i}" for i in classes_present]
    female_vals = [pcs.loc[i, "female"] * 100 if i in pcs.index and "female" in pcs.columns and pd.notna(pcs.loc[i, "female"]) else None for i in classes_present]
    male_vals = [pcs.loc[i, "male"] * 100 if i in pcs.index and "male" in pcs.columns and pd.notna(pcs.loc[i, "male"]) else None for i in classes_present]
    fig = go.Figure()
    fig.add_bar(name="Female", x=class_labels, y=female_vals)
    fig.add_bar(name="Male", x=class_labels, y=male_vals)
    fig.update_layout(
        barmode="group",
        title="Survival by class and sex",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        height=300,
        margin=dict(l=10, r=10, t=70, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def contribution_chart(contributions: list[tuple[str, float]], x_title: str) -> None:
    labels = [c[0] for c in contributions]
    values = [c[1] for c in contributions]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h"))
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title=x_title),
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def confusion_matrix_chart(cm: np.ndarray) -> None:
    labels = ["Did not survive", "Survived"]
    z = cm.tolist()
    z_text = [[str(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Predicted: {l}" for l in labels], y=[f"Actual: {l}" for l in labels],
        text=z_text, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def roc_chart(fpr: np.ndarray, tpr: np.ndarray, auc_value: float) -> None:
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance")
    fig.update_layout(
        title=f"ROC curve (AUC = {auc_value:.3f})",
        xaxis=dict(title="False positive rate"),
        yaxis=dict(title="True positive rate"),
        height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


 
# App
 
def main() -> None:
    with st.sidebar:
        st.header("Data source")
        train_file = st.file_uploader("train.csv (required)", type="csv", key="train_up")
        test_file = st.file_uploader("test.csv (optional)", type="csv", key="test_up")
        sub_file = st.file_uploader("gender_submission.csv (optional)", type="csv", key="sub_up")
        st.caption(
            "If nothing is uploaded, the app looks for train.csv, test.csv, and "
            "gender_submission.csv sitting next to this script."
        )
        st.header("Model settings")
        n_splits = st.slider("Cross-validation folds", min_value=3, max_value=10, value=5,
                             help="How many folds each model is trained and evaluated on.")
        model_choice = st.radio(
            "Model used for predictions",
            ["Logistic Regression", "Random Forest"],
            help="Logistic Regression is slightly less accurate but its estimate can be broken "
                 "down feature by feature. Random Forest usually scores a little higher, but the "
                 "prediction tab can only show its overall feature importances, not a per-passenger "
                 "breakdown.",
        )

    train_raw = load_csv(train_file, "train.csv")
    test_raw = load_csv(test_file, "test.csv")
    sub_raw = load_csv(sub_file, "gender_submission.csv")

    if train_raw is None:
        st.title("RMS Titanic - Passenger Manifest Analysis")
        st.write("Upload train.csv in the sidebar to get started "
                 "(test.csv and gender_submission.csv are optional).")
        st.stop()

    missing_train_cols = missing_required_columns(train_raw, REQUIRED_TRAIN_COLS)
    if missing_train_cols:
        st.error(
            "train.csv is missing the following required column(s): "
            f"{', '.join(missing_train_cols)}. Please upload a file that matches the Kaggle "
            "Titanic schema (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, "
            "Fare, Embarked, Cabin)."
        )
        st.stop()

    if test_raw is not None:
        missing_test_cols = missing_required_columns(test_raw, REQUIRED_TEST_COLS)
        if missing_test_cols:
            st.warning(
                "test.csv is missing column(s) "
                f"{', '.join(missing_test_cols)} and will be ignored for batch prediction."
            )
            test_raw = None

    missing_before_train = train_raw.isnull().sum()
    missing_before_test = test_raw.isnull().sum() if test_raw is not None else None

    train = engineer(train_raw)
    test = engineer(test_raw) if test_raw is not None else None

    overall_rate = train["Survived"].mean()
    bundle = train_models(train, n_splits=n_splits)

    tab_clean, tab_patterns, tab_performance, tab_ideas, tab_predict = st.tabs(
        ["Manifest & cleaning", "Patterns", "Model performance", "Ideas", "Predict"]
    )

    #  Manifest & cleaning 
    with tab_clean:
        st.title("RMS Titanic - passenger manifest analysis")
        st.write(
            f"A cleaned, exploratory look at the {len(train)} passengers with a known outcome, "
            "built for practicing classification, feature engineering, and probability-based "
            "prediction - not for drawing conclusions about any real person."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Passengers analyzed", f"{len(train)}")
        col2.metric("Recorded survival rate", f"{overall_rate * 100:.1f}%")
        col3.metric("Cross-validated accuracy", f"{bundle['lr_cv_acc'] * 100:.1f}%")
        if test is not None:
            st.write(f"Held-out passengers (label withheld): {len(test)}")
        if sub_raw is not None:
            st.caption(
                "gender_submission.csv is not an answer key - it's Kaggle's baseline entry "
                "(\"every woman survives, every man doesn't\"). It's useful as a floor to beat, "
                "not as ground truth for the held-out passengers."
            )

        st.subheader("Missing values, before cleaning")
        missing_df = pd.DataFrame({
            "Field": ["Age", "Cabin", "Embarked", "Fare"],
            "Missing in training set": [f"{int(missing_before_train.get(f, 0))} of {len(train_raw)} ({missing_before_train.get(f, 0) / len(train_raw) * 100:.1f}%)" for f in ["Age", "Cabin", "Embarked", "Fare"]],
            "Missing in held-out set": [
                f"{int(missing_before_test.get(f, 0))} of {len(test_raw)} ({missing_before_test.get(f, 0) / len(test_raw) * 100:.1f}%)"
                if test_raw is not None else "- (no test.csv loaded)"
                for f in ["Age", "Cabin", "Embarked", "Fare"]
            ]
        })
        st.table(missing_df)

        st.subheader("Cleaning steps applied")
        st.markdown(
            """
            - **Title extracted from Name** - "Mr", "Mrs", "Miss", "Master", or
              "Rare" for uncommon ones (Col, Dr, Countess, and similar), pulled out with a regex on
              the name string.
            - **Age imputed by title group median**, not one global median - a
              "Master" (a boy) and a "Mr" (an adult man) get very different fill values, which keeps
              the imputation from flattening the age distribution.
            - **Embarked filled with the mode**, "S" (Southampton).
            - **Fare filled with the median** fare for any missing rows.
            - **Cabin** is too sparse to impute meaningfully, so it's converted to a
              binary *cabin known / unknown* flag rather than dropped outright - cabin
              records survive disproportionately for higher fares, so the flag itself carries
              signal.
            - **Family size and "traveling alone"** engineered from SibSp + Parch + 1.
            """
        )

        st.subheader("Cleaned data preview")
        preview_cols = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare", "Embarked",
                        "Title", "FamilySize", "IsAlone", "CabinKnown"]
        st.dataframe(train[preview_cols].head(15), use_container_width=True, hide_index=True)
        csv_bytes = train.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned train.csv", csv_bytes, file_name="train_cleaned.csv", mime="text/csv")

    #  Patterns 
    with tab_patterns:
        st.subheader("Key pattern")
        st.write(
            "The single clearest pattern in the manifest: sex mattered more "
            "than class, but class still mattered enormously within each sex - a third-class woman "
            "had roughly the same odds as a first-class man."
        )
        class_sex_chart(train)

        col1, col2 = st.columns(2)
        with col1:
            rate_bar_chart(group_rate(train, "Sex", ["female", "male"]), "By sex", overall_rate, height=250)
        with col2:
            rate_bar_chart(group_rate(train, "Pclass", ["1", "2", "3"]), "By ticket class", overall_rate, height=250)

        age_bins = pd.cut(train["Age"], bins=[0, 12, 18, 30, 45, 60, 100],
                          labels=["0-12", "13-18", "19-30", "31-45", "46-60", "60+"])
        train_ab = train.assign(AgeBin=age_bins)
        fam_capped = train["FamilySize"].clip(upper=6).astype(str).replace("6", "6+")
        train_fs = train.assign(FamilySizeCap=fam_capped)
        try:
            fare_bins = pd.qcut(train["Fare"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        except ValueError:
            fare_bins = pd.cut(train["Fare"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        train_fb = train.assign(FareBin=fare_bins)

        col3, col4 = st.columns(2)
        with col3:
            rate_bar_chart(
                group_rate(train_ab, "AgeBin", ["0-12", "13-18", "19-30", "31-45", "46-60", "60+"]),
                "By age group", overall_rate,
                note="Children under 12 clearly outperform the ship average.",
                height=250,
            )
        with col4:
            rate_bar_chart(
                group_rate(train_fs, "FamilySizeCap", ["1", "2", "3", "4", "5", "6+"]),
                "By family size aboard", overall_rate,
                note="Small families (2-4) did better than solo travelers or large families.",
                height=250,
            )

        col5, col6 = st.columns(2)
        with col5:
            rate_bar_chart(group_rate(train, "Title", ["Master", "Miss", "Mrs", "Mr", "Rare"]),
                           "By title (from name)", overall_rate, height=250)
        with col6:
            rate_bar_chart(
                group_rate(train_fb, "FareBin", ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]),
                "By fare quartile", overall_rate,
                note="Fare tracks class closely, so this echoes the class pattern.",
                height=250,
            )

        embarked_labels = {"C": "Cherbourg (C)", "Q": "Queenstown (Q)", "S": "Southampton (S)"}
        train_emb = train.assign(EmbarkedLabel=train["Embarked"].map(embarked_labels))
        rate_bar_chart(
            group_rate(train_emb, "EmbarkedLabel", list(embarked_labels.values())),
            "By port of embarkation", overall_rate,
            note="Cherbourg looks best mostly because it embarked a wealthier, more first-class mix "
                 "of passengers - port itself isn't doing the causal work here.",
            height=250,
        )

    #  Model performance 
    with tab_performance:
        st.subheader("Cross-validated accuracy")
        st.write(
            f"Every number on this tab comes from {n_splits}-fold "
            "cross-validation: each passenger is scored only by a model that never saw that "
            "passenger during training, then the predictions are stitched back together. That's "
            "a more honest estimate of real-world performance than a single train/test split."
        )
        col1, col2 = st.columns(2)
        col1.metric("Logistic Regression mean CV accuracy", f"{bundle['lr_cv_acc'] * 100:.1f}%")
        col2.metric("Random Forest mean CV accuracy", f"{bundle['rf_cv_acc'] * 100:.1f}%")

        inspect_model = st.selectbox("Inspect diagnostics for", ["Logistic Regression", "Random Forest"])
        y_true = bundle["y"]
        if inspect_model == "Logistic Regression":
            y_pred, y_proba = bundle["lr_oof_pred"], bundle["lr_oof_proba"]
        else:
            y_pred, y_proba = bundle["rf_oof_pred"], bundle["rf_oof_proba"]

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, target_names=["Did not survive", "Survived"])
        auc_value = roc_auc_score(y_true, y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.subheader("Confusion matrix (out-of-fold)")
            confusion_matrix_chart(cm)
        with col_roc:
            roc_chart(fpr, tpr, auc_value)

        st.subheader("Precision, recall, F1")
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        if inspect_model == "Logistic Regression":
            st.subheader("Coefficient magnitude (scaled features)")
            coefs = sorted(
                ((LABEL_MAP.get(c, c), v) for c, v in zip(bundle["columns"], bundle["logreg"].coef_[0])),
                key=lambda c: abs(c[1]), reverse=True,
            )
            contribution_chart(coefs, "coefficient (log-odds per standard deviation)")
        else:
            st.subheader("Feature importance")
            importances = sorted(
                ((LABEL_MAP.get(c, c), v) for c, v in zip(bundle["columns"], bundle["rf"].feature_importances_)),
                key=lambda c: abs(c[1]), reverse=True,
            )
            contribution_chart(importances, "importance")

    #  Ideas 
    IDEAS = [
        ("Binary classification practice", "The canonical use of this data - train logistic regression, random forest, gradient boosting, and an SVM side by side, then compare on accuracy, precision/recall, and calibration rather than accuracy alone."),
        ("Feature engineering sandbox", "Titles, family size, cabin-known flags, ticket-group size (people sharing a ticket number) - this dataset is small enough to test one engineered feature at a time and watch cross-validated accuracy move."),
        ("Fairness / bias case study", "Frame it explicitly: a model trained here learns that sex and class predict who lived. That's historically accurate, but it's a clean, low-stakes dataset for practicing how you'd audit a model for encoding a protected attribute before deploying anything with real consequences."),
        ("Data storytelling piece", "Write the class-and-sex chart above into a short piece - 'women and children first' held selectively, and third-class passengers were disadvantaged well before the water reached them (deck location, distance to boats)."),
        ("Missing-data method comparison", "Age is roughly 20% missing - a good size for comparing median imputation, group-median imputation (used here), regression imputation, and dropping rows, and measuring how each choice moves downstream accuracy."),
        ("Explainability practice", "Take the logistic regression here and add SHAP or permutation importance on top of the coefficients already shown - a good first dataset for explainability tooling because you can sanity-check the output against domain intuition."),
        ("API + Docker packaging", "Wrap the trained model in a small FastAPI endpoint and containerize it - same shape as a fraud or churn model, but the input space is small enough to fully document and test."),
    ]

    with tab_ideas:
        st.subheader("What can this dataset be used for?")
        st.write(
            "891 labeled passengers is small, clean, and well-understood - that makes it more useful as a "
            "practice ground than as a place to squeeze out more accuracy."
        )
        for i, (title, body) in enumerate(IDEAS, start=1):
            st.markdown(f"**{i:02d}. {title}**")
            st.write(body)
            st.write("")  # spacing

    #  Predict 
    with tab_predict:
        col_form, col_result = st.columns(2)

        with col_form:
            with st.form("predict_form"):
                st.subheader("Describe a passenger")
                r1c1, r1c2 = st.columns(2)
                pclass = r1c1.selectbox("Ticket class", [1, 2, 3], index=1,
                                        format_func=lambda x: {1: "1st", 2: "2nd", 3: "3rd"}[x])
                sex = r1c2.selectbox("Sex", ["female", "male"])

                r2c1, r2c2 = st.columns(2)
                age = r2c1.number_input("Age", min_value=0, max_value=90, value=29)
                fare = r2c2.number_input("Fare paid", min_value=0.0, max_value=520.0, value=32.0, step=0.5)

                r3c1, r3c2 = st.columns(2)
                sibsp = r3c1.number_input("Siblings/spouse aboard", min_value=0, max_value=10, value=0)
                parch = r3c2.number_input("Parents/children aboard", min_value=0, max_value=10, value=0)

                r4c1, r4c2 = st.columns(2)
                embarked = r4c1.selectbox("Port boarded", ["C", "Q", "S"], index=2,
                                          format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])
                title = r4c2.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

                submitted = st.form_submit_button("Estimate survival odds")

            acc_shown = bundle["lr_cv_acc"] if model_choice == "Logistic Regression" else bundle["rf_cv_acc"]
            st.caption(
                f"Using **{model_choice}** ({n_splits}-fold cross-validated accuracy {acc_shown * 100:.1f}%). "
                "Change the model in the sidebar."
            )

        with col_result:
            st.subheader("Estimate")
            if submitted:
                prob, contributions, family_size, is_alone = predict_single(
                    bundle, model_choice, pclass, sex, age, fare, sibsp, parch, embarked, title
                )
                st.write(f"**Estimated chance of survival: {prob * 100:.1f}%**")
                st.progress(min(max(prob, 0.0), 1.0))
                st.write(f"Family size aboard: {family_size}  ({'traveling alone' if is_alone else 'traveling with family'})")
                if model_choice == "Logistic Regression":
                    st.subheader("What moved the estimate")
                    contribution_chart(contributions, "effect on log-odds")
                    st.caption(
                        "Gold bars pushed the estimate up, black bars pushed it "
                        "down. This is an illustrative model trained on historical records, not a claim "
                        "about any real person."
                    )
                else:
                    st.subheader("What the model weighs overall")
                    contribution_chart(contributions, "feature importance")
                    st.caption(
                        "Random Forest doesn't provide a per-passenger "
                        "breakdown the way Logistic Regression does, so this shows which features "
                        "the model relies on most across all passengers, not just this one. Switch "
                        "to Logistic Regression in the sidebar for a per-passenger explanation."
                    )
            else:
                st.write("Fill in the form and estimate to see a probability and the factors behind it.")

        if test is not None:
            st.subheader("Batch prediction: build a Kaggle submission")
            st.write(
                f"Runs **{model_choice}** over all {len(test)} passengers in test.csv and produces "
                "a two-column PassengerId/Survived file in the exact format Kaggle's Titanic competition "
                "expects for submission."
            )
            if st.button("Generate submission.csv"):
                preds = predict_batch(bundle, model_choice, test)
                submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
                st.dataframe(submission.head(10), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download submission.csv",
                    submission.to_csv(index=False).encode("utf-8"),
                    file_name="submission.csv",
                    mime="text/csv",
                )



if __name__ == "__main__":
    main()