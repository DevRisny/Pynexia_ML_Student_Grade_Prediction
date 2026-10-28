"""
Student Grade Prediction - Pynexia Machine Learning Internship, Week 1.

Predicts a student's final grade (0-100) from attendance, weekly study hours,
parental education and previous marks using Linear Regression, Ridge and a
Decision Tree regressor, scored with R^2 and RMSE. Students predicted to score
below a pass threshold are flagged for extra academic support.

The UCI "Student Performance" dataset can be used by dropping `student-mat.csv`
next to this script (see load_data). If it is not present, a realistic synthetic
dataset is generated so the project always runs end-to-end.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
PASS_MARK = 50
OUT = Path(__file__).parent
FEATURES = ["attendance", "study_hours", "parent_education", "previous_marks"]
PALETTE = {"Linear Regression": "#6c5ce7", "Ridge": "#00b894", "Decision Tree": "#ff7aa8"}


def load_data(n: int = 1000) -> pd.DataFrame:
    """Load UCI student-mat.csv if available, otherwise generate synthetic data."""
    uci = OUT / "student-mat.csv"
    if uci.exists():
        raw = pd.read_csv(uci, sep=";")
        return pd.DataFrame({
            "attendance": (100 - raw["absences"].clip(upper=40) * 2.0),
            "study_hours": raw["studytime"] * 2.5,
            "parent_education": ((raw["Medu"] + raw["Fedu"]) / 2).round(),
            "previous_marks": raw["G1"] * 5,
            "final_grade": raw["G3"] * 5,
        })

    rng = np.random.default_rng(SEED)
    attendance = rng.normal(82, 11, n).clip(40, 100)
    study_hours = rng.gamma(4.0, 1.6, n).clip(0, 25)
    parent_education = rng.integers(0, 5, n)  # 0 none ... 4 postgraduate
    previous_marks = rng.normal(62, 16, n).clip(10, 100)
    noise = rng.normal(0, 5, n)
    final = (
        0.52 * previous_marks + 0.22 * attendance + 1.1 * study_hours
        + 1.6 * parent_education - 8 + noise
    ).clip(0, 100)
    return pd.DataFrame({
        "attendance": attendance.round(1), "study_hours": study_hours.round(1),
        "parent_education": parent_education, "previous_marks": previous_marks.round(1),
        "final_grade": final.round(1),
    })


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    df = load_data()
    df.to_csv(OUT / "student_data.csv", index=False)
    print(f"Dataset: {df.shape[0]} students, {len(FEATURES)} features\n")
    print(df.describe().round(2).T[["mean", "std", "min", "max"]], "\n")

    X, y = df[FEATURES], df["final_grade"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=SEED),
    }

    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        cv = cross_val_score(model, X, y, cv=5, scoring="r2")
        rows.append({"Model": name, "R2": r2_score(y_test, p), "RMSE": rmse(y_test, p), "CV R2 (5-fold)": cv.mean()})

    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("Model comparison (hold-out 20%)\n", results, "\n")
    results.to_csv(OUT / "model_results.csv")

    best = results["R2"].idxmax()
    print(f"Best model: {best}")
    lin = models["Linear Regression"].named_steps["linearregression"]
    coefs = pd.Series(lin.coef_, index=FEATURES).sort_values(ascending=False).round(2)
    print("\nStandardised linear coefficients (impact on final grade):\n", coefs, "\n")

    # Students who may need additional support
    at_risk = X_test.copy()
    at_risk["predicted_grade"] = preds[best].round(1)
    at_risk = at_risk[at_risk["predicted_grade"] < PASS_MARK].sort_values("predicted_grade")
    at_risk.to_csv(OUT / "students_needing_support.csv")
    print(f"{len(at_risk)} of {len(X_test)} test students predicted below {PASS_MARK} -> flagged for support")

    # ---- charts (light theme) ---------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": "#ffffff", "figure.facecolor": "#f4f7ff"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, name in zip(axes, models):
        ax.scatter(y_test, preds[name], s=14, alpha=.55, color=PALETTE[name])
        ax.plot([0, 100], [0, 100], "--", color="#8a95b5", lw=1)
        ax.set_title(f"{name}\nR²={results.loc[name, 'R2']:.3f}  RMSE={results.loc[name, 'RMSE']:.2f}", fontsize=11, color="#1f2a44")
        ax.set_xlabel("Actual grade"); ax.set_ylabel("Predicted grade"); ax.grid(alpha=.25)
    fig.suptitle("Student Grade Prediction - actual vs predicted", fontsize=14, fontweight="bold", color="#1f2a44")
    fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(coefs.index[::-1], coefs.values[::-1], color="#6c5ce7")
    ax.set_title("Feature impact (standardised coefficients)", color="#1f2a44"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=140); plt.close(fig)

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="PuBuGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right"); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#1f2a44")
    ax.set_title("Feature correlation heatmap", color="#1f2a44")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=140); plt.close(fig)
    print("\nSaved: model_results.csv, students_needing_support.csv, student_data.csv and 3 charts")


if __name__ == "__main__":
    main()
