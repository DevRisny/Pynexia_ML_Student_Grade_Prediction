# Student Grade Prediction

Pynexia Machine Learning Internship — Week 1 project (Supervised Regression Analysis).

Predicts a student's final grade from attendance, weekly study hours, parental education and previous marks, and flags students who may need extra academic support.

## Models and results (20% hold-out, 1,000 students)
| Model | R² | RMSE | 5-fold CV R² |
|---|---|---|---|
| Linear Regression | 0.7445 | 5.04 | 0.7800 |
| Ridge | **0.7447** | **5.04** | 0.7800 |
| Decision Tree (depth 5) | 0.5727 | 6.52 | 0.6267 |

Ridge and Linear Regression perform almost identically; both clearly beat the Decision Tree, which means the relationship between the features and the final grade is mostly linear. `previous_marks` has the strongest impact, followed by `study_hours`, `parent_education` and `attendance`.

Students predicted below the pass mark (50) are written to `students_needing_support.csv` so teachers can offer early help.

## Outputs
- `model_results.csv`, `students_needing_support.csv`, `student_data.csv`
- `actual_vs_predicted.png`, `feature_importance.png`, `correlation_heatmap.png`

## Run
```bash
pip install -r requirements.txt
python student_grade_prediction.py
```

## Data
The script uses the UCI *Student Performance* dataset if `student-mat.csv` is placed next to it; otherwise it generates a realistic synthetic dataset (seeded, reproducible) so the project runs end-to-end. The results above are from the synthetic dataset.

Author: Junaideen Risny Suha
