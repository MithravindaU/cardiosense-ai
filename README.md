# 🫀 CardioSense-AI
### Heart Disease Classification Using Machine Learning

CardioSense-AI is an educational machine learning project that explores binary heart disease classification using a Random Forest model and structured health-related data.

The project demonstrates a reproducible preprocessing and evaluation pipeline, including missing-value handling, categorical encoding, class-imbalance handling, baseline comparison, and model performance visualizations.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical diagnostic tool, a clinically validated risk calculator, or a substitute for professional medical advice.

## 🎯 Project Objectives

- Build a binary classification pipeline for a heart disease dataset.
- Handle missing values in numerical and categorical features.
- Encode categorical variables consistently during training and prediction.
- Evaluate performance beyond accuracy.
- Compare the trained model with a simple baseline.
- Visualize classification errors, ROC performance, and feature importance.

## ⚙️ Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Google Colab or Jupyter Notebook

## 🔬 Machine Learning Workflow

1. Upload and load the dataset.
2. Inspect the data and missing values.
3. Remove identifier/source columns where present.
4. Convert the original target into a binary label:
   - `0` → No disease indicated by the dataset label
   - `1–4` → Disease indicated by the dataset label
5. Split the data into training and test sets using stratification.
6. Impute missing numerical values using the median.
7. Impute missing categorical values using the most frequent category.
8. Apply one-hot encoding to categorical features.
9. Train a Random Forest classifier with 300 trees and balanced class weights.
10. Evaluate the model on held-out test data.
11. Compare it with a most-frequent-class baseline.
12. Generate evaluation plots and feature-importance visualizations.

## 🧠 Model and Evaluation

**Classifier:** Random Forest

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

Accuracy alone can hide poor performance on one class. Precision, recall, F1-score, the confusion matrix, and ROC-AUC provide additional perspectives on model behavior.

The baseline classifier predicts the most frequent training class. Comparing it with the Random Forest helps determine whether the trained model offers useful improvement over this simple reference.

## 📊 Results and Visualizations

After running the notebook, add the generated plots below. All reported metrics should come from the same held-out test split.

| Visualization | File |
|---|---|
| Confusion matrix | `assets/confusion_matrix.png` |
| ROC curve | `assets/roc_curve.png` |
| Top feature importances | `assets/feature_importance.png` |
| Model metric comparison | `assets/metrics_comparison.png` |

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The matrix shows correct and incorrect predictions for both dataset classes.

### ROC Curve

![ROC Curve](assets/roc_curve.png)

The ROC curve illustrates the trade-off between the true-positive rate and false-positive rate across classification thresholds.

### Feature Importance

![Feature Importance](assets/feature_importance.png)

Random Forest feature importance indicates which encoded input features contributed most to the model's splits. It does not establish causation or medical importance.

### Evaluation Metrics

![Evaluation Metrics](assets/metrics_comparison.png)

The metric comparison summarizes performance on the held-out test set. It should be interpreted alongside the confusion matrix and class-specific report.

## 🚀 Run the Project

### Option 1: Google Colab

1. Open the notebook or Python script in Google Colab.
2. Run the cells in order.
3. Upload the dataset when prompted.
4. Review the evaluation metrics.
5. Download the generated plots and add them to the repository's `assets/` folder.

### Option 2: Local Python Environment

Install the dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the script:

```bash
python heart_disease_prediction.py
```

If running locally, replace the Google Colab file-upload step with a local CSV file path.

## ⚠️ Limitations

- Results depend on the dataset, preprocessing decisions, and train/test split.
- A single split may produce an unstable estimate of performance.
- Feature importance is not evidence of causation.
- The model has not been established as clinically safe or effective.
- The model's predicted probability is not automatically a calibrated medical risk estimate.
- The project does not establish generalization to different hospitals, populations, or data collection settings.

## 🔭 Future Improvements

- Add exploratory data analysis and target-distribution plots.
- Evaluate performance with stratified cross-validation.
- Compare Random Forest with logistic regression and other appropriate baselines.
- Inspect class-specific errors and tune the decision threshold using training/validation data only.
- Add calibration and model-interpretation analysis where appropriate.
- Record dataset provenance, licensing, and reproducible experiment settings.
- Add automated tests for preprocessing and prediction.

## 👩‍💻 Author

Mithravinda U

---

**Educational project. Not intended for clinical use.**
