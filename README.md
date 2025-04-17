# 🏡 Boston Housing Price Prediction using Decision Tree and Random Forest Regression

This is Day 9 of 30 ML models in 30 Days. This project predicts housing prices in Boston using **Decision Tree Regression** and compares its performance to a more powerful ensemble model, **Random Forest Regression**. The focus is on model evaluation, hyperparameter tuning, and deriving actionable insights from metrics like **Mean Squared Error (MSE)** and **R² Score**.

---

## 📊 Dataset: Boston Housing

The [Boston Housing Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-house-prices-dataset) includes 506 records and 13 features related to housing characteristics and socioeconomic factors in Boston.  
**Target variable**: `MEDV` — Median value of owner-occupied homes (in $1000s).

### Key Features:
- `CRIM`: Crime rate by town  
- `RM`: Average number of rooms per dwelling  
- `LSTAT`: % of lower status population  
- `NOX`: Nitric oxides concentration  
- `PTRATIO`: Pupil-teacher ratio  
- And more features tied to location, zoning, and accessibility can all be reffered to from the ipynb file.

---

## 🧠 Model 1: Decision Tree Regressor

The **Decision Tree Regressor** builds a model in the form of a binary tree, splitting nodes to reduce prediction error based on feature thresholds.

### ✅ Pros:
- Highly interpretable
- Handles non-linearity well
- No need for feature scaling

### ⚠️ Cons:
- Sensitive to noise
- Easily overfits on training data if not pruned

---

## ⚙️ Workflow

1. **Preprocessing**  
   - Loaded and explored the dataset  
   - Train-test split (80/20)

2. **Baseline Model**  
   - Trained a basic Decision Tree  
   - Evaluated using R² and MSE

3. **Hyperparameter Tuning**  
   - Used `GridSearchCV` with 5-fold cross-validation  
   - Tuned `max_depth`, `min_samples_split`, `min_samples_leaf`

4. **Model Comparison**  
   - Trained a **Random Forest Regressor** as a benchmark  
   - Compared metrics and generalization ability

---

## 📈 Evaluation Metrics

| Model |  Cross-Validated R² (CV)  |
|-------|----------------|
| **Decision Tree (baseline)**| ~0.19 → **0.52 (tuned)** |
| **Random Forest** | **0.63** |

> - **R² Score**: Proportion of variance explained by the model (closer to 1 is better).  
> - **MSE**: Measures average squared difference between predicted and actual values (lower is better).

---

## 🔧 Best Hyperparameters (Decision Tree)

```json
{
  "max_depth": 5,
  "min_samples_split": 2,
  "min_samples_leaf": 4
}
```

These parameters significantly improved cross-validated R² from **0.19 to 0.52**, reducing overfitting.

---

## 🌲 Why Random Forest Performs Better

Random Forest overcomes the limitations of a single tree by:
- Building multiple trees on random subsets of the data
- Averaging their outputs for stability
- Reducing variance and overfitting risk

This ensemble method led to a **higher R² score (0.63)** and **lower MSE (7.90)**, making it the superior choice in this regression task.

---

## ✅ Conclusion

- **Decision Tree Regressor** serves as a great baseline and offers model interpretability.
- **Hyperparameter tuning** is essential to improve generalization.
- **Random Forest** outperforms the single-tree model in both accuracy and stability.
- **Ensemble methods** are generally preferred for real-world regression tasks.

---

## 📁 Project Files

- `D9 Decision Tree Regressor.ipynb` — Jupyter Notebook with full code and analysis  
- Dataset: Loaded using `sklearn.datasets.fetch_openml(name="boston")`

---
