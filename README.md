# Australia_weather_prediction

# Rain in Australia: Predictive Modeling & Evaluation README

This notebook documents the implementation of a full end-to-end Machine Learning pipeline to predict whether it will rain tomorrow in Australia (`RainTomorrow`) using the **Weather in Australia** dataset from Kaggle.

---

## Data Pipeline & Preprocessing Steps
1. **Data Ingestion & Cleaning**:
   * Downloaded the dataset using the Kaggle API.
   * Removed rows with missing targets (`RainToday` or `RainTomorrow`).
2. **Temporal Train-Test Split**:
   * To avoid data leakage and respect chronological sequence, data was split by year:
     * **Training**: Data before 2015 (~98k rows)
     * **Validation**: Year 2015 (~17k rows)
     * **Test**: Years after 2015 (~25.7k rows)
3. **Feature Engineering & Imputation**:
   * Imputed missing numerical values using the mean strategy fitted exclusively on the raw dataset.
   * Scaled numerical variables to a `[0, 1]` range using a `MinMaxScaler`.
   * Encoded categorical features (`Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`) with `OneHotEncoder`.

---

## Models & Performance Comparison

We implemented and compared two models: **Logistic Regression** (baseline) and an **Optimized Random Forest Classifier**.

| Model | Training Accuracy | Validation Accuracy | Test Accuracy | Class 'Yes' Detection (Validation Recall) | Class 'No' Detection (Validation Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** *(liblinear)* | 85.19% | 85.40% | 84.20% | ~46.00% | ~96.00% |
| **Optimized Random Forest** *(depth=16)* | **90.81%** | **85.21%** | **84.16%** | ~40.00% | **97.00%** |

### **Key Insights & Trade-offs**:
* **Accuracy vs. Class Imbalance**: While the **Random Forest Classifier** achieved a much higher training accuracy (90.81%), both models converge to a validation/test accuracy of around **84-85%** because the majority class ('No Rain') dominates the distribution.
* **Recall Trade-offs**: 
  * **Logistic Regression** captured slightly more rainy days (46% recall on 'Yes' in validation) compared to Random Forest (40% recall).
  * **Random Forest** proved slightly more conservative and reliable when predicting no rain, correctly identifying 97% of dry days.
