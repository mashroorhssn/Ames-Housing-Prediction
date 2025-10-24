# Ames-Housing-Prediction
# Ames Housing Price Prediction

This project is a submission for the "Applied Machine Learning for Business" course. It demonstrates a complete machine learning workflow to predict house prices using the Ames, Iowa housing dataset.

The model is built using a `RandomForestRegressor` and follows a robust preprocessing and validation methodology based on course materials.

---

## 1. Project Goal

The objective was to build an accurate and reliable regression model to predict the final sale price of a house based on its features (e.g., quality, size, location, condition).

* **Dataset:** Ames Housing
* **Algorithm:** `RandomForestRegressor`
* **Key Methodology:** Train-Valid-Test Split

---

## 2. Methodology

The process followed a structured machine learning pipeline:

### Data Loading and Inspection
* Loaded the `train.csv` dataset, which contains 1460 rows and 81 columns.
* Initial analysis with `.info()` revealed a complex dataset with 41 categorical and 34 numerical features, plus significant missing values.

### Preprocessing and Feature Engineering
A `ColumnTransformer` pipeline was built to automate the entire data cleaning and preparation process:

* **Target Variable:** The `SalePrice` was highly skewed and was **log-transformed** (`np.log1p`) to create a normal distribution, improving model performance.
* **Feature Selection:** Dropped `Id` and columns with >80% missing data (`Alley`, `PoolQC`, `Fence`, `MiscFeature`).
* **Numerical Pipeline:**
    1.  Imputed missing values using the `median`.
    2.  Scaled all features using `StandardScaler`.
* **Categorical Pipeline:**
    1.  Imputed missing values using the `most_frequent` value.
    2.  Converted all 41 categorical features into numerical format using `OneHotEncoder`.

### Model Training & Validation
1.  **Train-Valid-Test Split:** To prevent overfitting and get an unbiased evaluation, the data was split into three sets:
    * **Training Set:** 70% (1022 rows)
    * **Validation Set:** 15% (219 rows)
    * **Test Set:** 15% (219 rows)
2.  **Algorithm:** A `RandomForestRegressor` was chosen. This is a powerful ensemble model that builds hundreds of decision trees and averages their results. It is excellent for this dataset as it naturally handles a large number of features and is highly robust against overfitting.
3.  **Training:** The model pipeline was trained *only* on the training set.

---

## 3. Results and Key Findings

The model's performance was evaluated on the unseen **Test Set**.

### Model Performance
* **$R^2$ Score:** **0.9005**
    * This indicates that the model successfully explained **90.05%** of the variance in house prices, demonstrating a very strong and reliable fit.
* **RMSE (Root Mean Squared Error):** **$30,014.27**
    * This is the real-world error. On average, the model's price prediction is off by approximately $30,014. This is a strong result given the wide range of house prices in the dataset.

### Top 15 Most Important Features
The model identified `OverallQual` as the single most important driver of price, accounting for 55% of the predictive power. This aligns with real-world intuition that the material and finish of a house are paramount to its value.

| Feature | Importance |
|:---|---:|
| **OverallQual** | **0.549773** |
| GrLivArea | 0.0989935 |
| GarageCars | 0.045296 |
| TotalBsmtSF | 0.034865 |
| GarageArea | 0.031209 |
| 1stFlrSF | 0.021151 |
| BsmtFinSF1 | 0.020179 |
| LotArea | 0.016612 |
| YearBuilt | 0.012216 |
| YearRemodAdd | 0.011319 |
| GarageFinish\_Unf | 0.009906 |
| 2ndFlrSF | 0.009179 |
| OverallCond | 0.008767 |
| BsmtQual\_Ex | 0.007108 |
| OpenPorchSF | 0.005853 |

---

## 4. How to Run This Project

1.  Clone this repository.
2.  Open the `.ipynb` file in Google Colab or a local Jupyter Notebook environment.
3.  Upload the `train.csv`, `test.csv`, and `data_description.txt` files to the same directory.
4.  Run all cells sequentially.
