# PJM Hourly Energy Load Analysis and Prediction

Big Data assignment: exploratory data analysis and machine learning on the PJM hourly electricity load dataset.

## Dataset

`PJM_Load_hourly.csv` — 32,896 hourly records of electricity consumption from the PJM interconnection grid (USA).

| Column | Description |
|---|---|
| `Datetime` | Hourly timestamp (1998–2001) |
| `PJM_Load_MW` | Electricity load in megawatts |

No missing values, no duplicate rows.

## Project Steps

1. **Data loading and inspection** — shape, dtypes, summary statistics
2. **Data quality checks** — null values and duplicates
3. **Datetime processing** — parse timestamps, sort chronologically
4. **Feature engineering** — extract Year, Month, Day, Hour, Weekday
5. **Exploratory Data Analysis**
   - Distribution histogram and box plot of load
   - Full time series plot
   - Average load by month, hour, year, and weekday
   - Correlation heatmap
6. **Modelling** — 80/20 train-test split
   - Linear Regression (baseline)
   - Random Forest Regressor (100 trees)
7. **Evaluation** — MAE, RMSE, R², actual vs predicted plots, error distribution, feature importance

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3839.39 | 4958.42 | 0.289 |
| Random Forest | 920.43 | 1508.17 | 0.934 |

**Random Forest clearly wins.** Linear Regression explains only about 29% of the variance because the relationship between time features and energy load is not linear — load rises in both summer and winter, and follows a curved daily pattern. Random Forest captures these non-linear patterns and reaches an R² of 0.934.

## Key Findings

- **Daily pattern:** Load is lowest in the early morning hours and peaks in the late afternoon and evening.
- **Seasonal pattern:** Consumption is highest in summer (air conditioning) and winter (heating), lowest in spring and autumn.
- **Weekly pattern:** Weekdays show higher load than weekends due to commercial and industrial activity.
- **Feature importance:** Hour is the strongest predictor of load, followed by Month.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Big_data.ipynb
```

Keep `PJM_Load_hourly.csv` in the same folder as the notebook.

## Files

```
├── Big_data.ipynb          # Main analysis notebook
├── PJM_Load_hourly.csv     # Dataset
└── README.md               # This file
```

## Tools Used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

## Possible Improvements

- Add lag features (load from previous hours/days) for time series forecasting
- Use a chronological train-test split instead of a random split
- Try Gradient Boosting (XGBoost / LightGBM)
- Tune Random Forest hyperparameters with GridSearchCV
