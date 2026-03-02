import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CSV_PATH   = "data/all_stocks_5yr.csv"
DATE_COL   = "date"
TICKER_COL = "name"
CLOSE_COL  = "close"

OUTPUT = "outputs"
os.makedirs(OUTPUT, exist_ok=True)

# Metrics evaluation
def evaluation_metrics(y_current, y_pred):
    absolute_error = mean_absolute_error(y_current, y_pred)
    squared_error = float(np.sqrt(mean_squared_error(y_current, y_pred)))
    r2 = r2_score(y_current, y_pred)
    return absolute_error, squared_error, r2



# Add indicators
def add_indicators_pandas(df):
 
    # convert date column into datetime type
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # sort value according to ticker and date
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    # define list of result
    results = []

    for tk, g in df.groupby(TICKER_COL, sort=False):
        g = g.sort_values(DATE_COL)
        # Exponential Moving Average (EMA)
        g["EMA_12"] = g[CLOSE_COL].ewm(span=12, adjust=False).mean()
        g["EMA_26"] = g[CLOSE_COL].ewm(span=26, adjust=False).mean()

        # Moving Average Convergence Divergence (MACD)
        g["MACD"] = g["EMA_12"] - g["EMA_26"]
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]

        # Relative Strength Index (RSI)
        period = 14
        diff_value = g[CLOSE_COL].diff()
        gain = diff_value.clip(lower=0)
        loss = (-diff_value).clip(lower=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss.replace(0, np.nan)) # avoid divide-by-zero
        g["RSI_14"] = 100 - (100 / (1 + rs))

        # Target: next day close
        g["Target_Close_NextDay"] = g[CLOSE_COL].shift(-1)

        results.append(g)

    df_out = pd.concat(results).dropna().reset_index(drop=True)
    return df_out


# Split date into 80/20 time-based per ticker

def split_80_20_date(df):
    train_list = []
    test_list = []

    for tk, g in df.groupby(TICKER_COL, sort=False):
        # date must be in order per ticker
        g = g.sort_values(DATE_COL).copy()
        # split 80/20 time-based per ticker
        split_point = int(len(g) * 0.8)
        train_list.append(g.iloc[:split_point])
        test_list.append(g.iloc[split_point:])
    
    # concat list's item into a big dataframe
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df


def main():
    
    df = pd.read_csv(CSV_PATH)

    t0 = time.perf_counter()
    # feature engineering
    df_indicators = add_indicators_pandas(df)
    t1 = time.perf_counter()
    print(f"Indicator calculation time (Pandas): {t1 - t0:.4f}s")

    # Check indicators
    print("\nResults of calculating indicators:")
    print(df_indicators[[DATE_COL, TICKER_COL, CLOSE_COL, "EMA_12", "MACD", "RSI_14", "Target_Close_NextDay"]].head())
    
    # split data
    print("\nSplitting 80/20 per ticker...")
    train_df, test_df = split_80_20_date(df_indicators)
    print("Train rows:", len(train_df), "| Test rows:", len(test_df))

    # Featuring 
    feature_cols = [CLOSE_COL, "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Hist", "RSI_14"]
    target_col = "Target_Close_NextDay"

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Train models
    print("\nTraining Model 1: Linear Regression")
    lr = LinearRegression()
    t0 = time.perf_counter()
    lr.fit(X_train, y_train)
    lr_fit = time.perf_counter() - t0
    lr_pred = lr.predict(X_test)
    lr_mae, lr_rmse, lr_r2 = evaluation_metrics(y_test, lr_pred)

    print("Training Model 2: Random Forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,max_depth=10, min_samples_leaf=10) # agruments added to avoid crashing local machine
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_fit = time.perf_counter() - t0
    rf_pred = rf.predict(X_test)
    rf_mae, rf_rmse, rf_r2 = evaluation_metrics(y_test, rf_pred)

    # save outputs for dashboard
    preds = test_df[[DATE_COL, TICKER_COL, CLOSE_COL, target_col] + feature_cols].copy()
    preds = preds.rename(columns={target_col: "Actual_NextDayClose"})
    preds["Pred_LR"] = lr_pred
    preds["Pred_RF"] = rf_pred
        
        # save prediction results
    preds_path = os.path.join(OUTPUT, "predictions.csv")
    preds.to_csv(preds_path, index=False)
        
        # save result of metrics of each model after training
    overall = pd.DataFrame([
        {"Model": "LinearRegression", "MAE": lr_mae, "RMSE": lr_rmse, "R2": lr_r2, "FitTimeSec": lr_fit},
        {"Model": "RandomForest",     "MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2, "FitTimeSec": rf_fit},
    ])
    overall_path = os.path.join(OUTPUT, "overall_metrics.csv")
    overall.to_csv(overall_path, index=False)
    rows = []
    for tk, g in preds.groupby(TICKER_COL, sort=False):
        y = g["Actual_NextDayClose"].values
        mae_lr, rmse_lr, r2_lr = evaluation_metrics(y, g["Pred_LR"].values)
        mae_rf, rmse_rf, r2_rf = evaluation_metrics(y, g["Pred_RF"].values)
        rows.append({
            TICKER_COL: tk,
            "n_test": len(g),
            "LR_MAE": mae_lr, "LR_RMSE": rmse_lr, "LR_R2": r2_lr,
            "RF_MAE": mae_rf, "RF_RMSE": rmse_rf, "RF_R2": r2_rf,
        })
        
        # save metrics by tickers
    by_ticker = pd.DataFrame(rows).sort_values("RF_MAE")
    by_ticker_path = os.path.join(OUTPUT, "metrics_by_ticker.csv")
    by_ticker.to_csv(by_ticker_path, index=False)

    print("\nSaved outputs:")
    print(" -", preds_path)
    print(" -", overall_path)
    print(" -", by_ticker_path)
    print("\nDONE")


if __name__ == "__main__":
    main()