import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

st.title('Data Analysis: Bike Sharing Dataset')
st.header("Attanabil Rabbani_50421230")
st.write("________________________________________________")
st.title('🚀 Data Understanding 🚀')

st.header("➝ Data Overview")
day_original = pd.read_csv("day.csv")
day_cleaned = pd.read_csv("day_cleaned.csv")

day_original.set_index("instant",inplace=True)
day_cleaned.set_index("instant",inplace=True)

st.dataframe(day_original)

st.header("➝ Statistics")
st.dataframe(day_original.describe())

st.write("________________________________________________")
st.title('🔎 Data Preparation 🔎')
st.header("➝ Outliers Check")
df_outlier = day_original.select_dtypes(exclude=['object', 'datetime64'])

for column in df_outlier:
    st.write(f"Boxplot: {column}")
    plt.figure(figsize=(22, 2))
    sns.boxplot(data=df_outlier, x=column)

    st.pyplot(plt)
    
    plt.clf()

st.write("Outlier found in columns: hum, windspeed, and casual.")

st.header("➝ Removing Outliers")
st.subheader("For hum (humidity) column: Trimming")
st.write("Outliers in humidity column only represents 1.78% of overall data.")
st.write("Before Trimming:")

plt.figure(figsize=(22, 2))
sns.boxplot(data=df_outlier, x=day_original["hum"])
st.pyplot(plt)
plt.clf()

st.write("After Trimming: ")
plt.figure(figsize=(22, 2))
sns.boxplot(data=df_outlier, x=day_cleaned["hum"])
st.pyplot(plt)
plt.clf()

st.write("\t------------------------------")

st.subheader("For Windspeed and Casual column: Winsorizing")
st.subheader("Windspeed Column")

st.write("Before Winsorizing:")
plt.figure(figsize=(22, 2))
sns.boxplot(data=df_outlier, x=day_original["windspeed"])
st.pyplot(plt)
plt.clf()

st.write("After Winsorizing:")
plt.figure(figsize=(22, 2))
sns.boxplot(data=df_outlier, x=day_cleaned["windspeed"])
st.pyplot(plt)
plt.clf()

st.subheader("Casual Column")

st.write("Before Winsorizing:")
plt.figure(figsize=(22, 2))
sns.boxplot(data=df_outlier, x=day_original["casual"])
st.pyplot(plt)
plt.clf()

st.write("After Winsorizing:")
plt.figure(figsize=(22, 2))
sns.boxplot(data=df_outlier, x=day_cleaned["casual"])
st.pyplot(plt)
plt.clf()
st.write("\t------------------------------")

st.header("➝ Visualizations")
st.subheader("1. Seasonal Rental")

plt.figure(figsize=(8, 6))
sns.barplot(x=['Spring', 'Summer', 'Fall', 'Winter'], y='cnt', data=day_cleaned.groupby('season')['cnt'].mean().reset_index(), errorbar=None, palette='coolwarm')

plt.title('Bike Rentals per Season', fontsize=16)
plt.xlabel("Season", fontsize=12)
plt.ylabel('Average Total Rentals (cnt)', fontsize=12)
st.pyplot(plt)
plt.clf()

st.subheader("2. Yearly Comparison of Bike Rentals")
plt.figure(figsize=(8, 6))
sns.barplot(x=["2011", "2012"] , y='cnt', data=day_cleaned.groupby('yr')['cnt'].sum().reset_index(), palette='Set2')

plt.title('Yearly Comparison of Bike Rentals', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Bike Rentals', fontsize=12)

plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.2f}M".format(x / 1e6)))
st.pyplot(plt)
plt.clf()

st.subheader("3. Average Daily Rentals")
plt.figure(figsize=(8, 6))
sns.barplot(x=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], y='cnt', data=day_cleaned.groupby('weekday')['cnt'].mean().sort_index().reset_index(), errorbar=None, palette='crest')

plt.title('Average Daily Rental', fontsize=16)
plt.ylabel('Average Rental', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

st.write("________________________________________________")
st.title('🔧 Modeling 🔧')
X = day_cleaned[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
y = day_cleaned['cnt']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

st.header("➝ Linear Regression")
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

lr_predict = linear_regression.predict(X_test)
lr_predict_df = pd.DataFrame({"Actual Value":y_test, "Predicted Value":lr_predict})

st.dataframe(lr_predict_df)

mae_lr = mean_absolute_error(y_test, lr_predict)
mse_lr = mean_squared_error(y_test, lr_predict)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, lr_predict)

metrics_heatmap = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae_lr, rmse_lr, r2_lr]
}).set_index('Metric')

plt.figure(figsize=(8, 4))
sns.heatmap(metrics_heatmap, annot=True, cmap='flare', cbar=True, fmt=".2f")
plt.title('Linear Regression Evaluation Metrics Heatmap')
st.pyplot(plt)
plt.clf()
st.write("\t------------------------------")

st.header("➝ K-Nearest-Neighbor")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_predict_df = pd.DataFrame({"Actual Value":y_test, "Predicted Value":knn_pred})

st.dataframe(knn_predict_df)

mae_knn = mean_absolute_error(y_test, knn_pred)
mse_knn = mean_squared_error(y_test, knn_pred)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, knn_pred)

knn_metrics_heatmap = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae_knn, rmse_knn, r2_knn]
}).set_index('Metric')

plt.figure(figsize=(8, 4))
sns.heatmap(knn_metrics_heatmap, annot=True, cmap='flare', cbar=True, fmt=".2f")
plt.title('KNN Evaluation Metrics Heatmap')
st.pyplot(plt)
plt.clf()
st.write("\t------------------------------")

st.header("➝ Decision Tree")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_predict_df = pd.DataFrame({"Actual Value":y_test, "Predicted Value":dt_pred})

st.dataframe(dt_predict_df)

mae_dt = mean_absolute_error(y_test, dt_pred)
mse_dt = mean_squared_error(y_test, dt_pred)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, dt_pred)

dt_metrics_heatmap = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae_dt, rmse_dt, r2_dt]
}).set_index('Metric')

plt.figure(figsize=(8, 4))
sns.heatmap(dt_metrics_heatmap, annot=True, cmap='flare', cbar=True, fmt=".2f")
plt.title('Decision Tree Evaluation Metrics Heatmap')
st.pyplot(plt)
plt.clf()
st.write("________________________________________________")

st.title(' 🤖 Model Evaluation 🤖')
MODEL =["Linear Regression", "KNN", "Decision Tree"]
MAE = [mae_lr, mae_knn, mae_dt]
RMSE = [rmse_lr, rmse_knn, rmse_dt]
R2 = [r2_lr, r2_knn, r2_dt]

metrics_combined = pd.DataFrame({"Model":MODEL, "MAE":MAE,"RMSE":RMSE,"R²":R2})
metrics_combined.set_index('Model', inplace=True)

st.header("➝ MAE And RMSE")
plt.figure(figsize=(12, 8))
bar_width = 0.4
x = np.arange(len(metrics_combined))

plt.bar(x - bar_width/2, metrics_combined['MAE'], width=bar_width, label='MAE', color="#BFDEF3")
plt.bar(x + bar_width/2, metrics_combined['RMSE'], width=bar_width, label='RMSE', color="#FFC9B4")

plt.title('Model Comparison: MAE and RMSE', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.xticks(x, metrics_combined.index)
plt.legend(loc='upper left')
plt.tight_layout()
st.pyplot(plt)
plt.clf()

st.header("➝ R Squared")
plt.figure(figsize=(8, 5))

plt.bar(["Linear Regression", "KNN", "Decision Tree"], metrics_combined["R²"], width=0.4, color="#B9E9E9")

plt.title('Model Comparison: R Squared', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Scores', fontsize=12)

st.pyplot(plt)
plt.clf()
st.write("\t------------------------------")
st.header("➝ Conclusion")
st.write("➺ Best Model: Linear Regression is the best-performing model among the three based on these metrics. It shows the lowest error and highest explanatory power.")
st.write("➺ KNN performs poorly compared to the other two models, with the highest errors and the lowest R² value.")
st.write("➺ While Decision Tree shows a reasonable performance, it still falls short of Linear Regression.")
st.write("In summary, for this particular dataset the best modeling approach is linear regression.")