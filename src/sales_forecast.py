import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/sales.csv")

# Handle missing values
data = data.dropna()

# Convert date column
data['date'] = pd.to_datetime(data['date'])
data['date_ordinal'] = data['date'].map(pd.Timestamp.toordinal)

# Features and target
X = data[['date_ordinal']]
y = data['revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Save predictions
predictions = pd.DataFrame({
    "Actual Revenue": y_test,
    "Predicted Revenue": y_pred
})

predictions.to_csv("outputs/predictions.csv", index=False)

# Plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Sales")
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()
