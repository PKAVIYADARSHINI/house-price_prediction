



Housing Price Prediction:

## Project Overview
This project predicts housing prices based on various features such as location, number of rooms, median income, and proximity to the ocean. The model was built using linear regression, trained on a dataset of housing data, and evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

steps involved:
Here’s a breakdown of the code, which you can use to explain your project during the interview:

### 1. **Importing Libraries**
   - **Purpose**: Import necessary Python libraries for data manipulation, machine learning, and model evaluation.
   - **Explanation**:
     - `pandas`: For data manipulation and analysis.
     - `train_test_split` from `sklearn.model_selection`: To split the dataset into training and testing sets.
     - `StandardScaler` from `sklearn.preprocessing`: To standardize the features.
     - `LinearRegression` from `sklearn.linear_model`: To implement the linear regression model.
     - `mean_absolute_error`, `mean_squared_error`, `r2_score` from `sklearn.metrics`: To evaluate the model's performance.
     - `SimpleImputer`: (not used in the final code, but can be used for handling missing values).

### 2. **Loading the Dataset**
   - **Purpose**: Load the housing dataset from a CSV file.
   - **Explanation**:
     - The dataset is read from the specified file path using `pandas.read_csv`.
     - The file path provided should be accurate and point to the dataset location.

### 3. **Handling Missing Values**
   - **Purpose**: Clean the dataset by removing rows with missing values.
   - **Explanation**:
     - `dropna(inplace=True)`: Removes any rows with NaN values from the dataset.
     - This ensures that the model doesn’t encounter any missing data during training.

### 4. **Checking for Missing Values**
   - **Purpose**: Verify that all missing values have been removed.
   - **Explanation**:
     - `isnull().sum()`: This line checks each column for any remaining missing values and prints the count.

### 5. **Encoding Categorical Variables**
   - **Purpose**: Convert categorical variables into a format that can be used in the model.
   - **Explanation**:
     - Identify categorical columns and convert them to category dtype.
     - Use `pd.get_dummies` to create dummy/indicator variables, which converts categorical variables into a binary matrix.

### 6. **Defining the Target Variable**
   - **Purpose**: Specify the column that contains the target variable (the variable to be predicted).
   - **Explanation**:
     - `target_column = 'median_house_value'`: Identifies the target variable.
     - The code checks if the target column is present in the dataset.

### 7. **Splitting the Data**
   - **Purpose**: Divide the data into training and testing sets.
   - **Explanation**:
     - `train_test_split`: 80% of the data is used for training, and 20% for testing, ensuring that the model is evaluated on unseen data.
     - `random_state=42` ensures that the split is reproducible.

### 8. **Feature Scaling**
   - **Purpose**: Standardize the features to have a mean of 0 and a standard deviation of 1.
   - **Explanation**:
     - `StandardScaler`: This is important for linear models as it ensures that all features contribute equally to the model’s predictions.

### 9. **Building and Training the Model**
   - **Purpose**: Train a linear regression model on the training data.
   - **Explanation**:
     - `model.fit(X_train, y_train)`: Fits the model to the training data, learning the relationship between the features and the target variable.

### 10. **Evaluating the Model**
   - **Purpose**: Measure the model's performance using different metrics.
   - **Explanation**:
     - `mean_absolute_error`, `mean_squared_error`, `r2_score`: These metrics assess how well the model predicts the target variable on the test set.
     - MAE and MSE indicate the average magnitude of prediction errors, while R² indicates how well the model explains the variance in the data.

### 11. **Making Predictions**
   - **Purpose**: Use the trained model to predict housing prices for new data.
   - **Explanation**:
     - Example data is provided, which must be preprocessed similarly to the training data.
     - The new data is transformed using the same scaler and structure as the training data.
     - `model.predict`: The model predicts the housing price for the new data, showcasing the practical application of the model.
    
my colab link:
     https://colab.research.google.com/drive/14uM6vYnwbncnWnUD8kwA-pLWaSSdWrGD?usp=sharing


## Dataset
The dataset contains various features related to housing in different locations. It includes both numerical and categorical variables.

- **Source**: [Mention the source of the dataset if publicly available]
- **Features**:
  - `longitude`: Longitude coordinate of the house.
  - `latitude`: Latitude coordinate of the house.
  - `housing_median_age`: Median age of the house.
  - `total_rooms`: Total number of rooms in the house.
  - `total_bedrooms`: Total number of bedrooms in the house.
  - `population`: Population of the area.
  - `households`: Number of households in the area.
  - `median_income`: Median income of the household.
  - `ocean_proximity`: Proximity of the house to the ocean.

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/housing-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd housing-price-prediction
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Load and Preprocess the Data**:
   - Load the dataset from `housing.csv`.
   - Handle missing values by dropping rows with NaN values.
   - Encode categorical variables.
   - Split the data into training and testing sets.
   - Standardize the features.

2. **Train the Model**:
   - Train a linear regression model on the training data.

3. **Evaluate the Model**:
   - Evaluate the model’s performance using MAE, MSE, and R-squared.

4. **Make Predictions**:
   - Use the trained model to predict housing prices for new data.

## Model Evaluation
After training, the model was evaluated on the test data, resulting in the following metrics:

- **Mean Absolute Error (MAE)**: [Add your MAE value here]
- **Mean Squared Error (MSE)**: [Add your MSE value here]
- **R-squared (R²)**: [Add your R² value here]

These metrics indicate how well the model is performing on unseen data.

## Example Prediction
An example prediction using the model for a new data point:

- **Input Data**:
  - Longitude: -122.23
  - Latitude: 37.88
  - Housing Median Age: 41.0
  - Total Rooms: 880.0
  - Total Bedrooms: 129.0
  - Population: 322.0
  - Households: 126.0
  - Median Income: 8.3252
  - Ocean Proximity: Near Ocean

- **Predicted Price**: [Add your predicted price here]

## Contributing
Contributions are welcome! If you find any issues or have suggestions, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

#
- **Requirements**: If you have specific libraries used in the project, include a `requirements.txt` file with them listed (e.g., `pandas`, `scikit-learn`).


code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = '/content/housing.csv'  # Path to the uploaded file
with open(file_path, 'r') as f:
    housing_data = pd.read_csv(f)

# Drop rows with any NaN values
housing_data.dropna(inplace=True)

# Verify that there are no missing values
print(housing_data.isnull().sum())

# Convert categorical columns to category dtype
categorical_cols = housing_data.select_dtypes(include=['object']).columns
housing_data[categorical_cols] = housing_data[categorical_cols].astype('category')

# Encode categorical variables
housing_data = pd.get_dummies(housing_data, drop_first=True)

# Update the target column name based on the actual dataset
target_column = 'median_house_value'  # Replace with the actual target column name if different

# Verify the target column is in the dataset
if target_column not in housing_data.columns:
    raise KeyError(f"'{target_column}' column not found in the dataset. Available columns: {housing_data.columns}")

# Separate features and target variable
X = housing_data.drop(target_column, axis=1)
y = housing_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nModel Evaluation:')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Make Predictions
# Example of making predictions with new data
new_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity_INLAND': [0],
    'ocean_proximity_ISLAND': [0],
    'ocean_proximity_NEAR BAY': [0],
    'ocean_proximity_NEAR OCEAN': [1]
})

# Ensure the new data has the same structure as the training data
new_data = pd.get_dummies(new_data, drop_first=True)
missing_cols = set(housing_data.columns) - set(new_data.columns)
for c in missing_cols:
    new_data[c] = 0
new_data = new_data[X.columns]

# Preprocess new data
new_data = scaler.transform(new_data)

# Make prediction
new_prediction = model.predict(new_data)
print(f'\nPredicted Price: {new_prediction[0]}')



excepted output:
longitude             0
latitude              0
housing_median_age    0
total_rooms           0
total_bedrooms        0
population            0
households            0
median_income         0
median_house_value    0
ocean_proximity       0
dtype: int64

Model Evaluation:
Mean Absolute Error: 50413.433308100364
Mean Squared Error: 4802173538.60416
R-squared: 0.6488402154431994

Predicted Price: 418238.60888893466


