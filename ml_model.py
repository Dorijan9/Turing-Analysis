# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/Users/dorijandonajmagasic/Documents/Uni modules/Individual Project/ai_job_market_insights.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 1: Data Cleaning

# Fill missing numerical values with the mean using NumPy
data['Salary_USD'] = data['Salary_USD'].fillna(np.nanmean(data['Salary_USD']))

# Fill missing categorical values with the mode
for column in ['Job_Title', 'Industry', 'Company_Size', 'Location', 'AI_Adoption_Level',
               'Automation_Risk', 'Required_Skills', 'Remote_Friendly', 'Job_Growth_Projection']:
    mode_value = data[column].mode()[0]
    data[column] = data[column].fillna(mode_value)

# Convert relevant columns to categorical data types
data['Company_Size'] = pd.Categorical(data['Company_Size'])
data['Remote_Friendly'] = pd.Categorical(data['Remote_Friendly'])
data['Job_Growth_Projection'] = pd.Categorical(data['Job_Growth_Projection'])
data['AI_Adoption_Level'] = pd.Categorical(data['AI_Adoption_Level'])
data['Automation_Risk'] = pd.Categorical(data['Automation_Risk'])

# Step 2: Feature Engineering

# Encode categorical features into numerical representations
data['Location_Code'] = data['Location'].astype('category').cat.codes
data['Industry_Code'] = data['Industry'].astype('category').cat.codes
data['Company_Size_Code'] = data['Company_Size'].cat.codes
data['AI_Adoption_Level_Code'] = data['AI_Adoption_Level'].cat.codes
data['Automation_Risk_Code'] = data['Automation_Risk'].cat.codes
data['Remote_Friendly_Code'] = data['Remote_Friendly'].cat.codes
data['Job_Growth_Projection_Code'] = data['Job_Growth_Projection'].cat.codes

# Create target feature: Salary Bracket (Low, Medium, High)
salary_brackets = pd.qcut(data['Salary_USD'], q=3, labels=['Low', 'Medium', 'High'])
data['Salary_Bracket'] = salary_brackets

# Step 3: Prepare data for modeling

# Select the features (X) and the target variable (y)
X = data[['Location_Code', 'Industry_Code', 'Company_Size_Code', 'AI_Adoption_Level_Code',
          'Automation_Risk_Code', 'Remote_Friendly_Code', 'Job_Growth_Projection_Code']]
y = data['Salary_Bracket']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the model

# Initialize the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy, classification report, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# End of script
