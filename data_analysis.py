#'/Users/dorijandonajmagasic/Documents/Uni modules/Individual Project/ai_job_market_insights.csv'
# Import necessary libraries
import pandas as pd # Importing Pandas
import numpy as np  # Importing NumPy
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/Users/dorijandonajmagasic/Documents/Uni modules/Individual Project/ai_job_market_insights.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 1: Data Cleaning

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Handle missing values using NumPy
# Fill missing numerical values with the mean using NumPy
data['Salary_USD'] = data['Salary_USD'].fillna(np.nanmean(data['Salary_USD']))

# Fill missing categorical values with the mode using NumPy
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

# Step 2: Data Exploration

# Statistical Summary of Numeric Columns
summary_stats = data.describe()

print("\nStatistical summary of Salary_USD:")
print(summary_stats)

# Visualization: Distribution of Salaries
plt.figure(figsize=(10, 6))
sns.histplot(data['Salary_USD'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Salaries in USD')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Boxplot of Salaries by Company Size
plt.figure(figsize=(10, 6))
sns.boxplot(x='Company_Size', y='Salary_USD', data=data)
plt.title('Salaries by Company Size')
plt.xlabel('Company Size')
plt.ylabel('Salary (USD)')
plt.grid(True)
plt.show()

# Bar Plot of AI Adoption Level by Remote Friendliness
plt.figure(figsize=(10, 6))
sns.countplot(x='AI_Adoption_Level', hue='Remote_Friendly', data=data)
plt.title('AI Adoption Level vs Remote Friendliness')
plt.xlabel('AI Adoption Level')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Salaries by Industry
plt.figure(figsize=(12, 8))
sns.barplot(x='Industry', y='Salary_USD', data=data, estimator=lambda x: sum(x) / len(x))
plt.title('Average Salary by Industry')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Industry')
plt.ylabel('Average Salary (USD)')
plt.grid(True)
plt.show()

# Step 3: Feature Analysis/Engineering

# Correlation matrix for numeric columns only
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

print("\nCorrelation matrix for numeric columns:")
print(correlation_matrix)

# Create new feature: Salary Bracket (Low, Medium, High) using NumPy
salary_brackets = pd.qcut(data['Salary_USD'], q=3, labels=['Low', 'Medium', 'High'])
data['Salary_Bracket'] = salary_brackets

# Feature: Add Salary normalized feature using NumPy
# Normalizing the salary using Min-Max Scaling
data['Salary_Normalized'] = (data['Salary_USD'] - np.min(data['Salary_USD'])) / (np.max(data['Salary_USD']) - np.min(data['Salary_USD']))

# Display the first few rows of the dataset with the new feature
print("\nDataset with the new Salary Bracket and Salary Normalized feature:")
print(data[['Job_Title', 'Salary_USD', 'Salary_Bracket', 'Salary_Normalized']].head())

# Visualization: Count of Jobs by Salary Bracket and Industry
plt.figure(figsize=(12, 8))
sns.countplot(x='Industry', hue='Salary_Bracket', data=data)
plt.title('Job Count by Industry and Salary Bracket')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

# End of script
