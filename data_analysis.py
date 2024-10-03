#'/Users/dorijandonajmagasic/Documents/Uni modules/Individual Project/ai_job_market_insights.csv'
# Import necessary libraries
import pandas as pd
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

# Fill missing numerical values with the mean
data['Salary_USD'] = data['Salary_USD'].fillna(data['Salary_USD'].mean())

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

# Create new feature: Salary Bracket (Low, Medium, High)
salary_brackets = pd.qcut(data['Salary_USD'], q=3, labels=['Low', 'Medium', 'High'])
data['Salary_Bracket'] = salary_brackets

# Display the first few rows of the dataset with the new feature
print("\nDataset with the new Salary Bracket feature:")
print(data[['Job_Title', 'Salary_USD', 'Salary_Bracket']].head())

# Visualization: Count of Jobs by Salary Bracket and Industry
plt.figure(figsize=(12, 8))
sns.countplot(x='Industry', hue='Salary_Bracket', data=data)
plt.title('Job Count by Industry and Salary Bracket')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

# Step 4: Correlation Matrix with Encoded Categorical Variables

# Convert categorical features to numeric encodings for correlation purposes
data['Location_Code'] = data['Location'].astype('category').cat.codes
data['Industry_Code'] = data['Industry'].astype('category').cat.codes
data['Company_Size_Code'] = data['Company_Size'].cat.codes
data['AI_Adoption_Level_Code'] = data['AI_Adoption_Level'].cat.codes
data['Automation_Risk_Code'] = data['Automation_Risk'].cat.codes
data['Remote_Friendly_Code'] = data['Remote_Friendly'].cat.codes
data['Job_Growth_Projection_Code'] = data['Job_Growth_Projection'].cat.codes

# Create correlation matrix for Salary and encoded features
correlation_columns = ['Salary_USD', 'Location_Code', 'Industry_Code', 'Company_Size_Code',
                       'AI_Adoption_Level_Code', 'Automation_Risk_Code',
                       'Remote_Friendly_Code', 'Job_Growth_Projection_Code']

correlation_matrix = data[correlation_columns].corr()

print("\nCorrelation Matrix for Salary and Encoded Features:")
print(correlation_matrix)

# Plot the updated correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix between Salary and Encoded Features')
plt.show()

# End of script
