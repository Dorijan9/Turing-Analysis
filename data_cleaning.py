# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/Users/dorijandonajmagasic/Documents/Uni modules/Individual Project/ai_job_market_insights.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 1: Data Cleaning
# Convert relevant columns to categorical data types
data['Company_Size'] = data['Company_Size'].astype('category')
data['Remote_Friendly'] = data['Remote_Friendly'].astype('category')
data['Job_Growth_Projection'] = data['Job_Growth_Projection'].astype('category')
data['AI_Adoption_Level'] = data['AI_Adoption_Level'].astype('category')
data['Automation_Risk'] = data['Automation_Risk'].astype('category')

# Display the first few rows of the cleaned dataset
print("First few rows of the cleaned dataset:")
print(data.head())

# Display the updated data types
print("\nUpdated data types:")
print(data.dtypes)

# Step 2: Statistical Summary of Numeric Columns
summary_stats = data.describe()

print("\nStatistical summary of Salary_USD:")
print(summary_stats)

# Step 3: Visualization

# Visualization 1: Distribution of Salaries
plt.figure(figsize=(10, 6))
sns.histplot(data['Salary_USD'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Salaries in USD')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualization 2: Boxplot of Salaries by Company Size
plt.figure(figsize=(10, 6))
sns.boxplot(x='Company_Size', y='Salary_USD', data=data)
plt.title('Salaries by Company Size')
plt.xlabel('Company Size')
plt.ylabel('Salary (USD)')
plt.grid(True)
plt.show()

# Visualization 3: Bar Plot of AI Adoption Level by Remote Friendliness
plt.figure(figsize=(10, 6))
sns.countplot(x='AI_Adoption_Level', hue='Remote_Friendly', data=data)
plt.title('AI Adoption Level vs Remote Friendliness')
plt.xlabel('AI Adoption Level')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Visualization 4: Salaries by Industry
plt.figure(figsize=(12, 8))
sns.barplot(x='Industry', y='Salary_USD', data=data, estimator=lambda x: sum(x) / len(x))
plt.title('Average Salary by Industry')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Industry')
plt.ylabel('Average Salary (USD)')
plt.grid(True)
plt.show()

# End of script
