# Turing-Analysis
AI Job Market Insights Analysis

Project Overview

This project involves analyzing data from the AI job market, specifically focusing on factors like salary distribution, industry trends, and remote work prevalence. The dataset used includes information about job titles, salaries, industries, company sizes, AI adoption levels, and more. The goal is to extract meaningful insights about the AI job market, such as salary trends across industries, the influence of company size on salaries, and the prevalence of remote-friendly job opportunities.

Steps of Analysis

1. Data Cleaning
Before performing any analysis, we cleaned the dataset to ensure its usability. The following steps were carried out:

Missing Values: We checked for missing values in the dataset. For numerical columns (e.g., Salary_USD), missing values were filled with the mean using NumPy (np.nanmean()). For categorical columns (e.g., Industry, Location), missing values were filled with the mode.
Data Type Conversion: Categorical columns like Company_Size, Remote_Friendly, and AI_Adoption_Level were converted to the appropriate categorical data type for easier analysis and visualization.
2. Data Exploration
To gain a better understanding of the dataset, we performed some exploratory data analysis:

Statistical Summary: A statistical summary of the Salary_USD column was generated to understand the basic distribution of salaries.
Visualizations: Various plots were created to visualize the data:
A histogram showing the distribution of salaries across the dataset.
A boxplot illustrating how salaries vary by company size.
A bar plot exploring the relationship between AI adoption levels and the prevalence of remote-friendly jobs.
A bar plot showing the average salary by industry, helping us understand how different industries compensate AI professionals.
3. Feature Analysis and Engineering
Next, we delved deeper into the relationships between various features in the dataset:

Correlation Matrix: We calculated a correlation matrix for numerical columns to identify any significant relationships between variables, such as salary and other numerical features.
Salary Bracket Creation: We created a new feature called Salary_Bracket by dividing the salary data into three brackets: Low, Medium, and High. This categorization helps us analyze job distributions and trends across different salary ranges.
Salary Normalization: To further enhance the analysis, we added a normalized version of the salary column using Min-Max scaling with NumPy. This allows us to compare salaries on a scale from 0 to 1.
4. Visualizing Feature Relationships
After creating new features, we visualized their relationships:

Job Count by Industry and Salary Bracket: A count plot was created to show the distribution of jobs in each industry, segmented by the Salary_Bracket. This helps identify which industries offer higher-paying jobs more frequently.
5. Interpretation of Results
The analysis highlighted several key findings:

Salary Distribution: Salaries in the AI job market vary significantly, with certain industries like tech and finance offering higher average salaries.
Company Size: Larger companies tend to offer higher salaries, as evidenced by the boxplot analysis.
Remote-Friendly Jobs: The relationship between AI adoption levels and remote work showed interesting patterns, with more innovative companies appearing to embrace remote work more frequently.
6. Future Work
Moving forward, additional analysis can be conducted by:

Expanding on the feature engineering to create new meaningful variables (e.g., years of experience, education levels).
Applying machine learning models like K-Means clustering to further analyze the relationship between salary, location, and other categorical features.
Code Documentation

Each part of the code is thoroughly commented to maintain readability and clarity. Key sections of the code contain explanations for the steps performed, such as filling missing values, creating visualizations, and generating new features. We’ve used both Pandas and NumPy extensively for data handling and manipulation, with Seaborn and Matplotlib employed for visualizations.
