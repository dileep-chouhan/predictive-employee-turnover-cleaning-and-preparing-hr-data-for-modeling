import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic HR data
num_employees = 500
data = {
    'EmployeeID': range(1, num_employees + 1),
    'Age': np.random.randint(20, 65, size=num_employees),
    'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], size=num_employees),
    'YearsOfExperience': np.random.randint(0, 30, size=num_employees),
    'Salary': np.random.randint(40000, 150000, size=num_employees),
    'Attrition': np.random.choice(['Yes', 'No'], size=num_employees, p=[0.2, 0.8]), # 20% attrition rate
    'OverTime': np.random.choice(['Yes', 'No'], size=num_employees, p=[0.3, 0.7]),
    'JobSatisfaction': np.random.randint(1, 5, size=num_employees), # 1-5 scale
    'WorkLifeBalance': np.random.randint(1, 5, size=num_employees) # 1-5 scale
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
#Handle missing values (although we don't have any in this synthetic data, it's good practice)
print("Number of missing values before handling:\n", df.isnull().sum())
df.fillna(df.mean(), inplace=True) #Simple imputation for numerical features.  More sophisticated methods could be used.
print("\nNumber of missing values after handling:\n", df.isnull().sum())
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Department', 'Attrition', 'OverTime'], drop_first=True)
# --- 3. Analysis ---
# Explore attrition rates by department
attrition_by_dept = df.groupby('Department_Marketing')['Attrition_Yes'].mean()
print("\nAttrition rates by department:\n", attrition_by_dept)
# --- 4. Visualization ---
# Visualize attrition rate by age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition_Yes', y='Age', data=df)
plt.title('Age Distribution by Attrition')
plt.xlabel('Attrition')
plt.ylabel('Age')
plt.tight_layout()
output_filename = 'age_vs_attrition.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Visualize attrition rate by years of experience
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition_Yes', y='YearsOfExperience', data=df)
plt.title('Years of Experience Distribution by Attrition')
plt.xlabel('Attrition')
plt.ylabel('Years of Experience')
plt.tight_layout()
output_filename = 'experience_vs_attrition.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis and modeling would follow here (e.g., logistic regression, etc.) but is omitted for brevity.