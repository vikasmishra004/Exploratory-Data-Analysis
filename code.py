import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Vikas\OneDrive\Desktop\MOR\Projects\Exploratory Data Analysis\train.csv"
data = pd.read_csv(file_path)

# Data Cleaning and Feature Engineering
data['Age'] = data['Age'].fillna(data['Age'].median())  # Avoid inplace=True
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # Avoid inplace=True
data['Fare'] = data['Fare'].fillna(data['Fare'].median())  # Avoid inplace=True

# Create Family Size Feature
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Create AgeGroup Feature
bins = [0, 12, 20, 40, 60, np.inf]
labels = ['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior']
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels)

# 1. Survival Rate by Gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=data, palette='husl', edgecolor='black', linewidth=2, errorbar=None)
plt.title('Survival Rate by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 2. Survival Rate by Passenger Class
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=data, palette='cubehelix', edgecolor='black', linewidth=2, errorbar=None)
plt.title('Survival Rate by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 3. Survival Rate by Age Group
plt.figure(figsize=(8, 6))
sns.barplot(x='AgeGroup', y='Survived', data=data, palette='viridis', edgecolor='black', linewidth=2, errorbar=None)
plt.title('Survival Rate by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 4. Survival Rate by Embarkation Port
plt.figure(figsize=(8, 6))
sns.barplot(x='Embarked', y='Survived', data=data, palette='coolwarm', edgecolor='black', linewidth=2, errorbar=None)
plt.title('Survival Rate by Embarkation Port', fontsize=16)
plt.xlabel('Embarkation Port', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 5. Survival Rate by Family Size
plt.figure(figsize=(10, 6))
sns.countplot(x='FamilySize', hue='Survived', data=data, palette='Set1', edgecolor='black', linewidth=2)
plt.title('Survival Rate by Family Size', fontsize=16)
plt.xlabel('Family Size', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Survived', labels=['No', 'Yes'], fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 6. Fare Distribution by Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Fare', hue='Survived', multiple='stack', kde=False, palette='coolwarm', binwidth=5, edgecolor='black', linewidth=0.5)
plt.title('Fare Distribution by Survival', fontsize=16)
plt.xlabel('Fare', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
