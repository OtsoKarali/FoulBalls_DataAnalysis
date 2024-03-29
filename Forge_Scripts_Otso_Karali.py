#!/usr/bin/env python
# coding: utf-8

# 

# # Importing Data Set and Cleaning Data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Data Set
FoulsData = pd.read_csv('foul-balls.csv')

# Remove 'used_zone' column
FoulsData = FoulsData.drop(columns=['used_zone'])

# Filter out all rows that have no value in "camera_zone" and "exit_velocity"
FilteredFoulsData = FoulsData.dropna(subset=['camera_zone', 'exit_velocity'])

# Function to remove outliers from a group based on the IQR
def remove_outliers(group):
    Q1 = group['exit_velocity'].quantile(0.2d5)
    Q3 = group['exit_velocity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group['exit_velocity'] >= lower_bound) & (group['exit_velocity'] <= upper_bound)]

# Apply function to data
FilteredFoulsData = FilteredFoulsData.groupby('type_of_hit').apply(remove_outliers).reset_index(drop=True)


# # Pie Chart of zone predictions

# In[2]:


# Count the occurrences of each category in predicted_zone
predicted_zone_counts = FilteredFoulsData['predicted_zone'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(predicted_zone_counts, labels=predicted_zone_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Predicted Zones')
plt.show()


# # Console Print of number of times prediction was correct

# In[3]:


# Calculate the number of times was correct
correct_predictions = (FilteredFoulsData['predicted_zone'] == FilteredFoulsData['camera_zone']).sum()

# Calculate the total number of predictions
total_predictions = len(FilteredFoulsData)

# Calculate the number of incorrect
incorrect_predictions = total_predictions - correct_predictions

# Calculate the percentage of correct predictions
percentage_correct = (correct_predictions / total_predictions) * 100

# Print Statements
print(f"Total Predictions: {total_predictions}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Percentage of Correct Predictions: {percentage_correct:.2f}%")


# # Violin plot of data

# In[4]:


# Ordered List for types of hits
order = ['Line', 'Fly', 'Pop Up', 'Ground', 'Batter hits self']

# Style
sns.set(style="whitegrid", context="talk")

# Create the violin plot
plt.figure(figsize=(16, 10))
sns.violinplot(x='type_of_hit', y='exit_velocity', data=FilteredFoulsData, scale='width', order=order)

# Title and Axis
plt.title('Distribution of Exit Velocity by Type of Hit', fontsize=0)
plt.xlabel('Type of Hit', fontsize=0)
plt.ylabel('Exit Velocity (mph)', fontsize=0)

# Customized ticks
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')

# Show Function
plt.tight_layout()
plt.show()



# # Print Base Statistics

# In[5]:


# Calculate mean
mean_exit_velocity = FilteredFoulsData['exit_velocity'].mean()

# Calculate median
median_exit_velocity = FilteredFoulsData['exit_velocity'].median()

# Calculate max 
max_exit_velocity = FilteredFoulsData['exit_velocity'].max()

# Calculate min 
min_exit_velocity = FilteredFoulsData['exit_velocity'].min()

# Calculate standard deviation
std_dev_exit_velocity = FilteredFoulsData['exit_velocity'].std()

# Print Statements
print(f"Mean exit velocity: {mean_exit_velocity:.2f} mph")
print(f"Median exit velocity: {median_exit_velocity:.2f} mph")
print(f"Highest exit velocity: {max_exit_velocity:.2f} mph")
print(f"Lowest exit velocity: {min_exit_velocity:.2f} mph")
print(f"Standard deviation of exit velocity: {std_dev_exit_velocity:.2f} mph")
print(f"Exit velocity range: {(max_exit_velocity)-(min_exit_velocity):.2f} mph")


# # Bell Curve

# In[6]:


plt.figure(figsize=(12, 6))
sns.histplot(FilteredFoulsData['exit_velocity'], kde=True, bins=20) 
plt.title('Exit Velocity Distribution')
plt.xlabel('Exit Velocity (mph)')
plt.ylabel('Density')
plt.show()


# # Bar Plot

# In[7]:


# Whitegrid graph style
sns.set_style("whitegrid")

# Get unique types of hits
hit_types = FilteredFoulsData['type_of_hit'].unique()

# Calculate the mean exit velocities
mean_exit_velocities = FilteredFoulsData.groupby('type_of_hit')['exit_velocity'].mean().sort_values()

# Create a bar plot with sorted means
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=mean_exit_velocities.index, y=mean_exit_velocities.values, palette="viridis")

# Annotate bars with the mean value
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                     textcoords = 'offset points')

# Expand Bar Sizes
plt.ylim(mean_exit_velocities.min() - 5, mean_exit_velocities.max() + 5)

# Label Axes
plt.title('Average Exit Velocity per Type of Hit')
plt.xlabel('Type of Hit')
plt.ylabel('Average Exit Velocity (mph)')
plt.xticks(rotation=45)  # Rotates the labels on the x-axis for better readability
plt.show()


# # Confidence Interval per Type of Hit

# In[8]:


import scipy.stats as stats

# Function to calculate the confidence interval for the mean exit velocity
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = data.mean()
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - margin_of_error, mean + margin_of_error

# Apply the function to each type of hit and store the results in a dictionary
ci_dict = {}
for hit_type in FilteredFoulsData['type_of_hit'].unique():
    hit_type_data = FilteredFoulsData[FilteredFoulsData['type_of_hit'] == hit_type]['exit_velocity']
    mean, lower_bound, upper_bound = mean_confidence_interval(hit_type_data)
    ci_dict[hit_type] = {
        'mean': mean,
        '95% CI Lower': lower_bound,
        '95% CI Upper': upper_bound
    }

# Convert the dictionary to a DataFrame
ci_df = pd.DataFrame(ci_dict).T
ci_df


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# One-hot encode categorical variables
categorical_features = ['type_of_hit', 'camera_zone']
one_hot = OneHotEncoder()
one_hot.fit(FilteredFoulsData[categorical_features])
transformed_X = one_hot.transform(FilteredFoulsData[categorical_features])

# Get feature names for one-hot encoded columns
feature_names = one_hot.get_feature_names_out(categorical_features)

# Prepare target variable
y = FilteredFoulsData['exit_velocity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Get model coefficients
coefficients = model.coef_

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xticks(rotation=45)
plt.title('Linear Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()


# In[10]:


# color palette
colors = sns.color_palette('tab10')

# Group the data by 'type_of_hit' and 'camera_zone' and count the occurrences
grouped_data = FilteredFoulsData.groupby(['type_of_hit', 'camera_zone']).size()

# Normalize the counts within each 'type_of_hit'
normalized_counts = grouped_data.groupby(level=0).apply(lambda x: x / float(x.sum()))

# Plot pie charts for each 'type_of_hit'
for hit_type in normalized_counts.index.levels[0]:
    fig, ax = plt.subplots()
    # Extract camera_zone labels and prepend with "Zone"
    labels = ['Zone ' + str(zone) for _, zone in normalized_counts[hit_type].index]
    normalized_counts[hit_type].plot.pie(
        labels=labels,  # Use the new labels with "Zone"
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        ax=ax
    )
    ax.set_ylabel('')  # Remove the y-axis label
    ax.set_title(f'Distribution of Camera Zones for {hit_type} Hits')
    plt.show()

