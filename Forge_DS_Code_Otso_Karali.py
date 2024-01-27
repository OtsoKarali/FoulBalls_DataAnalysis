# # Importing Data Set and Cleaning Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Data Set
FoulsData = pd.read_csv('foul-balls.csv')

# Filter out all rows that have no value in "camera_zone" and "exit_velocity"
FilteredFoulsData = FoulsData.dropna(subset=['camera_zone', 'exit_velocity'])

# Function to remove outliers from a group based on the IQR
def remove_outliers(group):
    Q1 = group['exit_velocity'].quantile(0.25)
    Q3 = group['exit_velocity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group['exit_velocity'] >= lower_bound) & (group['exit_velocity'] <= upper_bound)]

# Apply the function to remove outliers from data
FilteredFoulsData = FilteredFoulsData.groupby('type_of_hit').apply(remove_outliers).reset_index(drop=True)


# # Pie Chart of zone predictions

# Count the occurrences of each category in predicted_zone
predicted_zone_counts = FilteredFoulsData['predicted_zone'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(predicted_zone_counts, labels=predicted_zone_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Predicted Zones')
plt.show()


# # Console Print of number of times prediction was correct

# Calculate the number of times the prediction was correct
correct_predictions = (FilteredFoulsData['predicted_zone'] == FilteredFoulsData['camera_zone']).sum()

# Calculate the total number of predictions
total_predictions = len(FilteredFoulsData)

# Calculate the number of incorrect predictions
incorrect_predictions = total_predictions - correct_predictions

# Calculate the percentage of correct predictions
percentage_correct = (correct_predictions / total_predictions) * 100

# Print Statements
print(f"Total Predictions: {total_predictions}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Percentage of Correct Predictions: {percentage_correct:.2f}%")

# # Violin plot of data

# Violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='type_of_hit', y='exit_velocity', data=FilteredFoulsData)
plt.title('Distribution of Exit Velocity by Type of Hit')
plt.xlabel('Type of Hit')
plt.ylabel('Exit Velocity (mph)')
plt.show()


# # Print Base Statistics

# Calculate mean
mean_exit_velocity = FilteredFoulsData['exit_velocity'].mean()

# Calculate median
median_exit_velocity = FilteredFoulsData['exit_velocity'].median()

# Calculate max (high)
max_exit_velocity = FilteredFoulsData['exit_velocity'].max()

# Calculate min (low)
min_exit_velocity = FilteredFoulsData['exit_velocity'].min()

# Calculate standard deviation
std_dev_exit_velocity = FilteredFoulsData['exit_velocity'].std()

# Print Statements
print(f"Mean exit velocity: {mean_exit_velocity:.2f} mph")
print(f"Median exit velocity: {median_exit_velocity:.2f} mph")
print(f"Highest exit velocity: {max_exit_velocity:.2f} mph")
print(f"Lowest exit velocity: {min_exit_velocity:.2f} mph")
print(f"Standard deviation of exit velocity: {std_dev_exit_velocity:.2f} mph")


# # Bell Curve

# Plot the distribution of exit velocities with a bell curve
plt.figure(figsize=(12, 6))
sns.histplot(FilteredFoulsData['exit_velocity'], kde=True, bins=20) 
plt.title('Exit Velocity Distribution')
plt.xlabel('Exit Velocity (mph)')
plt.ylabel('Density')
plt.show()


# # Bar Plot

# Whitegrid graph style
sns.set_style("whitegrid")

# Get unique types of hits
hit_types = FilteredFoulsData['type_of_hit'].unique()

# Calculate the mean exit velocities
mean_exit_velocities = FilteredFoulsData.groupby('type_of_hit')['exit_velocity'].mean().sort_values()

# Create a bar plot with the sorted means
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=mean_exit_velocities.index, y=mean_exit_velocities.values, palette="viridis")

# Annotate the bars with the mean value
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                     textcoords = 'offset points')

# Set the y-axis to show a broader range to make the differences less extreme
plt.ylim(mean_exit_velocities.min() - 5, mean_exit_velocities.max() + 5)

# Label Axes
plt.title('Average Exit Velocity per Type of Hit')
plt.xlabel('Type of Hit')
plt.ylabel('Average Exit Velocity (mph)')
plt.xticks(rotation=45)  # Rotates the labels on the x-axis for better readability
plt.show()

import pandas as pd
import scipy.stats as stats
# # Confidence Interval per Type of Hit
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

# Convert the dictionary to a DataFrame for better visualization
ci_df = pd.DataFrame(ci_dict).T
ci_df
