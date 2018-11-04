"""
Assignment 10
Data Science Masters

"""
"""
We have the min and max temperatures in a city In India for each months of the year. 
We would like to find a function to describe this and show it graphically, 
the dataset given below. Task: fitting it to the periodic function
plot the fit Data Max = 39, 41, 43, 47, 49, 51, 45, 38, 37, 29, 27, 25 
Min = 21, 23, 27, 28, 32, 35, 31, 28, 21, 19, 17, 18
"""
#%%
import numpy as np

# take values
temp_max = np.array([39, 41, 43, 47, 49, 51, 45, 38, 37, 29, 27, 25])
temp_min = np.array([21, 23, 27, 28, 32, 35, 31, 28, 21, 19, 17, 18])

import matplotlib.pyplot as plt

# plot
months = np.arange(12)
plt.plot(months, temp_max, 'go')
plt.plot(months, temp_min, 'co')
plt.xlabel('Month')
plt.ylabel('Min and max temperature')

#%%
# Fitting it to a periodic function
from scipy import optimize
def yearly_temps(times, avg, ampl, time_offset):
    return (avg + ampl * np.cos((times + time_offset) * 1.8 * np.pi / times.max()))

res_max, cov_max = optimize.curve_fit(yearly_temps, months,
                                      temp_max, [40, 20, 0])
res_min, cov_min = optimize.curve_fit(yearly_temps, months,
                                      temp_min, [-40, 20, 0])

#%%
# Plotting the fit
days = np.linspace(0, 12, num=365)

plt.figure()
plt.plot(months, temp_max, 'go')
plt.plot(days, yearly_temps(days, *res_max), 'm-')
plt.plot(months, temp_min, 'co')
plt.plot(days, yearly_temps(days, *res_min), 'y-')
plt.xlabel('Month')
plt.ylabel('Temperature ($^\circ$C)')

plt.show()
#%%
#This assignment is for visualization using matplotlib: data to use: 
#url= https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv titanic = pd.read_csv(url)
#Charts to plot:
#1) Create a pie chart presenting the male/female proportion
#2) Create a scatterplot with the Fare paid and the Age, differ the plot color by gender
#%%
import pandas as pd
# read data set from url
titanic = pd.read_csv('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv')
titanic.head()
#%%
# finding the proportion 
proportion = titanic.groupby(by = 'sex')['name'].count()
proportion
#%%
# Plot the pie chart
plt.pie(proportion, labels=(proportion.index), shadow=True, autopct='%.2f%%')
plt.axis('equal')
#%%
plt.scatter(titanic.fare, titanic.age,c = (titanic.sex.astype('category').cat.codes))
plt.ylabel('Age')
plt.title('Plot of Fare Paid Vs Age')