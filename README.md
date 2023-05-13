
# EDA-AMES HOUSING PRICING 

This project involves performing EDA on the housing dataset.
The Ames Housing dataset contains information about home sales in Ames, Iowa between 2006 and 2010.

## Authors

- [@alpha_guya](https://github.com/Mmayi1)

## Documentation
The data is available in Kaggle. Sign up and get access to all the material.
[Documentation](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


## FAQ

#### Question 1. Load the Dataset with Pandas
Answer 1

Import pandas with the standard alias `pd` and load the data into a dataframe with the standard name `df`.
```Code 
# Loading the data
df = pd.read_csv('data/ames.csv', index_col=0)
```
```Code
#Providing details on the AMES dataset
df.info()
missing_values = df[['SalePrice', 'TotRmsAbvGrd', 'OverallCond', 'YrSold', 'YearBuilt', 'LandSlope']].isna().sum()
# print the number of missing values in each column important column
missing_values
#Checking how the values in the columns of interest in dataframe relate to each other
pd.plotting.scatter_matrix(df[['SalePrice', 'TotRmsAbvGrd', 'OverallCond', 'YrSold', 'YearBuilt', 'LandSlope']])




#### Question 2. Explore Data Distributions

Answer 2


Generating visualizations showing the distributions of `SalePrice`, `TotRmsAbvGrd`, and `OverallCond`.

Visualization have appropriate title and axes labels, as well as a red vertical line indicating the mean of the dataset. See the documentation for [plotting histograms](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html), [customizing axes](https://matplotlib.org/stable/api/axes_api.html#axis-labels-title-and-legend), and [plotting vertical lines](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html#matplotlib.axes.Axes.axvline) as needed.

### Sale Price

 A histogram for `SalePrice`.
```
#funtion to plot the visualization
def plot_histogram(df, column, title, xlabel, ylabel):
    # Extract the relevant data
    data = df[column]
    mean = data.mean()
    # Set up plot
    fig, ax = plt.subplots(figsize=(10,7))
    # Plot histogram
    ax.hist(data, bins="auto")
    # Plot vertical line
    ax.axvline(mean, color="black")
    # Customize title and axes labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

plot_histogram(
    df,
    "SalePrice",
    "Distribution of Sale Prices",
    "Sale Price",
    "Number of Houses"
)
```
![png]()
**Printing out mean of Sale Price 
df['SalePrice'].mean()
Mean of Sale Prce is 180921.19589041095

**Printing out median of Sale Price 
df['SalePrice'].median()
Median of Sale Prce is 163000.0

**Printing out std of Sale Price 
df['SalePrice'].median()
Std of Sale Prce is 79442.50288288663

```Also the below code was clean
#

def print_stats(df, column):    
    print("Mean:              ", df[column].mean())
    print("Median:            ", df[column].median())
    print("Standard Deviation:", df[column].std())
    
print_stats(df, "SalePrice")
```

    Mean:               180921.19589041095
    Median:             163000.0
    Standard Deviation: 79442.50288288662
Interpretion of the statistics:


```
"""
#"The data distribution shows a symmetrical pattern resembling a bell curve, indicative of a normal distribution.
#Most of the houses in the sample are clustered around the median value of $163,000. 
#Nevertheless, the mean value is higher than the median at over $180,000 due to the influence of high-priced properties."
"""
```

### Total Rooms Above Grade

 Bar chart for `TotRmsAbvGrd`.


```# Examining the spread of total rooms above grade

sns.countplot(x = 'TotRmsAbvGrd', data = df)
```
Print out of the mean, median, and standard deviation:


```
print_stats(df, "TotRmsAbvGrd")
```

    Mean:               6.517808219178082
    Median:             6.0
    Standard Deviation: 1.6253932905840505


Interpret the above information.


```
"""
The distribution of the number of rooms in houses is quite similar to a normal distribution, with the average and median both around 6 rooms. Although there are some houses with twice as many rooms as the average, the overall distribution is less skewed than the distribution of sale prices.
"""
```

### Overall Condition

In the cell below, produce a histogram for `OverallCond`.

```

# We are again reusing the same function

plot_histogram(
    df,
    "OverallCond",
    "Distribution of Overall Condition of Houses on a 1-10 Scale",
    "Condition of House",
    "Number of Houses"
)
```
    

Print out of the mean, median, and standard deviation:


```

print_stats(df, "OverallCond")
```

    Mean:               5.575342465753424
    Median:             5.0
    Standard Deviation: 1.1127993367127316


Interpretation of the above information.


```
"""
Most homes have a condition of 5. It seems like we should
treat this as a categorical rather than numeric variable,
since the difference between conditions is so abrupt
"""
```




##### Question 3. Explore Differences between Subsets


Answer 3


Overall condition of the house should be   a categorical variable.

Creation of categories of the full dataset based on that categorical variable, then plotting their distributions based on `SalePrice` as variable.

The categories created  were:

* `below_average_condition`: home sales where the overall condition was less than 5
* `average_condition`: home sales where the overall condition was exactly 5
* `above_average_condition`: home sales where the overall condition was greater than 5

```
below_average_condition = df[df["OverallCond"] < 5]
average_condition = df[df["OverallCond"] == 5]
above_average_condition = df[df["OverallCond"] > 5]
```
# Create a bar plot of mean sale price for each condition group using Seaborn
mean_sale_price = [below_average_condition["SalePrice"].mean(),
                   average_condition["SalePrice"].mean(),
                   above_average_condition["SalePrice"].mean()]

sns.barplot(x=['Below average', 'Average', 'Above average'], y=mean_sale_price)
plt.title('Mean Sale Price vs Overall Condition')
plt.xlabel('Overall Condition')
plt.ylabel('Mean Sale Price')
plt.show()

#"Firstly, it's worth noting that most of the houses in the dataset are considered to be in average condition, with around one-third of them being in above-average condition and less than 10% being in below-average condition. 
#This suggests that the average condition group covers a wider range of sale prices compared to the other two groups. 
#As expected, houses in below-average condition have lower sale prices compared to the other two groups. 
#However, it's surprising that houses in above-average condition don't have significantly higher average sale prices than those in the average condition group. 
#Interestingly, above-average condition houses appear to be concentrated in the $100,000 to $200,000 price range, while houses in the average condition group tend to sell more frequently for prices above $200,000. 
#Further investigation is necessary to better understand the characteristics of above-average condition houses, as this finding challenges the common belief that better condition always leads to higher prices."

#Summary statistics for Overall condition retrieved by the function
def print_stats(df, column):    
    print("Mean:              ", df[column].mean())
    print("Median:            ", df[column].median())
    print("Standard Deviation:", df[column].std())
    
print_stats(df, "OverallCond")
Which is :
Mean:               5.575342465753424
Median:             5.0
Standard Deviation: 1.1127993367127316


```
"""
##Generally, the plot clearly shows that most  houses are below average and average interms of their overall condition with 1 or 2 kitchens above grade.
#"Houses that have two kitchens were sold for less than $200,000, whereas some homes with only one kitchen were sold for considerably more
#Sale Price increases with increase in ranking on the basis of overall condition of the house."
"""
```

# Examining the spread of total rooms above grade

sns.countplot(x = 'TotRmsAbvGrd', data = df)
#The distribution of the number of rooms in houses is quite similar to a normal distribution, with the average and median both around 6 rooms.
#Although there are some houses with twice as many rooms as the average, the overall distribution is less skewed than the distribution of sale prices.






#### Question 4. Explore Correlations

Answer 4


To understand more about the variables of these homes lead to higher sale prices, correlations knowledge was key. 

Check the correlations with numeric data type.




```
# Get a list of correlations with SalePrice, sorted from smallest
# to largest
correlation_series = df.corr()['SalePrice'].sort_values()
# Select second to last correlation, since the highest (last)
# correlation will be SalePrice correlating 100% with itself
max_corr_value = correlation_series.iloc[-2]
max_corr_column = correlation_series.index[-2]
print("Most Positively Correlated Column:", max_corr_column)
print("Maximum Correlation Value:", max_corr_value)


# We can just find the smallest value, not the second smallest,
# since we aren't avoiding the perfect correlation with itself
min_corr_value = correlation_series.iloc[0]
min_corr_column = correlation_series.index[0]

print("Most Negatively Correlated Column:", min_corr_column)
print("Minimum Correlation Value:", min_corr_value)

```To plot the max-min correlation graph

import seaborn as sns

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))

# Plot distribution of column with highest correlation
sns.boxplot(
    x=df[max_corr_column],
    y=df["SalePrice"],
    ax=ax1
)
# Plot distribution of column with most negative correlation
sns.boxplot(
    x=df[min_corr_column],
    y=df["SalePrice"],
    ax=ax2
)

# Customized labels
ax1.set_title("Overall Quality vs. Sale Price")
ax1.set_xlabel("Overall Quality")
ax1.set_ylabel("Sale Price")
ax2.set_title("Number of Kitchens vs. Sale Price")
ax2.set_xlabel("Number of Kitchens Above Ground")
ax2.set_ylabel("Sale Price");
```


    


Interpretion of the results is as below with reference from the data dictionary.


```
"""
The column with the highest correlation is overall quality.
According to the data description/data dictionary:

OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
#It is seemed hard to analyse all those groups and therefore, I regrouped the responses into three categories to make the analysis more clear.
       
#"Firstly, it's worth noting that most of the houses in the dataset are considered to be in average condition, with around one-third of them being in above-average condition and less than 10% being in below-average condition. 
#This suggests that the average condition group covers a wider range of sale prices compared to the other two groups. 
#As expected, houses in below-average condition have lower sale prices compared to the other two groups. 
#However, it's surprising that houses in above-average condition don't have significantly higher average sale prices than those in the average condition group. 
#Interestingly, above-average condition houses appear to be concentrated in the $100,000 to $200,000 price range, while houses in the average condition group tend to sell more frequently for prices above $200,000. 
#Further investigation is necessary to better understand the characteristics of above-average condition houses, as this finding challenges the common belief that better condition always leads to higher prices."
"""
```

#Viewing the correlation dataframe
df[['SalePrice', 'TotRmsAbvGrd', 'OverallCond', 'YrSold', 'YearBuilt', 'LandSlope']].corr()

#Storing the correlation dataframe
hm = df[['SalePrice', 'TotRmsAbvGrd', 'OverallCond', 'YrSold', 'YearBuilt']].corr()
hm

#visual of heatmap for correlation among select variables
corr_viz = sns.heatmap(hm, annot = True)


#Sale Price has a correlation coefficient of 0.53 with total rooms above grade indicates a moderate positive correlation between two variables. This means that as one variable increases, the other variable tends to increase as well, but the relationship is not particularly strong.
#The same applies to correlation between Sale Price and Year the house was built which has moderate positive correlation of strength 0.52.
#A correlation coefficient of -0.029(Sale Price vs Year house was sold) indicates a very weak negative correlation between two variables. This means that as one variable increases, the other variable tends to decrease slightly, but the relationship is not significant.





#### Question 5. Engineer and Explore Age

Answer 5

```
# #creating a new variable and naming it Age
df["Age"] = df["YrSold"] - df["YearBuilt"]

# plot for Sale Price vs Age 
f# Plot the scatter plot of age vs sale price using seaborn
sns.scatterplot(x='Age', y='SalePrice', data=df, color='gold')
sns.set_style("darkgrid")
plt.title('Relationship between Age and Sale Price')
plt.show()


  
```Interpretation of the plot
"""
#"In general, newer homes tend to command higher prices, with their values increasing as they age. 
#However, it's important to note that the variability in sale prices appears to rise after homes reach the age of 100 years.
#This is because some houses may have higher-than-average sale prices, but there are relatively few home sales in general. 
#Moreover, there may have been periods of rapid expansion and decline in the housing market in recent decades. 
#Evidence of this is apparent in the relatively low number of homes sold that are around 20 years old compared to those that are slightly above 20 years old but less than approximately 25 years old.
#More exploration of this pattern may reveal valuable insights which needs to be explored.".
"""
```
**THE END**


## Features

- Light/dark mode toggle
- Live previews
- Fullscreen mode
- Cross platform


## 🚀 About Me
A junior but passionate data analyst. Obsessed with Data viz.


## 🛠 Skills
Excel, Stata, Rstudio, Nvivo, MaxQDA


## Lessons Learned

While building this project i learnt that focus in practice is key. 
Challenges faced included: spending Sleepless nights practising code, tight daily schedules,inadequate/poor network most of the time and insufficient access to revision material tutorials from my tutor.
I overcome the above challenges by assuming the tutor is unable to understand and used Antony Muikos few videos.


## Acknowledgements

 - [Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
 - Antony Muiko-Data Science tutor.
 - [plotting histograms](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html), [customizing axes](https://matplotlib.org/stable/api/axes_api.html#axis-labels-title-and-legend), and [plotting vertical lines](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html#matplotlib.axes.Axes.axvline)
 

