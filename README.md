# Portfolio Optimization Analyst Rating


## Unlocking Insights 2.0: A Cutting-Edge Approach to Sell-Side Equity Research with Advanced Machine Learning Models

[INSERT SCREENSHOT OF THE TITLE SLIDE HERE SIMILAR TO BELOW]
![Intro Image](read_me_images/intro_image.png)

This file will explore Portfolio Optimization utilizing the Efficient Frontier, as well as Machine Learning to determine the optimal weights of assets in a portfolio. The portfolio contains eight assets: GLD (Gold), VNQ (Real Estate), USO (Oil Commodity), K (Consumer Staple), AAPL (Tech), TSLA (Tech), AGG (Investment Grade Bonds), JNK (Junk Bonds). We look to the risk/return tradeoff using two portfolio options: the Sharpe optimized portfolio, as well as the minimum volatility (low risk) portfolio. We utilize two categories of indicators: technical (moving averages, price changes, etc.) as well as macro indicators (yield curve). Finally, we utilize four predictive machine learning models: the random forest, the logistic regression, the neural network, and the XGBoost algorithm. 

[INSERT SCREENSHOT OF THE FLOWCHART HERE]

## Usage: 

> Note: `PortOp_Analyst_Rating` is found in the main folder. Datasets, code for models, etc. are found in the branch Folder.

### Import the following libraries and dependencies: 
```
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
```

### Read and open `max_rating.csv` and `min_rating.csv` for the four models (XGBoost, Neural Network, RandomForest, LogisticRegression) 
```# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
df = df = pd.read_csv("max_rating.csv")

# Review the DataFrame
df
```

## Usage for `max_rating.csv` and `min_rating.csv`:

[INSERT DATA MANIPULATION PROCESS HERE] 

For example: 

```python
# Pull aapl ticker with date range "03/32/2023" to "07/03/2023"
aapl_1 = yf.download("AAPL", start='2023-03-31', end='2023-07-03', interval="1d")

aapl_1['Total Debt'] = aapl_debt.loc['2023-03-31']
aapl_1['Shares'] = aapl_shares.loc['2023-03-31']
aapl_1['Cash'] = aapl_cash.loc['2023-03-31']
aapl_1['EPS'] = 1.53
aapl_1['EBITDA'] = 31260000000
```

> Note: EPS and EBITDA were hard-coded into the data frame. You can view these values in Yahoo Finance. 

## Dataset Building: 

[INSERT DETAILS ABOUT DATASET BUILDING HERE]

Now that we have built out the primary dataset, we can begin to use Machine Learning to build a predictive classification model.

![Intro Image](read_me_images/dataset.png)

## The Efficient Frontier: 

[INSERT THE EFFICIENT FRONTIER CODE AND INTRODUCTION HERE]

```python
# Call pipeline to run `sentiment-analysis`
classifier = pipeline('sentiment-analysis')
```

## The Efficient Frontier 2.0: 

[INSERT THE EFFICIENT FRONTIER 2.0 CODE AND INTRODUCTION HERE]

```python
# Call pipeline to run `sentiment-analysis`
classifier = pipeline('sentiment-analysis')
```
## Exploratory Data Analysis: Distribution of Key Variables

![Intro Image](read_me_images/random_forest.png)

By examining the distribution of key variables' closing prices, we gain insights into their behavior.
This information is crucial for understanding market trends, volatility, and potential investment opportunities.

Key Takeaways: 
* GLD: The distribution of GLD closing prices appears symmetric, with most prices centered around a particular value.
* VNQ: VNQ's distribution is slightly skewed, suggesting some variations in its closing prices.
* USO: USO's distribution shows a narrow spread, indicating relatively stable closing prices.
* TSLA: TSLA's distribution has a wide spread, indicating greater volatility in its closing prices.
* AAPL: AAPL's distribution resembles a bell curve, which is common in well-behaved datasets.
* AGG: AGG's distribution appears symmetric, suggesting a consistent range of closing prices.
* JNK: JNK's distribution exhibits a single peak, showing a dominant range of closing prices.

```
# Define the key variables
key_vars = ['GLD', 'VNQ', 'USO', 'TSLA', 'AAPL', 'AGG', 'JNK']

# Calculate the number of rows and columns for subplots
n_cols = 4
n_rows = -(-len(key_vars) // n_cols)  # Ceiling division to calculate rows

# Create the histogram subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 10))
fig.suptitle('Distribution of Key Variables', fontsize=20, y=1.03)

# Plot histograms for each variable with royal blue color and modified x-axis labels
for i, var in enumerate(key_vars):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    sns.histplot(df[var], bins=20, ax=ax, color='#4169E1', alpha=0.7, kde=True)  # Royal blue color
    ax.set_title(var)
    ax.set_xlabel(f'{var} Closing Price')
    ax.set_ylabel('Frequency')

# Remove empty subplots if needed
if len(key_vars) < n_rows * n_cols:
    for i in range(len(key_vars), n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the plots as an image
plt.savefig('key_variables_histograms_max.png')

# Show the plots
plt.show()
```

## Exploratory Data Analysis: Time Series

![Intro Image](read_me_images/random_forest.png)

Exploring Trends Over Time:

Understanding Price Movements: Time series plots show how the prices of key variables (GLD, VNQ, USO, TSLA, AAPL, AGG, JNK) change over
the given time period (as indicated by the "Date" on the x-axis).
Identifying Trends: By observing the trajectory of the lines, you can identify trends such as upward or downward movement, 
stability, or periods of volatility.
Detecting Seasonality:

Recurring Patterns: Time series plots can reveal recurring patterns or seasonality in the data.
These patterns may correspond to certain times of the year or economic cycles.
Identifying Anomalies and Outliers:

Unusual Movements: Sudden spikes or drops in the plot can indicate anomalies or outliers. 
These may be caused by significant events affecting the market.
Understanding Volatility:

Fluctuations: The degree of movement in the plot indicates the volatility of the variable's price.
Wide fluctuations suggest higher volatility, while steadier movements suggest stability.
Impact of External Factors:

News and Events: Time series plots can show how external events, such as economic announcements or major news,
impact the prices of the key variables.
Long-term and Short-term Analysis:

Observing Long-term Trends: By analyzing the entire time range, you can spot long-term trends, enabling you to make 
informed investment decisions.
Zooming into Short-term Movements: You can also zoom into shorter time periods to analyze short-term movements and make tactical decisions.

```
# Define the time variables
time_vars = ['GLD', 'VNQ', 'USO', 'TSLA', 'AAPL', 'AGG', 'JNK']

# Set the style using Seaborn
sns.set(style="darkgrid")

# Create the time series subplots
fig, axes = plt.subplots(nrows=len(time_vars), ncols=1, figsize=(8, 12), sharex=True)
fig.suptitle('Time Series Plots', fontsize=20)

# Plot time series for each variable with enhanced style
for i, var in enumerate(time_vars):
    ax = axes[i]
    sns.lineplot(data=df, y=var, x='Date', ax=ax, linewidth=2)
    ax.set_xlabel(None)
    ax.tick_params(axis='both', which='both', labelsize=10)
    ax.legend(labels=[var], loc='upper left', fontsize=10)
    ax.margins(x=0.02)  # Add a small margin to the x-axis
    
    # Format y-axis ticks with proper spacing
    if i == 0:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True))  # Adjust number of ticks
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))  # Adjust number of ticks

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add space at the top for the title
plt.subplots_adjust(hspace=0.4)  # Adjust vertical space between subplots

# Save the plots as an image
plt.savefig('time_series_plots_max.png')

# Show the plots
plt.show()
```

## Exploratory Data Analysis: Pairplot

![Intro Image](read_me_images/random_forest.png)

Pairplot for Exploratory Data Analysis (EDA):

Visual Exploration: Pairplot displays scatter plots for selected variables, 
providing a quick visual exploration of relationships and trends between pairs of variables.

Correlation Insight: It helps identify positive/negative correlations between variables.
Correlated variables can impact model performance and guide feature selection.

Pattern Recognition: Scatter plots reveal patterns like clusters, trends, and outliers,
aiding in understanding data behavior and potential anomalies.

Distribution Insight: Histograms on the diagonal show variable distributions, 
offering a snapshot of their characteristics.

Decision Support: Pairplots assist in selecting variables for modeling by highlighting informative relationships.

Non-Technical Communication: The visual nature of pairplots makes them effective for
communicating insights to non-technical stakeholders.

Preprocessing Hints: Nonlinear relationships and skewed distributions might prompt data transformation or preprocessing.

Limitation Awareness: Detecting multicollinearity, where variables are highly correlated, is vital as it can impact model interpretation.

```
# Define the selected variables
selected_vars = ['GLD', 'VNQ', 'USO', 'TSLA', 'AAPL', 'AGG', 'JNK']

# Set the style using Seaborn
sns.set(style="ticks")

# Create the pairplot with the 'viridis' colormap
pairplot = sns.pairplot(df[selected_vars], diag_kind="kde", kind="scatter", palette="viridis")

# Add "Closing Price" to the axes labels
for i, var in enumerate(selected_vars):
    pairplot.axes[i, 0].set_ylabel(f'{var} Closing Price')
    pairplot.axes[-1, i].set_xlabel(f'{var} Closing Price')

# Add a title and adjust layout
plt.suptitle('Pairplot of Selected Variables', y=1.02, fontsize=20)
pairplot.fig.subplots_adjust(top=0.93)

# Save the pairplot as an image
pairplot.savefig('pairplot_selected_variables_max.png')

# Show the plot
plt.show()
```

## Exploratory Data Analysis: Pairwise Correlation Heatmap

![Intro Image](read_me_images/random_forest.png)

In the context of your machine-learning model,
This heatmap helps you identify potential multicollinearity
between independent variables, which might affect the model's performance 
or interpretation. If two independent variables are highly correlated,
It could mean they provide similar information to the model,
and you might consider removing one to avoid redundancy.

 It doesn't directly indicate the relationship between the independent variables and the excluded "Signal" variable,
 since the heatmap only considers pairwise relationships between the independent variables.

```
# Exclude the 'signal' column from the DataFrame
corr_matrix = AAPL_data_df.drop(columns=['Signal']).corr()

# Adjust the figure size
plt.figure(figsize=(16, 12))

# Increase the spacing between cells using cbar_kws
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})

# Add more space between the heatmap and the title
plt.title('Correlation Heatmap', pad=20)

# Save the plot as an image
plt.savefig('correlation_heatmap_max.png')

# Show the plot
plt.show()
```

## XGBoost Model: 

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## RandomForest Model: 

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## LogisticRegression Model: 

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## NeuralNetwork Model: 

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## [INSERT STREAMLIT HERE]

![Intro Image](read_me_images/logistic_regression_results.png) [INSERT STREAMLIT EXPLANATION OF RESULTS HERE]

## Next Steps 

![Intro Image](read_me_images/next_steps.png)

We found this classification problem to be very interesting, and found that there were many opportunities to further enhance our model:

* More intricate trading algorithm by incorporating further complexity to enhance predictions 
* News sentiment, technical indicators, fundamental indicators, etc. 
* Enhancing feature engineering and selection for all three ML models 
* fundamental, technical, and macro to improve predictive accuracy
* More hyperparameter tuning with the Neural Network
* experimenting with different optimizers, loss, activation functions, and the number of epochs 


## SOURCES:

* https://www.machinelearningplus.com/machine-learning/exploratory-data-analysis-eda/
* https://www.youtube.com/watch?v=gfwNK3o45ng
* https://towardsdatascience.com/is-it-possible-to-predict-stock-prices-with-a-neural-network-d750af3de50b

