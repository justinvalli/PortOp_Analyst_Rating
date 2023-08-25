# Portfolio Optimization Analyst Rating

## Unlocking Insights 2.0: A Cutting-Edge Approach to Sell-Side Equity Research with Advanced Machine Learning Models

![Intro Image](read_me_images/intro.png)

This file will explore Portfolio Optimization utilizing the Efficient Frontier, as well as Machine Learning to determine the optimal weights of assets in a portfolio. The portfolio contains eight assets: GLD (Gold), VNQ (Real Estate), USO (Oil Commodity), K (Consumer Staple), AAPL (Tech), TSLA (Tech), AGG (Investment Grade Bonds), JNK (Junk Bonds). We look to the risk/return tradeoff using two portfolio options: the Sharpe optimized portfolio, as well as the minimum volatility (low risk) portfolio. We utilize two categories of indicators: technical (moving averages, price changes, etc.) as well as macro indicators (yield curve). Finally, we utilize four predictive machine learning models: the random forest, the logistic regression, the neural network, and the XGBoost algorithm. 

![Intro Image](read_me_images/flow_chart.png)

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
import seaborn as sns
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

```[INSERT CODE HERE FOR DATASET MANIPULATION AND BUILDING]
```

> Note: EPS and EBITDA were hard-coded into the data frame. You can view these values in Yahoo Finance. 

## Dataset Building: 

[INSERT DETAILS ABOUT DATASET BUILDING HERE]

Now that we have built out the primary dataset, we can begin to use Machine Learning to build a predictive classification model.

![Intro Image](read_me_images/data_preparation.png)

## The Efficient Frontier: 

![Intro Image](read_me_images/efficient_frontier_1.png)
[INSERT THE EFFICIENT FRONTIER CODE AND INTRODUCTION HERE]

```[INSERT CODE HERE FOR EFFICIENT FRONTIER HERE]
```

## The Efficient Frontier 2.0: 

![Intro Image](read_me_images/efficient_frontier_2.png)
[INSERT THE EFFICIENT FRONTIER 2.0 CODE AND INTRODUCTION HERE]

```[INSERT CODE HERE FOR EFFICIENT FRONTIER 2.0 HERE]
```
## Exploratory Data Analysis: Distribution of Key Variables

![Intro Image](read_me_images/histogram.png)

We look at the distribution of the independent variables in our model in order to observe market trends and volatility.

We found some interesting insights in the histograms:  
* GLD: The distribution of GLD closing prices looks almost symmetric
* VNQ: VNQ's distribution is slightly skewed, suggesting  variation within its closing prices.
* USO: USO's distribution shows a narrow spread, indicating stable prices.

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

![Intro Image](read_me_images/time_series.png)

We utilize time series plots to show how the prices of the independent variables (GLD, VNQ, USO, TSLA, AAPL, AGG, JNK) change over the given time period. We noticed some interesting trends:

( 1 ) Gold as a safe haven - The upward trend in Gold during periods of market turbulence suggests that investors turned to gold during uncertain times and sudden spikes related to geopolitical events or economic instability
( 2 ) Fluctuations in Oil market dynamics related to OPEC decisions, geopolitical tensions, and economic growth affecting oil prices - energy market conditions
( 3 ) TSLA and APPL trends can provide insight into the technology sector's overall health, indicating robust consumer demand


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

![Intro Image](read_me_images/pairplot.png)

Pairplot for Exploratory Data Analysis (EDA):

We used a pair plot to see visual representations of relationships and trends between pairs of variables.

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

![Intro Image](read_me_images/heatmap.png)

We want to identify multicollinearity between independent variables to determine if they should be included in the machine learning models. 

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

![Intro Image](read_me_images/xg_boost.png)

This algorithm yielded a resounding accuracy score in discerning the “Buy” or “Sell” signal.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## RandomForest Model: 

[INSERT RANDOM FOREST IMAGE HERE]
![Intro Image](read_me_images/random_forest.png)

After backtesting and manual optimization, we found the random forest to yield very high accuracy when predicting the correct BUY or SELL classification. 

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier

After incorporating the additional hyperparameters into the code, we observed a notable improvement in the accuracy scores for both the original and newly generated datasets. 

```
![Machine_learning_app]

Code in Streamlit with hyperparameters

if Model == 'Random Forest':
        st.subheader('The **Random Forest Regressor** is used to build a regression model using the **Random Forest** algorithm. Try adjusting the hyperparameters!')
        st.write("## Apple - AAPL")
        #define the ticker symbol
        tickerSymbol ='AAPL'
        #get data on this ticker
        tickerData = yf.Ticker(tickerSymbol)
        #get the historical prices for this ticker
        tickerDf = tickerData.history(period='1d', start='2015-08-01', end='2023-08-01')
        # Open	High	Low	Close	Volume	Dividends	Stock Splits

        st.write("""
        ## Closing Price
        """)
        st.line_chart(tickerDf.Close)
        # Displays the dataset
        st.subheader('Dataset')
        
        if uploaded_file is not None:  # Check if a file is uploaded
            df = pd.read_csv(uploaded_file)
            df.reset_index(drop=True, inplace=True)
            df.drop(columns=["Unnamed: 0"], inplace=True)  
            st.markdown('**Sample of dataset**')
            st.write(df)

            # Model building
            X = df.iloc[:,:-1]  # Using all columns except for the last column as X
            Y = df.iloc[:,-1]   # Selecting the last column as Y

            # Data splitting
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

            st.markdown('**Data splits**')
            st.write('Training set')
            st.info(X_train.shape)
            st.write('Test set')
            st.info(X_test.shape)

            st.markdown('**Variable details**:')
            st.write('X variable')
            st.info(list(X.columns))
            st.write('Y variable')
            st.info(Y.name)

            rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
            random_state=parameter_random_state,
            max_features=parameter_max_features,
            criterion=parameter_criterion,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            bootstrap=parameter_bootstrap,
            #oob_score=parameter_oob_score,
            n_jobs=parameter_n_jobs)
            rf.fit(X_train, Y_train)

            st.subheader('Model Performance')

            st.markdown('**Training set**')
            Y_pred_train = rf.predict(X_train)
            st.write('Train Accuracy Score:')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**Test set**')
            Y_pred_test = rf.predict(X_test)
            st.write('Test Accuracy Score:')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('Model Parameters')
            st.write(rf.get_params())
               
        else:
            st.write("Upload a CSV file to start model training.")
```

## LogisticRegression Model: 

[INSERT LOGISTIC REGRESSION IMAGE HERE]
![Intro Image](read_me_images/logistic_regression_results.png)

After backtesting and manual optimization, we found the logistic regression to yield very high accuracy when predicting the correct BUY or SELL classification.  

```
elif Model == 'Logistic Regression':
        st.subheader("Our best model was the Logistic Regression Model. This was in line with our expectations, given the formatting of our dataset and the fact that this is ultimately a classification problem. Please see the image outlining the precision, recall, F1 score, support, and accuracy score of the model.")
        st.image("logistic_regression1.png", width=800)
        st.image("logistic_regression_results.png", width=800)
        #READ IN DATA
        df= pd.read_csv("max_rating1.csv")
        df.head()
        # Select features and target variable
        X = df.drop(columns=["Signal"])
        y = df["Signal"]

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create a pipeline for preprocessing and modeling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])

        # Define the hyperparameters grid for grid search
        param_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        }

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the model on training data
        y_train_pred = best_model.predict(X_train)
        train_report = classification_report(y_train, y_train_pred)
        st.subheader("Training Report:")
        st.text(train_report)

        # Evaluate the model on testing data
        y_test_pred = best_model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred)
        st.subheader("Testing Report:")
        st.text(test_report)
    elif Model == 'Neural Network':
        st.subheader("After backtesting and manual optimization, we found the neural network to be less than ideal for predicting the correct BUY or SELL classification. We utilized 2 models. The first model yielded an accuracy score of 0.4701, while the second model gave 0.5203 accuracy score. The difference between the first two models can be found within the hyperparameter tuning. For example, we changed the loss function from Categorical Cross-Entropy to Mean Squared Error, number of hidden nodes from 10 to 20, number of neurons form 2 to 3, and optimizer function from sigmoid to adam.")
        st.image("neural network.png")        

```

## NeuralNetwork Model: 

[INSERT NEURAL NETWORK IMAGE HERE]
![Intro Image](read_me_images/neural_network.png)

After backtesting and manual optimization, we found a significant increase in the accuracy score of the model for predicting the correct BUY or SELL classification. 

```
[INSERT NEURAL NETWORK REGRESSION MODEL]
```

## [INSERT STREAMLIT HERE]

[INSERT SCREENSHOTS OF STREAMLIT HERE]
![Intro Image](read_me_images/logistic_regression_results.png) [INSERT STREAMLIT EXPLANATION OF RESULTS FOR THE MACHINE LEARNING MODELS HERE]

```
[INSERT STREAMLIT CODE HERE]
```

## Next Steps 

[INSERT IMAGE OF COMPANY BANK LOGOS HERE]
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

