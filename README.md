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

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## Exploratory Data Analysis: Time Series

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## Exploratory Data Analysis: Pairplot

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```

## Exploratory Data Analysis: Pairwise Correlation Heatmap

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
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

