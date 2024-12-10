*UCSD DSC80: Principles of Data Science*  
**Authors:** Leo Bangayan and Ryan Xavier  

## Overview  
This project focuses on analyzing recipe and rating data to uncover trends, patterns, and insights into user preferences and recipe performance. By leveraging various data science techniques, we explore key factors influencing ratings, identify popular recipes and ingredients, and find the realtionshps minutes, calories and average rating.

## Cleaning and EDA
We started with two datasets one containing recipes and another reviews for thsoe recipes. We took the average rating per recipe from the review data and then appended that to our recipe data. The result is that every recipe now includes the average rating from our review data!

| name                                 |   minutes |   n_steps |   n_ingredients |   average_rating |   calories |
|:-------------------------------------|----------:|----------:|----------------:|-----------------:|-----------:|
| 1 brownies in the world    best ever |        40 |        10 |               9 |                4 |      138.4 |
| 1 in canada chocolate chip cookies   |        45 |        12 |              11 |                5 |      595.1 |
| 412 broccoli casserole               |        40 |         6 |               9 |                5 |      194.8 |
| millionaire pound cake               |       120 |         7 |               7 |                5 |      878.3 |
| 2000 meatloaf                        |        90 |        17 |              13 |                5 |      267   |

<iframe src="assets/rate_plot.html" width=800 height=600 frameBorder=0></iframe>
This is a univarite plot showing the distribution of average rating of our dataset. It shows that higher average ratings
are much more common then lower ones. One possible explanion is that people are more likely to give a review on a recipe they really like and this could possibly skew the reviews higher.


<iframe src="assets/Bivariate.html" width=800 height=600 frameBorder=0></iframe>
This is a bivarite plot showing the realtionship between the number of ingredients in a recipe and the average calories those recipes had. After filtering out some ouliers the general trend shown is that as the number of ingredients increases so does the average calorie count. A possible explanation is that more elaborate recipes are more indulgent and as such have more calories.



---
## Assesment of Missingness

In our assement of the data we had only three columns with any missing data. The first was average rating which we know to be mssing by design since it is a product of the way we merged our data. If a recipe didn't have a review then its average rating would be missing since there is nothing to base it off of. Then we looked at name but it had only one missing value so we ignored it. That left only missing reviews to look at. After running some permuation tests we found that the shape of the distributions of number of ingredients is different for recipes with missing discripions vs recipes that have descriptions. Meaning that descriptions is MAR dependent on number of ingredients.
<iframe src="assets/missingness.html" width=800 height=600 frameBorder=0></iframe>


## Hypothesis Test

We ran a permuation test to see if the distribution of average ratings was the same for recipes under 60 minutes with those overs. Our null hypothesis was that there is not a difference in the shape of distributions of the average recipes for recipes under 60 minutes to those over. Our alternative hypothesis is that thier is a difference in the shape of the distrubiton of average rating for reciupes under 60 minutes and those over. After running the test we got a p value = 0.022 which is significant at our chosen level of 0.05. We chose this test and this statistic could because if the distributions are different then minutes might make a good feature for a model to predict the average rating of a recipe.

## Prediction Problem

Our predictive model will be using a linear regresssion model in order to predict a recipe's average rating. We chose this variable because we wanted to see what factors go into a populer vs unpopuler recipe. Also predicting for average rating would allow us to use all other features because at its time of prediction all other information is already submitted by the creater of the recipe. The way information is submitted is the creater recipe submits a recipe with all its information(minutes, calories, etc) and then people review it so at the time of prediction we has all the submitted data. In order to evulate our model, we are using root mean squared error. We chose this metric because it is easier to intrept since it is in the same units our predection variable of average rating and is widly used.

## Baseline Model
Our first model has three features. The first two are numurical(number of ingredients and calories) and the last one is catagorical which would be the author of the of the recipe. In order to use the author we had use a hot one encode transformer but we just passed through the other two features. Using this current model we got an Root Mean Squared Error(RMSE) of about 0.668. This was a pretty good score it means that our predections are aproximatly 0.67 rating points different from the actual rating of the recipe.



## Final Model
To improve our model we added two new features. The first is a binerizer for the number of calories a recipe has. This is useful because it will make our model more robust to outliers of recipes that have alot of a calories. The other feature we added was a quantile transformer for our minutes columns. We did this because we noticed that minutes had a couple outlers since some recipes were multiple months long so this transformer added more robustness to our model. 

We started a using a lasso Regession model and used Cross validation tests with 5 folds to tune our hyperparamter. The best value we found for our alpha value(hyperparamter) is 0.01. After running this model we got a new RSME of 0.63. This is an improvement from our basemodel as we reduced RSME by 0.04 making our model more accurate.


## Fairness Analysis

For our Fairness analysis we divided our data into two groups recipes with high or low number of ingrdients at threshold of 10 ingredients. We used RSME as our evulation metric and the difference in RSME is our sample statistic. Our null Hypothesis is that there is no difference in RSME from recipes with low or high number of ingredints. Our Alternative Hypothesis is that there is a difference in RSME from recipes with low or high number of ingrdients.  After running our permuation test we got a p-value of 0.096 which is not below a 0.05 significance level. This means we fail to reject our null hypothesis that there isn't a difference in predicion accuracy for recipes with high or low number of ingredients.