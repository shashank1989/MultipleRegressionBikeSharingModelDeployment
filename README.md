# Bike-Sharing System
> A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.

- A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario.
- It has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.


## Table of Contents
* [Business Overview](#Business-Objective)
* [Technologies Used](#Technologies-Used)
* [Exploratory Data Analysis & Model Steps](#Steps)
* [Conclusions](#Conclusions)

## Business-Objective

BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


- They wanted to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market.

-The company wants to know:

- - Which variables are significant in predicting the demand for shared bikes.
- - How well those variables describe the bike demands
- - Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors.

> dataset : day.csv

## Technologies-Used
- pandas      - Version : 1.2.4
- numpy       - Version : 1.20.1
- seaborn     - Version : 0.11.1
- matplotlib  - Version : 3.3.4
- statsmodels - Version : 0.12.2
- pandasql    - Version : 0.7.3
- sklearn     - Version :0.24.2

# Exploratory Data Analysis & Model Steps

## Steps

> ## EDA

- Data Clearning and Missing Data Analysis
- Outlier Analysis & Treatment Assumption values > Q3+1.5*IQR and values < Q1-1.5*IQR will be treated
- Deriving Categorical Columns
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis

> ## ModelPreparation

- Training and Test data split
- Feature Scaling - StandardScaler
- Feature Engineering & Selection using RFE and Variance Inflation factor
- ModelPreparation
- Residual Analysis
- Model Evaluation & Assessment
- Prediction
- Final Conclusion & Analysis

## Conclusions


###  EDA ANALYSIS

- Year wise : In year 2019 we have 62.33 % of total Bike sharing
- Month wise : Aug,June,Sep,July,May,Oct - from may to oct we have observe a upward trend in bike sharing
- Highest no of bike sharing will be in fall season and then in summer and spring have least no of bike sharing.
- The above trend follows the same for every season i.e. for every season Fall have highest no of bike sharing then summer then winter and spring have least no of bike sharing
- Around 69.6% of bike sharing is use to be done on working days
- HIghest no of bike sharing around 42% between temp 25-35 c and then 40% between 15 - 25 c
- Temp 15-35 c constitutes to 82 percent of total bike sharing and count is increasing with the increase in temp
- aTemp 15-35 c constitutes to 82 percent of total bike sharing and count is increasing with the increase in atemp
- HIghest no of bike sharing around 52% between atemp 25-35 c
- Around 65% of total bike sharing between 50-75 humidity and Count is decreasing with the increase in humidity
- Around 69% of total bike sharing with cloudy weather and the mist with 30%
- Count is decreasing with the increase in windspeed
- Around 63.4 % of Bike sharing population are cloudy
- Linear relationship between cnt & temp ,atemp
- Variables which have high correlation with cnt
- cnt have correlation with below fields (In decreasing correlation order)
- cnt ~ registered,temp,atemp,yr,season,mnth,windspeed(negative corr)
- temp have high correlation with atemp it seems atemp is derived from temp so we can keep temp column and remove atemp
- mnth & season have good correlation
- weathersit & hum have good correlation

### Model Analysis

- During spring season company may notice the downfall
- During winter & summer company may notice the increase in bike sharing count
- Company should expand the business based on the locality Temp 15-35 c constitutes to 82 percent of total bike sharing and with the increase in temp from 15 to 35 bike sharing will increase.
- During rain and thunderstorm company may notice the downfall
- Company Many Notice increase in Bike sharing count for below weather
-       - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
-       - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- If for a particular region if the windspeed is more company should plan the business accordingly since with the increase in windspeed bike sharing will decrease
- Company should target places for business expansion where the humidity is less

## Model Prediction Analysis

- For the year 2019 on the working day for sep month and season winter and weather : Mist + Cloudy, Mist + Broken clouds, Mist + Few

- with the increase in temp and decrease in humidity ,decrease in windspeed company can experience more Bikesharing

### Significant Variables

-         temp
-         season(spring & winter)
-         yr(2019)
-         mnth : May,Sep,Mar
-         weathersit_Mist
-         mnth_oct
-         holiday
-         windspeed
-         workingday
-         humidity

## Contact
Created by [@shashank1989] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
