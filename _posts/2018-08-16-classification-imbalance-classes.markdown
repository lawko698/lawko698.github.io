---
title: "Detecting Prostate Cancers"
layout: post
date: 2018-08-16 20:35
tag: 
- multiclass classification
- R
- Imbalance Classes
projects: true
hidden: true # don't count this post in blog pagination
description: "Classifying imbalance data"
category: project
author: Lawrence
externalLink: false
---

# A new tool for detecting prostate cancers
- Date: 30/10/2017
- R version: 3.5.1

Based on the available information, prostate cancer is the most common diagnosed cancer in Australia and the third most common cause of cancer death in men. With 85% of cases diagnosed in men over 65 years of age, it is more common in older men. In addition, there are numerous parameters having direct effects on the development  of this kind of cancer including the age, and physical and mental well-being. The cancer is generally categorised in four different stages, namely stage I to stage IV, based on the severity of cancerous cells. Besides different types of physical urinary changes in men, some test and sampling including bone and CT scans are used to determine the spread of cancerous cells.

The dataset obtained from a prostate cancer lab containing 3000 observations with 10 numerical features. In addition, there is a column showing the result of the test (class attribute), which is 0 for curable, and 1 for tumour stage, 2 for node stage and 3 for incurable cancers. All the features are numerical measurements rounded to the closest integer number between 1 and 10.

1)    Filling missing values with regression:
There are 4 missing values in the dataset. Our first task is to estimate them by means of regression analysis. Using the remaining full observations, find regression models, and estimate the values of the missing components.

2)    Build a classifier:
Build a classifier of your choice to learn from the data and perform the classification.


```R
#libraries Used
library(psych) #detailed summary stats
library(ggplot2) #visualizations
library(gridExtra) #ggplot layouts
library(car) #detailed scatterplot
library(randomForest) #random forest
library(hydroGOF) #rmse calculation
library(caret) # Linear regression with cross-validation
library(MASS) #LDA
library(pROC) #Multi-class ROC
library(glmnet) #logistic regularization
library(e1071) #support vector machine
library(VIM) #kNN for imputation
```

    
    Attaching package: 'ggplot2'
    
    The following objects are masked from 'package:psych':
    
        %+%, alpha
    
    Loading required package: carData
    
    Attaching package: 'car'
    
    The following object is masked from 'package:psych':
    
        logit
    
    randomForest 4.6-14
    Type rfNews() to see new features/changes/bug fixes.
    
    Attaching package: 'randomForest'
    
    The following object is masked from 'package:gridExtra':
    
        combine
    
    The following object is masked from 'package:ggplot2':
    
        margin
    
    The following object is masked from 'package:psych':
    
        outlier
    
    Loading required package: zoo
    
    Attaching package: 'zoo'
    
    The following objects are masked from 'package:base':
    
        as.Date, as.Date.numeric
    
    Loading required package: lattice
    Type 'citation("pROC")' for a citation.
    
    Attaching package: 'pROC'
    
    The following objects are masked from 'package:stats':
    
        cov, smooth, var
    
    Loading required package: Matrix
    Loading required package: foreach
    Loaded glmnet 2.0-16
    
    
    Attaching package: 'glmnet'
    
    The following object is masked from 'package:pROC':
    
        auc
    
    Loading required package: colorspace
    
    Attaching package: 'colorspace'
    
    The following object is masked from 'package:pROC':
    
        coords
    
    Loading required package: grid
    Loading required package: data.table
    VIM is ready to use. 
     Since version 4.0.0 the GUI is in its own package VIMGUI.
    
              Please use the package to use the new (and old) GUI.
    
    Suggestions and bug-reports can be submitted at: https://github.com/alexkowa/VIM/issues
    
    Attaching package: 'VIM'
    
    The following object is masked from 'package:datasets':
    
        sleep
    
    


```R
set.seed(123) #For reproducibility
```

# The Data

The dataset obtained from a prostate cancer lab containing 3000 observations with 10 numerical features. In addition, there is a column showing the result of the test (class attribute), which is 0 for curable, and 1 for tumour stage, 2 for node stage and 3 for incurable cancers. All the features are numerical measurements rounded to the closest integer number between 1 and 10.


```R
df <- read.csv('data3000Final.csv',header = TRUE, sep = ',') #load the dataset
```


```R
dim(df) #We have 3000 records with 12 attributes.
```


<ol class=list-inline>
	<li>3000</li>
	<li>12</li>
</ol>




```R
head(df) # show the top 6 rows of the dataset
```


<table>
<thead><tr><th scope=col>ID</th><th scope=col>ATT1</th><th scope=col>ATT2</th><th scope=col>ATT3</th><th scope=col>ATT4</th><th scope=col>ATT5</th><th scope=col>ATT6</th><th scope=col>ATT7</th><th scope=col>ATT8</th><th scope=col>ATT9</th><th scope=col>ATT10</th><th scope=col>Result</th></tr></thead>
<tbody>
	<tr><td>1 </td><td>1 </td><td>4 </td><td>1 </td><td>4 </td><td>3 </td><td>7 </td><td> 1</td><td>2 </td><td>6 </td><td>8 </td><td>0 </td></tr>
	<tr><td>2 </td><td>? </td><td>8 </td><td>9 </td><td>1 </td><td>1 </td><td>1 </td><td> 1</td><td>5 </td><td>6 </td><td>1 </td><td>1 </td></tr>
	<tr><td>3 </td><td>10</td><td>7 </td><td>? </td><td>7 </td><td>? </td><td>5 </td><td> 2</td><td>7 </td><td>1 </td><td>1 </td><td>2 </td></tr>
	<tr><td>4 </td><td>3 </td><td>4 </td><td>3 </td><td>? </td><td>2 </td><td>8 </td><td> 4</td><td>6 </td><td>7 </td><td>2 </td><td>1 </td></tr>
	<tr><td>5 </td><td>3 </td><td>5 </td><td>2 </td><td>1 </td><td>6 </td><td>5 </td><td> 3</td><td>1 </td><td>7 </td><td>1 </td><td>0 </td></tr>
	<tr><td>6 </td><td>2 </td><td>7 </td><td>3 </td><td>2 </td><td>1 </td><td>4 </td><td>10</td><td>3 </td><td>9 </td><td>5 </td><td>1 </td></tr>
</tbody>
</table>




```R
tail(df) # show the last 6 rows of the dataset
```


<table>
<thead><tr><th></th><th scope=col>ID</th><th scope=col>ATT1</th><th scope=col>ATT2</th><th scope=col>ATT3</th><th scope=col>ATT4</th><th scope=col>ATT5</th><th scope=col>ATT6</th><th scope=col>ATT7</th><th scope=col>ATT8</th><th scope=col>ATT9</th><th scope=col>ATT10</th><th scope=col>Result</th></tr></thead>
<tbody>
	<tr><th scope=row>2995</th><td>2995</td><td>2   </td><td>6   </td><td>1   </td><td>2   </td><td>1   </td><td>1   </td><td>1   </td><td> 2  </td><td>2   </td><td>2   </td><td>0   </td></tr>
	<tr><th scope=row>2996</th><td>2996</td><td>4   </td><td>8   </td><td>6   </td><td>7   </td><td>2   </td><td>2   </td><td>4   </td><td> 1  </td><td>7   </td><td>8   </td><td>1   </td></tr>
	<tr><th scope=row>2997</th><td>2997</td><td>4   </td><td>1   </td><td>7   </td><td>6   </td><td>8   </td><td>1   </td><td>6   </td><td> 9  </td><td>5   </td><td>2   </td><td>1   </td></tr>
	<tr><th scope=row>2998</th><td>2998</td><td>6   </td><td>5   </td><td>8   </td><td>5   </td><td>1   </td><td>5   </td><td>2   </td><td> 7  </td><td>8   </td><td>7   </td><td>2   </td></tr>
	<tr><th scope=row>2999</th><td>2999</td><td>2   </td><td>4   </td><td>6   </td><td>4   </td><td>1   </td><td>2   </td><td>7   </td><td>10  </td><td>2   </td><td>2   </td><td>0   </td></tr>
	<tr><th scope=row>3000</th><td>3000</td><td>1   </td><td>8   </td><td>9   </td><td>4   </td><td>4   </td><td>3   </td><td>6   </td><td> 5  </td><td>5   </td><td>6   </td><td>2   </td></tr>
</tbody>
</table>




```R
summary(df) # summerize data
```


           ID              ATT1          ATT2             ATT3          ATT4    
     Min.   :   1.0   1      :517   Min.   : 1.000   1      :507   1      :497  
     1st Qu.: 750.8   2      :437   1st Qu.: 2.000   2      :432   2      :436  
     Median :1500.5   3      :372   Median : 4.000   3      :401   3      :370  
     Mean   :1500.5   4      :333   Mean   : 4.501   4      :354   4      :353  
     3rd Qu.:2250.2   5      :322   3rd Qu.: 7.000   6      :290   5      :301  
     Max.   :3000.0   6      :258   Max.   :10.000   5      :258   6      :295  
                      (Other):761                    (Other):758   (Other):748  
          ATT5          ATT6             ATT7             ATT8       
     1      :517   Min.   : 1.000   Min.   : 1.000   Min.   : 1.000  
     2      :448   1st Qu.: 2.000   1st Qu.: 2.000   1st Qu.: 2.000  
     3      :402   Median : 4.000   Median : 4.000   Median : 4.000  
     4      :359   Mean   : 4.431   Mean   : 4.479   Mean   : 4.478  
     5      :306   3rd Qu.: 6.000   3rd Qu.: 7.000   3rd Qu.: 7.000  
     7      :252   Max.   :10.000   Max.   :10.000   Max.   :10.000  
     (Other):716                                                     
          ATT9            ATT10            Result     
     Min.   : 1.000   Min.   : 1.000   Min.   :0.000  
     1st Qu.: 2.000   1st Qu.: 2.000   1st Qu.:0.000  
     Median : 4.000   Median : 4.000   Median :1.000  
     Mean   : 4.446   Mean   : 4.459   Mean   :0.984  
     3rd Qu.: 6.000   3rd Qu.: 7.000   3rd Qu.:2.000  
     Max.   :10.000   Max.   :10.000   Max.   :3.000  
                                                      



```R
str(df) #structure of the data
```

    'data.frame':	3000 obs. of  12 variables:
     $ ID    : int  1 2 3 4 5 6 7 8 9 10 ...
     $ ATT1  : Factor w/ 11 levels "?","1","10","2",..: 2 1 3 5 5 4 2 10 4 2 ...
     $ ATT2  : int  4 8 7 4 5 7 1 5 9 1 ...
     $ ATT3  : Factor w/ 11 levels "?","1","10","2",..: 2 11 1 5 4 5 5 4 4 2 ...
     $ ATT4  : Factor w/ 11 levels "?","1","10","2",..: 6 2 9 1 2 4 7 8 2 10 ...
     $ ATT5  : Factor w/ 11 levels "?","1","10","2",..: 5 2 1 4 8 2 11 4 6 11 ...
     $ ATT6  : int  7 1 5 8 5 4 3 7 6 2 ...
     $ ATT7  : int  1 1 2 4 3 10 7 3 8 2 ...
     $ ATT8  : int  2 5 7 6 1 3 1 6 8 5 ...
     $ ATT9  : int  6 6 1 7 7 9 10 7 4 1 ...
     $ ATT10 : int  8 1 1 2 1 5 7 3 3 3 ...
     $ Result: int  0 1 2 1 0 1 1 1 1 0 ...
    

Since ID does not contain any useful information, I have decided to drop the column from the data frame.


```R
df$ID <- NULL #drop ID column
```

From the structure of the data, we can see there are 4 missing values in attribute 1,3,4,5, which are factors. The rest of the columns contain only integers. Let's see the rows with missing data.


```R
df[df$ATT1 == "?" | df$ATT3== "?" | df$ATT4== "?" | df$ATT5== "?",] 
```


<table>
<thead><tr><th></th><th scope=col>ATT1</th><th scope=col>ATT2</th><th scope=col>ATT3</th><th scope=col>ATT4</th><th scope=col>ATT5</th><th scope=col>ATT6</th><th scope=col>ATT7</th><th scope=col>ATT8</th><th scope=col>ATT9</th><th scope=col>ATT10</th><th scope=col>Result</th></tr></thead>
<tbody>
	<tr><th scope=row>2</th><td>? </td><td>8 </td><td>9 </td><td>1 </td><td>1 </td><td>1 </td><td>1 </td><td>5 </td><td>6 </td><td>1 </td><td>1 </td></tr>
	<tr><th scope=row>3</th><td>10</td><td>7 </td><td>? </td><td>7 </td><td>? </td><td>5 </td><td>2 </td><td>7 </td><td>1 </td><td>1 </td><td>2 </td></tr>
	<tr><th scope=row>4</th><td>3 </td><td>4 </td><td>3 </td><td>? </td><td>2 </td><td>8 </td><td>4 </td><td>6 </td><td>7 </td><td>2 </td><td>1 </td></tr>
</tbody>
</table>



Row 2, 3 and 4 contains 4 missing values in 4 different attributes (1, 3, 4 and 5). Although result is a categorical variable, it will be left as a integer variable for EDA, then converted into a factor type

# 1. Filling missing values with regression:

There are 4 missing values in the dataset. Our first task is to estimate them by means of regression analysis. Using the remaining full observations, find regression models, and estimate the values of the missing components. 

We should remove the missing data and explore the dataset.


```R
df.completecase <- df[-c(2,3,4),] #Select all rows except 2, 3 and 4
```

Now we need to transform the factors into integers for attribute 1,3,4,5.


```R
df.completecase<-droplevels(df.completecase) #drop unused levels for factor types
df.completecase$ATT1 <- as.integer(as.character(df.completecase$ATT1)) #convert to integer for att1
df.completecase$ATT3 <- as.integer(as.character(df.completecase$ATT3)) #convert to integer for att3
df.completecase$ATT4 <- as.integer(as.character(df.completecase$ATT4)) #convert to integer for att4
df.completecase$ATT5 <- as.integer(as.character(df.completecase$ATT5)) #convert to integer for att5
```

Recheck the structure of the dataframe.


```R
str(df.completecase)
```

    'data.frame':	2997 obs. of  11 variables:
     $ ATT1  : int  1 3 2 1 8 2 1 9 6 5 ...
     $ ATT2  : int  4 5 7 1 5 9 1 3 2 2 ...
     $ ATT3  : int  1 2 3 3 2 2 1 5 2 8 ...
     $ ATT4  : int  4 1 2 5 6 1 8 7 1 5 ...
     $ ATT5  : int  3 6 1 9 2 4 9 7 1 8 ...
     $ ATT6  : int  7 5 4 3 7 6 2 5 2 1 ...
     $ ATT7  : int  1 3 10 7 3 8 2 2 4 3 ...
     $ ATT8  : int  2 1 3 1 6 8 5 3 8 6 ...
     $ ATT9  : int  6 7 9 10 7 4 1 2 6 2 ...
     $ ATT10 : int  8 1 5 7 3 3 3 6 10 1 ...
     $ Result: int  0 0 1 1 1 1 0 1 1 1 ...
    


```R
round(describe(df.completecase), 3) # view more detailed statistics and round some variables up to 3 d.p
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td> 1    </td><td>2997  </td><td>4.423 </td><td>2.721 </td><td>4     </td><td>4.228 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.429 </td><td>-0.960</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT2</th><td> 2    </td><td>2997  </td><td>4.499 </td><td>2.749 </td><td>4     </td><td>4.313 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.403 </td><td>-0.979</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT3</th><td> 3    </td><td>2997  </td><td>4.421 </td><td>2.713 </td><td>4     </td><td>4.226 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.434 </td><td>-0.963</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT4</th><td> 4    </td><td>2997  </td><td>4.440 </td><td>2.693 </td><td>4     </td><td>4.252 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.420 </td><td>-0.933</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT5</th><td> 5    </td><td>2997  </td><td>4.328 </td><td>2.669 </td><td>4     </td><td>4.124 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.481 </td><td>-0.877</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT6</th><td> 6    </td><td>2997  </td><td>4.431 </td><td>2.694 </td><td>4     </td><td>4.240 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.447 </td><td>-0.929</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT7</th><td> 7    </td><td>2997  </td><td>4.481 </td><td>2.741 </td><td>4     </td><td>4.300 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.392 </td><td>-1.036</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT8</th><td> 8    </td><td>2997  </td><td>4.476 </td><td>2.720 </td><td>4     </td><td>4.291 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.399 </td><td>-0.962</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT9</th><td> 9    </td><td>2997  </td><td>4.446 </td><td>2.679 </td><td>4     </td><td>4.254 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.429 </td><td>-0.900</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT10</th><td>10    </td><td>2997  </td><td>4.462 </td><td>2.697 </td><td>4     </td><td>4.276 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.407 </td><td>-0.940</td><td>0.049 </td></tr>
	<tr><th scope=row>Result</th><td>11    </td><td>2997  </td><td>0.984 </td><td>0.915 </td><td>1     </td><td>0.898 </td><td>1.483 </td><td>0     </td><td> 3    </td><td>3     </td><td>0.545 </td><td>-0.646</td><td>0.017 </td></tr>
</tbody>
</table>



All of the independent variables have very similar mean, standard deviations, kurtosis, skewness and range.

The result contains values of 0 to 3 as expected, but it seems a large proportion of the data has result of 1 or 0.

We should look at more specific statistics for each class (result).


```R
df.Result.zero <- df.completecase[df.completecase$Result == 0,] #subset data where Result == 0
```


```R
round(describe(df.Result.zero), 3) # view more detailed statistics and round some variables up to 3 d.p
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td> 1    </td><td>1082  </td><td>3.288 </td><td>2.274 </td><td>3     </td><td>2.965 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.988 </td><td> 0.230</td><td>0.069 </td></tr>
	<tr><th scope=row>ATT2</th><td> 2    </td><td>1082  </td><td>3.375 </td><td>2.339 </td><td>3     </td><td>3.052 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.959 </td><td> 0.124</td><td>0.071 </td></tr>
	<tr><th scope=row>ATT3</th><td> 3    </td><td>1082  </td><td>3.320 </td><td>2.270 </td><td>3     </td><td>3.018 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.928 </td><td> 0.023</td><td>0.069 </td></tr>
	<tr><th scope=row>ATT4</th><td> 4    </td><td>1082  </td><td>3.411 </td><td>2.279 </td><td>3     </td><td>3.127 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.846 </td><td>-0.144</td><td>0.069 </td></tr>
	<tr><th scope=row>ATT5</th><td> 5    </td><td>1082  </td><td>3.231 </td><td>2.170 </td><td>3     </td><td>2.925 </td><td>1.483 </td><td>1     </td><td>10    </td><td>9     </td><td>1.007 </td><td> 0.320</td><td>0.066 </td></tr>
	<tr><th scope=row>ATT6</th><td> 6    </td><td>1082  </td><td>3.414 </td><td>2.268 </td><td>3     </td><td>3.129 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.844 </td><td>-0.156</td><td>0.069 </td></tr>
	<tr><th scope=row>ATT7</th><td> 7    </td><td>1082  </td><td>3.380 </td><td>2.315 </td><td>3     </td><td>3.082 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.904 </td><td>-0.062</td><td>0.070 </td></tr>
	<tr><th scope=row>ATT8</th><td> 8    </td><td>1082  </td><td>3.460 </td><td>2.301 </td><td>3     </td><td>3.172 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.847 </td><td>-0.061</td><td>0.070 </td></tr>
	<tr><th scope=row>ATT9</th><td> 9    </td><td>1082  </td><td>3.361 </td><td>2.271 </td><td>3     </td><td>3.055 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.948 </td><td> 0.205</td><td>0.069 </td></tr>
	<tr><th scope=row>ATT10</th><td>10    </td><td>1082  </td><td>3.359 </td><td>2.306 </td><td>3     </td><td>3.045 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.908 </td><td>-0.029</td><td>0.070 </td></tr>
	<tr><th scope=row>Result</th><td>11    </td><td>1082  </td><td>0.000 </td><td>0.000 </td><td>0     </td><td>0.000 </td><td>0.000 </td><td>0     </td><td> 0    </td><td>0     </td><td>  NaN </td><td>   NaN</td><td>0.000 </td></tr>
</tbody>
</table>




```R
df.Result.one <- df.completecase[df.completecase$Result == 1,] #subset data where Result == 1
```


```R
round(describe(df.Result.one[1:10,]), 3) # view more detailed statistics and round some variables up to 3 d.p
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td> 1    </td><td>10    </td><td>4.3   </td><td>2.908 </td><td>4.0   </td><td>4.125 </td><td>2.965 </td><td>1     </td><td> 9    </td><td>8     </td><td> 0.269</td><td>-1.618</td><td>0.920 </td></tr>
	<tr><th scope=row>ATT2</th><td> 2    </td><td>10    </td><td>4.0   </td><td>2.789 </td><td>2.5   </td><td>3.750 </td><td>1.483 </td><td>1     </td><td> 9    </td><td>8     </td><td> 0.553</td><td>-1.456</td><td>0.882 </td></tr>
	<tr><th scope=row>ATT3</th><td> 3    </td><td>10    </td><td>4.4   </td><td>2.951 </td><td>3.0   </td><td>4.000 </td><td>1.483 </td><td>2     </td><td>10    </td><td>8     </td><td> 0.697</td><td>-1.237</td><td>0.933 </td></tr>
	<tr><th scope=row>ATT4</th><td> 4    </td><td>10    </td><td>4.1   </td><td>2.644 </td><td>5.0   </td><td>4.000 </td><td>3.706 </td><td>1     </td><td> 8    </td><td>7     </td><td>-0.032</td><td>-1.744</td><td>0.836 </td></tr>
	<tr><th scope=row>ATT5</th><td> 5    </td><td>10    </td><td>4.6   </td><td>2.875 </td><td>4.5   </td><td>4.500 </td><td>3.706 </td><td>1     </td><td> 9    </td><td>8     </td><td> 0.109</td><td>-1.634</td><td>0.909 </td></tr>
	<tr><th scope=row>ATT6</th><td> 6    </td><td>10    </td><td>3.8   </td><td>2.098 </td><td>3.5   </td><td>3.750 </td><td>2.224 </td><td>1     </td><td> 7    </td><td>6     </td><td> 0.172</td><td>-1.723</td><td>0.663 </td></tr>
	<tr><th scope=row>ATT7</th><td> 7    </td><td>10    </td><td>4.5   </td><td>2.953 </td><td>3.5   </td><td>4.250 </td><td>2.224 </td><td>1     </td><td>10    </td><td>9     </td><td> 0.559</td><td>-1.235</td><td>0.934 </td></tr>
	<tr><th scope=row>ATT8</th><td> 8    </td><td>10    </td><td>5.3   </td><td>3.057 </td><td>6.0   </td><td>5.375 </td><td>3.706 </td><td>1     </td><td> 9    </td><td>8     </td><td>-0.255</td><td>-1.755</td><td>0.967 </td></tr>
	<tr><th scope=row>ATT9</th><td> 9    </td><td>10    </td><td>5.2   </td><td>2.860 </td><td>5.0   </td><td>5.000 </td><td>3.706 </td><td>2     </td><td>10    </td><td>8     </td><td> 0.307</td><td>-1.405</td><td>0.904 </td></tr>
	<tr><th scope=row>ATT10</th><td>10    </td><td>10    </td><td>5.4   </td><td>2.591 </td><td>5.5   </td><td>5.375 </td><td>2.224 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.018</td><td>-0.982</td><td>0.819 </td></tr>
	<tr><th scope=row>Result</th><td>11    </td><td>10    </td><td>1.0   </td><td>0.000 </td><td>1.0   </td><td>1.000 </td><td>0.000 </td><td>1     </td><td> 1    </td><td>0     </td><td>   NaN</td><td>   NaN</td><td>0.000 </td></tr>
</tbody>
</table>




```R
df.Result.two <- df.completecase[df.completecase$Result == 2,] #subset data where Result == 2
```


```R
round(describe(df.Result.two), 3) # view more detailed statistics and round some variables up to 3 d.p
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td> 1    </td><td>641   </td><td>5.576 </td><td>2.737 </td><td>6     </td><td>5.628 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.134</td><td>-1.099</td><td>0.108 </td></tr>
	<tr><th scope=row>ATT2</th><td> 2    </td><td>641   </td><td>5.505 </td><td>2.746 </td><td>6     </td><td>5.526 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.060</td><td>-1.149</td><td>0.108 </td></tr>
	<tr><th scope=row>ATT3</th><td> 3    </td><td>641   </td><td>5.537 </td><td>2.747 </td><td>6     </td><td>5.538 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.010</td><td>-1.174</td><td>0.108 </td></tr>
	<tr><th scope=row>ATT4</th><td> 4    </td><td>641   </td><td>5.402 </td><td>2.665 </td><td>6     </td><td>5.398 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td> 0.006</td><td>-1.087</td><td>0.105 </td></tr>
	<tr><th scope=row>ATT5</th><td> 5    </td><td>641   </td><td>5.431 </td><td>2.698 </td><td>6     </td><td>5.448 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.051</td><td>-1.120</td><td>0.107 </td></tr>
	<tr><th scope=row>ATT6</th><td> 6    </td><td>641   </td><td>5.515 </td><td>2.745 </td><td>6     </td><td>5.515 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td> 0.015</td><td>-1.176</td><td>0.108 </td></tr>
	<tr><th scope=row>ATT7</th><td> 7    </td><td>641   </td><td>5.457 </td><td>2.757 </td><td>6     </td><td>5.478 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.043</td><td>-1.198</td><td>0.109 </td></tr>
	<tr><th scope=row>ATT8</th><td> 8    </td><td>641   </td><td>5.225 </td><td>2.739 </td><td>5     </td><td>5.218 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td> 0.031</td><td>-1.179</td><td>0.108 </td></tr>
	<tr><th scope=row>ATT9</th><td> 9    </td><td>641   </td><td>5.343 </td><td>2.691 </td><td>5     </td><td>5.326 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td> 0.072</td><td>-1.048</td><td>0.106 </td></tr>
	<tr><th scope=row>ATT10</th><td>10    </td><td>641   </td><td>5.746 </td><td>2.624 </td><td>6     </td><td>5.778 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.071</td><td>-1.003</td><td>0.104 </td></tr>
	<tr><th scope=row>Result</th><td>11    </td><td>641   </td><td>2.000 </td><td>0.000 </td><td>2     </td><td>2.000 </td><td>0.000 </td><td>2     </td><td> 2    </td><td>0     </td><td>   NaN</td><td>   NaN</td><td>0.000 </td></tr>
</tbody>
</table>




```R
df.Result.three <- df.completecase[df.completecase$Result == 3,] #subset data where Result == 3
```


```R
round(describe(df.Result.three), 3) # view more detailed statistics and round some variables up to 3 d.p
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td> 1    </td><td>196   </td><td>6.526 </td><td>2.581 </td><td>7     </td><td>6.665 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.368</td><td>-0.875</td><td>0.184 </td></tr>
	<tr><th scope=row>ATT2</th><td> 2    </td><td>196   </td><td>6.500 </td><td>2.628 </td><td>7     </td><td>6.639 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.357</td><td>-0.945</td><td>0.188 </td></tr>
	<tr><th scope=row>ATT3</th><td> 3    </td><td>196   </td><td>6.485 </td><td>2.849 </td><td>7     </td><td>6.684 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.497</td><td>-0.985</td><td>0.204 </td></tr>
	<tr><th scope=row>ATT4</th><td> 4    </td><td>196   </td><td>6.811 </td><td>2.518 </td><td>7     </td><td>7.019 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.553</td><td>-0.648</td><td>0.180 </td></tr>
	<tr><th scope=row>ATT5</th><td> 5    </td><td>196   </td><td>6.480 </td><td>2.553 </td><td>7     </td><td>6.633 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.426</td><td>-0.809</td><td>0.182 </td></tr>
	<tr><th scope=row>ATT6</th><td> 6    </td><td>196   </td><td>6.138 </td><td>2.844 </td><td>6     </td><td>6.241 </td><td>4.448 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.254</td><td>-1.223</td><td>0.203 </td></tr>
	<tr><th scope=row>ATT7</th><td> 7    </td><td>196   </td><td>6.372 </td><td>2.783 </td><td>7     </td><td>6.532 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.459</td><td>-1.038</td><td>0.199 </td></tr>
	<tr><th scope=row>ATT8</th><td> 8    </td><td>196   </td><td>6.668 </td><td>2.690 </td><td>7     </td><td>6.905 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.600</td><td>-0.683</td><td>0.192 </td></tr>
	<tr><th scope=row>ATT9</th><td> 9    </td><td>196   </td><td>6.531 </td><td>2.595 </td><td>7     </td><td>6.690 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.394</td><td>-0.834</td><td>0.185 </td></tr>
	<tr><th scope=row>ATT10</th><td>10    </td><td>196   </td><td>6.352 </td><td>2.628 </td><td>7     </td><td>6.462 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>-0.285</td><td>-1.147</td><td>0.188 </td></tr>
	<tr><th scope=row>Result</th><td>11    </td><td>196   </td><td>3.000 </td><td>0.000 </td><td>3     </td><td>3.000 </td><td>0.000 </td><td>3     </td><td> 3    </td><td>0     </td><td>   NaN</td><td>   NaN</td><td>0.000 </td></tr>
</tbody>
</table>



From above, we can see result 0 and 1 have similar number of points with different means for the same attribute. The skew of result 0 is larger and kurtosis is larger for result 1.

Result 2 and 3 have smaller number of points with result 3 having the smallest number of datapoints. This means the dataset have an inbalanced class distribution. This can make it difficuit to identify classes correctly especially result 3 has similar attributes to result 2.

<h1>Visual Representation of class distribution </h1>

The next step is to produce visual representation of the distribution of the classes. Since we have 10 attributes, this can make it difficuit to visualize unless we use principal component analysis to plot it in 2 dimensions.


```R
# Use PCA to obtain PC1 and PC2
p.comp<-prcomp(df.completecase[1:10])
# stores the results for plotting
post <- matrix( NA, nrow=2997, ncol=4)
# for graphing, requires to have a binary matrix to indicate result class for each observation
for (k in 1:2997){
    for (j in 1:4){
         post[k,j] <- ifelse((df.completecase$Result[k] == (j-1)),1,0)  
    }
}
```


```R
# graph the results using PCA
ggplot(data=as.data.frame(p.comp$x), aes(x = PC1, y = PC2)) + 
        geom_point(color = rgb(post), alpha=0.75) + theme_minimal()
```




![png](https://lawko698.github.io/assets/images/Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_36_1.png)


We can see that each class is mostly contained within their own area, but there are instances it goes into another area dominated by another class.
In addition, Result 2 (Blue) and 3 (Black) are very close to each other, nearly mixing into each other. This class inbalance can pose problems of classification later on.

<h1>Visual Representation for quantitative data: Boxplots and Histograms </h1>

The next step is to produce visual representation of the distribution of the quantitative variables.


```R
par(mfrow = c(1,5)) # display boxplots in a 1 x 5 grid
# for each numerical variable plot a boxplot
for (i in 1:11) {
        boxplot(df.completecase[,i], main = names(df.completecase[i]), type="l", col = 'lightblue')
}
```


![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_39_0.png)



![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_39_1.png)



![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_39_2.png)


The variables seem to be distributed well but we should plot histograms to understand the distribution more clearly.


```R
#ATT1 histogram
p1<-ggplot(df.completecase,aes(x = ATT1)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT1 distribution')

#ATT2 histogram
p2<-ggplot(df.completecase,aes(x = ATT2)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT2 distribution')

#ATT3 histogram
p3<-ggplot(df.completecase,aes(x = ATT3)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT3 distribution')

#ATT4 histogram
p4<-ggplot(df.completecase,aes(x = ATT4)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT4 distribution')

#ATT5 histogram
p5<-ggplot(df.completecase,aes(x = ATT5)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT5 distribution')

#ATT6 histogram
p6<-ggplot(df.completecase,aes(x = ATT6)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT6 distribution')

#ATT7 histogram
p7<-ggplot(df.completecase,aes(x = ATT7)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT7 distribution')

#ATT8 histogram
p8<-ggplot(df.completecase,aes(x = ATT8)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT8 distribution')

#ATT9 histogram
p9<-ggplot(df.completecase,aes(x = ATT9)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT9 distribution')

#ATT10 histogram
p10<-ggplot(df.completecase,aes(x = ATT10)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('ATT10 distribution')

#Result histogram
p11<-ggplot(df.completecase,aes(x = Result)) +
    geom_histogram(color = I('black'), fill = "blue") + 
    ggtitle('Result distribution')

# plot all 6 graphs, 3 x 2 grid
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, ncol = 2)
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    


![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_41_1.png)


From the histogram, we do not see a normal distribution in any 10 attributes. Interestingly, in attribute 1,3,4,5, as you move along the x-axis the number of points at value 3 dips quite sharply. Overall, there is a slow declining trend as you move along the x-axis. Since it is not heavily skewed, a log transformation will not convert it to a normal distribution.

From earlier, we can see result 0 and 1 are similar but 2 and 3 are smaller in number.

<h1>Association between variables: Correlation matrix</h1>

We can use [cor()](https://www.rdocumentation.org/packages/stats/versions/3.4.1/topics/cor) function to find the pearson correlation coefficient between numeric variables. 


```R
round(cor(df.completecase[-c(1)]),3) #pearson correlation by default  
```


<table>
<thead><tr><th></th><th scope=col>ATT2</th><th scope=col>ATT3</th><th scope=col>ATT4</th><th scope=col>ATT5</th><th scope=col>ATT6</th><th scope=col>ATT7</th><th scope=col>ATT8</th><th scope=col>ATT9</th><th scope=col>ATT10</th><th scope=col>Result</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT2</th><td>1.000</td><td>0.067</td><td>0.062</td><td>0.032</td><td>0.039</td><td>0.011</td><td>0.030</td><td>0.094</td><td>0.073</td><td>0.351</td></tr>
	<tr><th scope=row>ATT3</th><td>0.067</td><td>1.000</td><td>0.047</td><td>0.065</td><td>0.040</td><td>0.031</td><td>0.026</td><td>0.060</td><td>0.085</td><td>0.365</td></tr>
	<tr><th scope=row>ATT4</th><td>0.062</td><td>0.047</td><td>1.000</td><td>0.080</td><td>0.045</td><td>0.059</td><td>0.021</td><td>0.028</td><td>0.083</td><td>0.360</td></tr>
	<tr><th scope=row>ATT5</th><td>0.032</td><td>0.065</td><td>0.080</td><td>1.000</td><td>0.050</td><td>0.071</td><td>0.029</td><td>0.084</td><td>0.084</td><td>0.374</td></tr>
	<tr><th scope=row>ATT6</th><td>0.039</td><td>0.040</td><td>0.045</td><td>0.050</td><td>1.000</td><td>0.023</td><td>0.041</td><td>0.067</td><td>0.063</td><td>0.334</td></tr>
	<tr><th scope=row>ATT7</th><td>0.011</td><td>0.031</td><td>0.059</td><td>0.071</td><td>0.023</td><td>1.000</td><td>0.078</td><td>0.037</td><td>0.067</td><td>0.341</td></tr>
	<tr><th scope=row>ATT8</th><td>0.030</td><td>0.026</td><td>0.021</td><td>0.029</td><td>0.041</td><td>0.078</td><td>1.000</td><td>0.043</td><td>0.034</td><td>0.327</td></tr>
	<tr><th scope=row>ATT9</th><td>0.094</td><td>0.060</td><td>0.028</td><td>0.084</td><td>0.067</td><td>0.037</td><td>0.043</td><td>1.000</td><td>0.012</td><td>0.350</td></tr>
	<tr><th scope=row>ATT10</th><td>0.073</td><td>0.085</td><td>0.083</td><td>0.084</td><td>0.063</td><td>0.067</td><td>0.034</td><td>0.012</td><td>1.000</td><td>0.373</td></tr>
	<tr><th scope=row>Result</th><td>0.351</td><td>0.365</td><td>0.360</td><td>0.374</td><td>0.334</td><td>0.341</td><td>0.327</td><td>0.350</td><td>0.373</td><td>1.000</td></tr>
</tbody>
</table>



From above, we see very small correlation between any of the independent variables. Only variable Result between other independent variables have a weak linear correlation. We can visualize these correlations with a scatter graph.


```R
scatterplotMatrix(~ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + ATT8 + ATT9 + ATT10 + Result, df.completecase,cex=0.2)
```


![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_47_0.png)


The scatter graph confirms our correlation matrix, in which the data has no linear relationship between any independent variables which imply no multicollinearity. However, each independent variable has a linear relationship with the result variable.

Although most of the attribute variables are not linearly correlated with each other, we can use regression tree as an imputation method. 

We can use a regression tree to make predictions on those missing values due to its advantages such as:

- Does not require any statistical assumptions.
- Can operate on a non-linear and complex relationship between predictors and response variable.

However, regression trees may have disadvantages such as:
- high variance, a small change in data may result into different results
- if there is a linear relationship between the predictor and response, it is better off to use linear regression


```R
#convert Results variable into a factor type
df.completecase$Result <- factor(df.completecase$Result)
```

The random forest function from [randomForest](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) package produces an ensemble of trees to make a prediction of the response variable. Although bootstrap aggregation of trees is a special version of random forest, random forest improves upon bagged trees by decorrelating the generate trees.

The randomForest function operates by:
- Training set for the current tree is drawn by sampling with replacement, about one-third of the cases are left out of the sample. This oob (out-of-bag) data is used to get a running unbiased estimate of the regression error as trees are added to the forest. The predicted values of the input data is based on out-of-bag samples.


```R
#generate a random forest model that predicts attribute 1
forest_att1 <- randomForest(ATT1 ~ ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + 
                            ATT8 + ATT9 + ATT10, data = df.completecase, mtry = 10, ntree = 500, importance = TRUE)
forest_att1 #output the model
```

    Warning message in randomForest.default(m, y, ...):
    "invalid mtry: reset to within valid range"


    
    Call:
     randomForest(formula = ATT1 ~ ATT2 + ATT3 + ATT4 + ATT5 + ATT6 +      ATT7 + ATT8 + ATT9 + ATT10, data = df.completecase, mtry = 10,      ntree = 500, importance = TRUE) 
                   Type of random forest: regression
                         Number of trees: 500
    No. of variables tried at each split: 9
    
              Mean of squared residuals: 7.566737
                        % Var explained: -2.22



```R
#look at which variable is important to the model
importance(forest_att1)
```


<table>
<thead><tr><th></th><th scope=col>%IncMSE</th><th scope=col>IncNodePurity</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT2</th><td>0.7335200</td><td>2308.859 </td></tr>
	<tr><th scope=row>ATT3</th><td>4.4306467</td><td>2119.311 </td></tr>
	<tr><th scope=row>ATT4</th><td>0.4923638</td><td>2342.945 </td></tr>
	<tr><th scope=row>ATT5</th><td>1.4583184</td><td>2350.110 </td></tr>
	<tr><th scope=row>ATT6</th><td>2.4575233</td><td>2255.329 </td></tr>
	<tr><th scope=row>ATT7</th><td>2.3674374</td><td>2142.071 </td></tr>
	<tr><th scope=row>ATT8</th><td>1.4721853</td><td>2348.368 </td></tr>
	<tr><th scope=row>ATT9</th><td>0.8466364</td><td>2366.720 </td></tr>
	<tr><th scope=row>ATT10</th><td>1.5278499</td><td>2292.749 </td></tr>
</tbody>
</table>




```R
#calculate rmse on random forest 
rmse(round(forest_att1$predicted), df.completecase$ATT1)
```


2.76531739835538


Random forest's % variance explained tells us it shouldn't be used for imputation as the rest of the attributes does not a good job at explaining ATT1. Therefore, we need another method to impute our missing values. 

We can use kNN (K Nearest Neighbours) to find similar datapoints to those with missing attributes, and impute them according to their neighbour's attributes.


```R
# Change ? values into NA
df[df$ATT1 == "?",'ATT1'] <- NA
df[df$ATT3 == "?",'ATT3'] <- NA
df[df$ATT4 == "?",'ATT4'] <- NA
df[df$ATT5 == "?",'ATT5'] <- NA
```


```R
# find k = 21 neighbours for each data point with missing value(s).
df <- kNN(df, variable = c('ATT1', 'ATT3', 'ATT4', 'ATT5'), k = 21)
```


```R
# Imputation results
df[c(2,3,4),]
```


<table>
<thead><tr><th></th><th scope=col>ATT1</th><th scope=col>ATT2</th><th scope=col>ATT3</th><th scope=col>ATT4</th><th scope=col>ATT5</th><th scope=col>ATT6</th><th scope=col>ATT7</th><th scope=col>ATT8</th><th scope=col>ATT9</th><th scope=col>ATT10</th><th scope=col>Result</th><th scope=col>ATT1_imp</th><th scope=col>ATT3_imp</th><th scope=col>ATT4_imp</th><th scope=col>ATT5_imp</th></tr></thead>
<tbody>
	<tr><th scope=row>2</th><td>9    </td><td>8    </td><td>9    </td><td>1    </td><td>1    </td><td>1    </td><td>1    </td><td>5    </td><td>6    </td><td>1    </td><td>1    </td><td> TRUE</td><td>FALSE</td><td>FALSE</td><td>FALSE</td></tr>
	<tr><th scope=row>3</th><td>10   </td><td>7    </td><td>5    </td><td>7    </td><td>5    </td><td>5    </td><td>2    </td><td>7    </td><td>1    </td><td>1    </td><td>2    </td><td>FALSE</td><td> TRUE</td><td>FALSE</td><td> TRUE</td></tr>
	<tr><th scope=row>4</th><td>3    </td><td>4    </td><td>3    </td><td>5    </td><td>2    </td><td>8    </td><td>4    </td><td>6    </td><td>7    </td><td>2    </td><td>1    </td><td>FALSE</td><td>FALSE</td><td> TRUE</td><td>FALSE</td></tr>
</tbody>
</table>




```R
# Remove the last four columns
df.imputed <- df[-c(12,13,14,15)]
```


```R
df.imputed$ATT1 <- as.integer(as.character(df.imputed$ATT1))
df.imputed$ATT3 <- as.integer(as.character(df.imputed$ATT3))
df.imputed$ATT4 <- as.integer(as.character(df.imputed$ATT4))
df.imputed$ATT5 <- as.integer(as.character(df.imputed$ATT5))

# PCA for graphing reference
p.comp.imputed<-prcomp(df.imputed[1:10])
post <- matrix( NA, nrow=3000, ncol=4)
for (k in 1:3000){
    for (j in 1:4){
      post[k,j] <- ifelse((df.imputed$Result[k] == (j-1)),1,0)  
    }
}

df.imputed$Result <- factor(as.character(df.imputed$Result)) # convert result into factor
str(df.imputed) # recheck the structure 
round(describe(df.imputed), 3) # view more detailed statistics and round some variables up to 3 d.p
```

    'data.frame':	3000 obs. of  11 variables:
     $ ATT1  : int  1 9 10 3 3 2 1 8 2 1 ...
     $ ATT2  : int  4 8 7 4 5 7 1 5 9 1 ...
     $ ATT3  : int  1 9 5 3 2 3 3 2 2 1 ...
     $ ATT4  : int  4 1 7 5 1 2 5 6 1 8 ...
     $ ATT5  : int  3 1 5 2 6 1 9 2 4 9 ...
     $ ATT6  : int  7 1 5 8 5 4 3 7 6 2 ...
     $ ATT7  : int  1 1 2 4 3 10 7 3 8 2 ...
     $ ATT8  : int  2 5 7 6 1 3 1 6 8 5 ...
     $ ATT9  : int  6 6 1 7 7 9 10 7 4 1 ...
     $ ATT10 : int  8 1 1 2 1 5 7 3 3 3 ...
     $ Result: Factor w/ 4 levels "0","1","2","3": 1 2 3 2 1 2 2 2 2 1 ...
    


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td> 1    </td><td>3000  </td><td>4.426 </td><td>2.723 </td><td>4     </td><td>4.231 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.429 </td><td>-0.962</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT2</th><td> 2    </td><td>3000  </td><td>4.501 </td><td>2.748 </td><td>4     </td><td>4.315 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.402 </td><td>-0.980</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT3</th><td> 3    </td><td>3000  </td><td>4.422 </td><td>2.713 </td><td>4     </td><td>4.228 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.434 </td><td>-0.964</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT4</th><td> 4    </td><td>3000  </td><td>4.440 </td><td>2.693 </td><td>4     </td><td>4.252 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.419 </td><td>-0.934</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT5</th><td> 5    </td><td>3000  </td><td>4.326 </td><td>2.668 </td><td>4     </td><td>4.122 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.481 </td><td>-0.876</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT6</th><td> 6    </td><td>3000  </td><td>4.431 </td><td>2.694 </td><td>4     </td><td>4.240 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.446 </td><td>-0.930</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT7</th><td> 7    </td><td>3000  </td><td>4.479 </td><td>2.741 </td><td>4     </td><td>4.297 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.393 </td><td>-1.035</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT8</th><td> 8    </td><td>3000  </td><td>4.478 </td><td>2.720 </td><td>4     </td><td>4.292 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.398 </td><td>-0.962</td><td>0.050 </td></tr>
	<tr><th scope=row>ATT9</th><td> 9    </td><td>3000  </td><td>4.446 </td><td>2.678 </td><td>4     </td><td>4.254 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.428 </td><td>-0.901</td><td>0.049 </td></tr>
	<tr><th scope=row>ATT10</th><td>10    </td><td>3000  </td><td>4.459 </td><td>2.698 </td><td>4     </td><td>4.271 </td><td>2.965 </td><td>1     </td><td>10    </td><td>9     </td><td>0.409 </td><td>-0.940</td><td>0.049 </td></tr>
	<tr><th scope=row>Result*</th><td>11    </td><td>3000  </td><td>1.984 </td><td>0.914 </td><td>2     </td><td>1.898 </td><td>1.483 </td><td>1     </td><td> 4    </td><td>3     </td><td>0.544 </td><td>-0.646</td><td>0.017 </td></tr>
</tbody>
</table>



# 2. Build a classifier

Our response variable has 4 classes:

- Result 0 as curable
- Result 1 as tumour stage
- Result 2 as node stage
- Result 3 as incurable

We need to use a multiclass classification model. Here are the following models used:
- Linear and Quadratic Discriminant Analysis
- Multinomial Logistic Regression
- Random Forest Classifier
- Support Vector Machine

## Reference to True model


```R
# Reference to True model
ggplot(data=as.data.frame(p.comp.imputed$x), aes(x = PC1, y = PC2)) + 
        geom_point(color = rgb(post), alpha=0.75)  +
        ggtitle ("True Model") + theme_minimal()
```




![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_62_1.png)


## Training and Test Data

Each model will be trained using the training dataset, while testing will be on the test dataset.


```R
# 50% split into train and 50% test dataset
# To ensure training and test set have equal number of classes, separate data in terms of class then sample
df.imputed.zero <- df.imputed[df.imputed$Result == 0,]
df.imputed.one <- df.imputed[df.imputed$Result == 1,]
df.imputed.two <- df.imputed[df.imputed$Result == 2,]
df.imputed.three <- df.imputed[df.imputed$Result == 3,]

# Sample from data when result = 0
indx <- sample(1:nrow(df.imputed.zero),nrow(df.imputed.zero)/2, replace=FALSE) #sample 50% of data
df.imputed.train <- df.imputed.zero[indx,] #generate training set
df.imputed.test <- df.imputed.zero[-indx,] #generate test set

# Sample from data when result = 1
indx <- sample(1:nrow(df.imputed.one),nrow(df.imputed.one)/2, replace=FALSE) #sample 50% of data
df.imputed.train <- rbind(df.imputed.one[indx,],df.imputed.train) #combine with previous training set
df.imputed.test <- rbind(df.imputed.one[-indx,],df.imputed.test) #combine with previous test set

# Sample from data when result = 2
indx <- sample(1:nrow(df.imputed.two),nrow(df.imputed.two)/2, replace=FALSE) #sample 50% of data
df.imputed.train <- rbind(df.imputed.two[indx,],df.imputed.train) #combine with previous training set
df.imputed.test <- rbind(df.imputed.two[-indx,],df.imputed.test) #combine with previous test set

# Sample from data when result = 3
indx <- sample(1:nrow(df.imputed.three),nrow(df.imputed.three)/2, replace=FALSE) #sample 50% of data
df.imputed.train <- rbind(df.imputed.three[indx,],df.imputed.train) #combine with previous training set
df.imputed.test <- rbind(df.imputed.three[-indx,],df.imputed.test) #combine with previous test set

```

<h1>Linear Discriminant Analysis (LDA) </h1>

We can use linear discriminant analysis as it is a popular multi-class classification model. If the classes are well separated the parameter estimates are stable. Otherwise, a logistic regression should be used. Although the classes do not seem to be perfectly separated as seen from the visualization, we can see how well LDA performs in sight of these problems to be compared with other methods.

Assumptions:
- Gaussian distributions of classes
- Classes have a common covariance matrix

I have used lda function from [MASS](https://cran.r-project.org/web/packages/MASS/MASS.pdf) package. It will build a model with cross validation by setting CV=TRUE to get predictions of class membership that are derived from leave-one-out cross-validation.


```R
fit.lda = lda(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + ATT8 + ATT9 + ATT10, 
              data = df.imputed, CV = TRUE)
```

Now we can evaluate the classification with leave-one-out cross-validation against the true labels. 


```R
# table the results into proportion
tab.lda <- table(df.imputed$Result, fit.lda$class)
lda.CV <- rbind(tab.lda[1, ]/sum(tab.lda[1, ]), tab.lda[2, ]/sum(tab.lda[2, ]),
                tab.lda[3, ]/sum(tab.lda[3, ]), tab.lda[4, ]/sum(tab.lda[4, ]))

#add row and column names
dimnames(lda.CV)<- list(Actual = c("curable", "tumour","node","incurable"), 
                  "Predicted (cv)" = c("curable", "tumour","node","incurable")) 
#print table
print(round(lda.CV, 3))
```

               Predicted (cv)
    Actual      curable tumour  node incurable
      curable     0.914  0.086 0.000     0.000
      tumour      0.000  0.996 0.004     0.000
      node        0.000  0.050 0.950     0.000
      incurable   0.000  0.000 0.189     0.811
    

From the table above, we can see a large proportion of misclassification of result 3 (incurable) compared to the others. It incorrectly predicts it as result 2 (node) since these two classes are very similar as seen in the visualization.

We should use AUC score to evaluate model prediction that accounts for different threshold. This provides a better evaluation criteria than accuracy (since it only evaluates at one threshold).


```R
multiclass.roc(df.imputed$Result, as.integer(fit.lda$class) ,levels=base::levels(df.imputed$Result), percent=FALSE)
```


    
    Call:
    multiclass.roc.default(response = df.imputed$Result, predictor = as.integer(fit.lda$class),     levels = base::levels(df.imputed$Result), percent = FALSE)
    
    Data: as.integer(fit.lda$class) with 4 levels of df.imputed$Result: 0, 1, 2, 3.
    Multi-class area under the curve: 0.973



```R
# To be used for plotting predicted results
post.lda <- matrix( NA, nrow=3000, ncol=4)
for (k in 1:3000){
    for (j in 1:4){
      post.lda[k,j] <- ifelse((fit.lda$class[k] == (j)),1,0)  
    }
}
```


```R
# plot the predicted results
ggplot(data=as.data.frame(p.comp.imputed$x),aes(x=PC1, y=PC2)) + 
geom_point(color = rgb(post.lda), alpha=0.75)+
ggtitle("LDA Model") + theme_minimal()
```




![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_75_1.png)


The AUC score is a healthy 0.973 and the graph shows it manages to classifiy most of the observations correctly. However, result 3 proves to be hard to classify it might not be linearly separatable. In addition, it has problems classifying points that spill into different regions between result 0 and 1, result 1 and 2.  

<h1>Quadratic Discriminant Analysis (QDA) </h1>

We made an assumption that our classes are linearly separatable decision boundary. However, we do not know if non-linear decision boundaries perform better. QDA uses a non-linear (quadratic) separatable decision boundary which may be a better candidate than LDA.

One difference between LDA is the assumption we need for QDA is that every class has a different covariance matrix.

QDA function performs similarly to LDA function from above.


```R
fit.qda = qda(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + ATT8 + ATT9 + ATT10, 
              data = df.imputed, CV = TRUE)
```

Now we can evaluate the classification with leave-one-out cross-validation against the true labels. 


```R
# table the proportions
tab.qda <- table(df.imputed$Result, fit.qda$class)
qda.CV <- rbind(tab.qda[1, ]/sum(tab.qda[1, ]), tab.qda[2, ]/sum(tab.qda[2, ]), tab.qda[3, ]/sum(tab.qda[3, ]),
                tab.qda[4, ]/sum(tab.qda[4, ]))

#label the row and columns
dimnames(qda.CV)<- list(Actual = c("curable", "tumour","node","incurable"), 
                  "Predicted (cv)" = c("curable", "tumour","node","incurable"))
#print table
print(round(qda.CV, 3))
```

               Predicted (cv)
    Actual      curable tumour  node incurable
      curable      0.97  0.030 0.000     0.000
      tumour       0.02  0.970 0.009     0.000
      node         0.00  0.023 0.969     0.008
      incurable    0.00  0.000 0.031     0.969
    

From the results above, there are some trade-offs between result 0 (curable) accuracy increase to 1 (tumour) accuracy decrease and result 2 (node) accuracy increase to 3 (incurable) accuracy decrease compared to LDA's results.


```R
multiclass.roc(df.imputed$Result, as.integer(fit.qda$class) ,levels=base::levels(df.imputed$Result), percent=FALSE)
```


    
    Call:
    multiclass.roc.default(response = df.imputed$Result, predictor = as.integer(fit.qda$class),     levels = base::levels(df.imputed$Result), percent = FALSE)
    
    Data: as.integer(fit.qda$class) with 4 levels of df.imputed$Result: 0, 1, 2, 3.
    Multi-class area under the curve: 0.9899



```R
# To be used for plotting predicted results
post.qda <- matrix( NA, nrow=3000, ncol=4)
for (k in 1:3000){
    for (j in 1:4){
      post.qda[k,j] <- ifelse((fit.lda$class[k] == (j)),1,0)  
    }
}
```


```R
# plot the predicted results
ggplot(data=as.data.frame(p.comp.imputed$x),aes(x=PC1, y=PC2)) + 
geom_point(color = rgb(post.qda), alpha=0.75)+
ggtitle("QDA Model") + theme_minimal()
```




![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_84_1.png)


The AUC score has improved to 0.9899 and the graph shows it manages to classifiy most of the observations correctly similar to LDA. QDA performs slightly better than QDA.

<h1>Multinomial Logistic Regression</h1>

We saw from LDA's classification that suggets the classes do not seem to be well separated. In addition, we have assumed the classes have a normal distribution. Violation of this assumption may bias the model's parameters. Therefore, we can use Multinomial Logisitic regression as it requires less assumptions and work better than LDA when classes are not well seperated.


```R
#create matrix of test data
test.xmat <- model.matrix(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + ATT8 + ATT9 + ATT10, 
                           data = df.imputed.test, family = 'multinomial')[,-1]
```


```R
#create matrix of train data
train.xmat <- model.matrix(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + ATT8 + ATT9 + ATT10, 
                           data = df.imputed.train, family = 'multinomial')[,-1]
```


```R
# Multinomial Logisitic regression
# cross-validation lasso regularization
# find the best lambda parameter
multilogit.cvfit <- cv.glmnet(train.xmat, df.imputed.train$Result, family="multinomial", alpha = 1)
```


```R
plot(multilogit.cvfit)
```


![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_90_0.png)



```R
# cross-validation to find best lamda for lasso regularization
bestlam.lasso <- multilogit.cvfit$lambda.min
```


```R
# make predictions on test set
multi.pred <- as.vector(as.integer(predict(multilogit.cvfit, test.xmat, s = bestlam.lasso, type = "class")))
```


```R
# table the proportions
tab.multi <- table(df.imputed.test$Result, multi.pred)
multi.CV <- rbind(tab.multi[1, ]/sum(tab.multi[1, ]), tab.multi[2, ]/sum(tab.multi[2, ]), tab.multi[3, ]/sum(tab.multi[3, ]),
                tab.multi[4, ]/sum(tab.multi[4, ]))

#label the row and columns
dimnames(multi.CV)<- list(Actual = c("curable", "tumour","node","incurable"), 
                  "Predicted (cv)" = c("curable", "tumour","node","incurable"))
# print table
print(round(multi.CV, 3))
```

               Predicted (cv)
    Actual      curable tumour  node incurable
      curable         1  0.000 0.000     0.000
      tumour          0  1.000 0.000     0.000
      node            0  0.003 0.988     0.009
      incurable       0  0.000 0.000     1.000
    

From the table above, we can see multinomial logistic regression is able to better predict result 3 compared to LDA and QDA. However, it misclassifies result 1 more than LDA and QDA. 


```R
multiclass.roc(df.imputed.test$Result, multi.pred, levels=base::levels(df.imputed.test$Result), percent=TRUE)
```


    
    Call:
    multiclass.roc.default(response = df.imputed.test$Result, predictor = multi.pred,     levels = base::levels(df.imputed.test$Result), percent = TRUE)
    
    Data: multi.pred with 4 levels of df.imputed.test$Result: 0, 1, 2, 3.
    Multi-class area under the curve: 99.9%


The AUC score is a healthy 99.9% which is a slight improvement upon QDA. This backs up the notion that the classes are well separated, where multinominal logisitic model performs better than LDA.

Note: The graph is not plotted since the PCA does not show the same shape previously shown due to sampling a portion of the original data.

<h1>Random Forest</h1>

Random Forest can also be used as a classification model instead of a regression model that was used in missing value imputation. The advantages and disadvantages of random forest regression are applied to the classifier model. Random forest provides an alternative to LDA, QDA and Multinominal logistic regression as it segments the predictor space into a number of simple regions, which works well with non-linear and complex relationship between the features.


```R
#generate a random forest model
rf.fit <- randomForest(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + 
                            ATT8 + ATT9 + ATT10, data = df.imputed, mtry = 10, ntree = 500, importance = TRUE)
rf.fit #output the model
```


    
    Call:
     randomForest(formula = Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 +      ATT6 + ATT7 + ATT8 + ATT9 + ATT10, data = df.imputed, mtry = 10,      ntree = 500, importance = TRUE) 
                   Type of random forest: classification
                         Number of trees: 500
    No. of variables tried at each split: 10
    
            OOB estimate of  error rate: 23.33%
    Confusion matrix:
        0   1   2  3 class.error
    0 964 118   0  0   0.1090573
    1 127 900  53  0   0.1666667
    2   0 256 384  2   0.4018692
    3   0   3 141 52   0.7346939


The random forest function uses out-of-bag error estimation. It shows a 23.3% error rate with the fitted model. Most of the errors are attributed to the classification of result 3, and and result 2 which is expected.


```R
#look at which variable is important to the model
importance(rf.fit)
```


<table>
<thead><tr><th></th><th scope=col>0</th><th scope=col>1</th><th scope=col>2</th><th scope=col>3</th><th scope=col>MeanDecreaseAccuracy</th><th scope=col>MeanDecreaseGini</th></tr></thead>
<tbody>
	<tr><th scope=row>ATT1</th><td>59.02299</td><td>33.49498</td><td>39.75936</td><td>26.56463</td><td>77.75822</td><td>197.1023</td></tr>
	<tr><th scope=row>ATT2</th><td>50.87716</td><td>30.72737</td><td>31.26563</td><td>22.23357</td><td>62.98704</td><td>211.7276</td></tr>
	<tr><th scope=row>ATT3</th><td>51.32102</td><td>30.46145</td><td>33.57914</td><td>20.88016</td><td>65.77979</td><td>212.7885</td></tr>
	<tr><th scope=row>ATT4</th><td>44.74751</td><td>26.84208</td><td>25.29077</td><td>30.95077</td><td>62.04120</td><td>202.7325</td></tr>
	<tr><th scope=row>ATT5</th><td>51.23424</td><td>27.38835</td><td>32.40236</td><td>28.98439</td><td>62.73447</td><td>205.4879</td></tr>
	<tr><th scope=row>ATT6</th><td>45.99043</td><td>23.97962</td><td>30.67019</td><td>14.31418</td><td>54.30447</td><td>204.2850</td></tr>
	<tr><th scope=row>ATT7</th><td>54.49510</td><td>28.43069</td><td>25.68874</td><td>18.98994</td><td>62.83274</td><td>208.8188</td></tr>
	<tr><th scope=row>ATT8</th><td>48.98404</td><td>30.94731</td><td>24.10396</td><td>27.82564</td><td>63.82490</td><td>215.0017</td></tr>
	<tr><th scope=row>ATT9</th><td>52.61634</td><td>27.25110</td><td>25.33940</td><td>24.80699</td><td>63.03772</td><td>208.7095</td></tr>
	<tr><th scope=row>ATT10</th><td>49.68486</td><td>31.14477</td><td>41.07969</td><td>20.68647</td><td>68.89368</td><td>202.3407</td></tr>
</tbody>
</table>



From the table above, there isn't a variable that significantly stands out as the most important one in the model, but rather every variable plays a role in the model.


```R
multiclass.roc(df.imputed$Result, as.vector(as.integer(rf.fit$predicted)),
               levels=base::levels(df.imputed$Result), percent=TRUE)
```


    
    Call:
    multiclass.roc.default(response = df.imputed$Result, predictor = as.vector(as.integer(rf.fit$predicted)),     levels = base::levels(df.imputed$Result), percent = TRUE)
    
    Data: as.vector(as.integer(rf.fit$predicted)) with 4 levels of df.imputed$Result: 0, 1, 2, 3.
    Multi-class area under the curve: 90.19%



```R
# To be used for plotting predicted results
post.rf <- matrix( NA, nrow=3000, ncol=4)
for (k in 1:3000){
    for (j in 1:4){
      post.rf[k,j] <- ifelse((rf.fit$predicted[k] == (j)),1,0)  
    }
}
```


```R
# plot the predicted results
ggplot(data=as.data.frame(p.comp.imputed$x),aes(x=PC1, y=PC2)) + 
geom_point(color = rgb(post.rf), alpha=0.75)+
ggtitle("Random Forest Model") + theme_minimal()
```




![png](Detecting%20Prostate%20Cancers_files/Detecting%20Prostate%20Cancers_104_1.png)


The AUC score is a decent 90.19% and the graph shows it manages to classifiy most of the observations correctly. However, its performance compared to the earlier models are a bit lacking. We can see result 3 proves to be hardest to classify as it mostly classifies it as result 2.

This tells us random forest regressions does not perform well on the data, where it segments the predictor space into a number of simple regions. Using a simple linear decision boundary is better. 

<h1>Support Vector Machine</h1>

Support Vector Machine is a classification method that can apply linear or non-linear decision boundaries. I have used package [e1071](https://cran.r-project.org/web/packages/e1071/index.html). It relies on less assumptions on the data compared to other methods such as LDA. In addition to linear decision boundaries, we can use a radial kernal that can encapsulate similar classes within a guassian/radial decision boundary. We can see if a non-linear (radial) decision boundary perform better than a linear one.


```R
# Use cross validation to find the best cost parameter (the price of the misclassification)
# Use Radial decision boundary
tune.out <- tune(svm, Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + 
                            ATT8 + ATT9 + ATT10, data = df.imputed.train, kernel = "radial", 
                 ranges = list(cost = c(10^seq(-2, 1, by = 0.25))))
summary(tune.out)
```


    
    Parameter tuning of 'svm':
    
    - sampling method: 10-fold cross validation 
    
    - best parameters:
         cost
     1.778279
    
    - best performance: 0.08733333 
    
    - Detailed performance results:
              cost      error dispersion
    1   0.01000000 0.58666667 0.09972803
    2   0.01778279 0.45933333 0.04739641
    3   0.03162278 0.39800000 0.04489219
    4   0.05623413 0.36000000 0.04776486
    5   0.10000000 0.25866667 0.04272219
    6   0.17782794 0.16400000 0.04288371
    7   0.31622777 0.12733333 0.02938884
    8   0.56234133 0.09866667 0.02824933
    9   1.00000000 0.09000000 0.02182987
    10  1.77827941 0.08733333 0.01871159
    11  3.16227766 0.09600000 0.02041544
    12  5.62341325 0.09400000 0.02361209
    13 10.00000000 0.09666667 0.02689715
    


From the cross-validation approach, it suggest a cost value of 1.778 should be used in the model.


```R
# fit the model with the best cost parameter
svm.fit <- svm(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + 
                            ATT8 + ATT9 + ATT10, df.imputed.train, kernel = "radial", cost = tune.out$best.parameter$cost)
# output a summary
summary(svm.fit)

# make a prediction on test dataset
svm.pred <- as.numeric(predict(svm.fit, df.imputed.test))
```


    
    Call:
    svm(formula = Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + 
        ATT7 + ATT8 + ATT9 + ATT10, data = df.imputed.train, kernel = "radial", 
        cost = tune.out$best.parameter$cost)
    
    
    Parameters:
       SVM-Type:  C-classification 
     SVM-Kernel:  radial 
           cost:  1.778279 
          gamma:  0.1 
    
    Number of Support Vectors:  791
    
     ( 71 247 323 150 )
    
    
    Number of Classes:  4 
    
    Levels: 
     0 1 2 3
    
    
    



```R
multiclass.roc(df.imputed.test$Result, as.vector(as.integer(svm.pred)),
               levels=base::levels(df.imputed.test$Result), percent=TRUE)
```


    
    Call:
    multiclass.roc.default(response = df.imputed.test$Result, predictor = as.vector(as.integer(svm.pred)),     levels = base::levels(df.imputed.test$Result), percent = TRUE)
    
    Data: as.vector(as.integer(svm.pred)) with 4 levels of df.imputed.test$Result: 0, 1, 2, 3.
    Multi-class area under the curve: 96.9%


From the prediction results, we obtain a 96.9% AUC score. The model performs well in predicting the test data but still slightly worse than LDA, QDA and multinominal logistic regression. This suggests the datasets class distribution are better off seperated by a linear decision boundry rather than a gaussian boundry.

Since linear decision boundaries seem to perform the 'best' so far we should look at SVM linear kernal. 


```R
#Use cross validation to find the best cost parameter (the price of the misclassification)
# Use Linear decision boundary
tune.out <- tune(svm, Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + 
                            ATT8 + ATT9 + ATT10, data = df.imputed.train, kernel = "linear", 
                 ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)
```


    
    Parameter tuning of 'svm':
    
    - sampling method: 10-fold cross validation 
    
    - best parameters:
     cost
       10
    
    - best performance: 0.001333333 
    
    - Detailed performance results:
              cost       error  dispersion
    1   0.01000000 0.096666667 0.018921540
    2   0.01778279 0.072000000 0.020560060
    3   0.03162278 0.055333333 0.018338383
    4   0.05623413 0.039333333 0.018176093
    5   0.10000000 0.033333333 0.015071844
    6   0.17782794 0.029333333 0.015137232
    7   0.31622777 0.022666667 0.013033670
    8   0.56234133 0.018666667 0.010327956
    9   1.00000000 0.015333333 0.009454243
    10  1.77827941 0.016666667 0.009558139
    11  3.16227766 0.003333333 0.006478835
    12  5.62341325 0.002666667 0.004661373
    13 10.00000000 0.001333333 0.004216370
    



```R
# fit the model with the best cost parameter
svm.linear <- svm(Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + ATT7 + 
                            ATT8 + ATT9 + ATT10, kernel = "linear", data = df.imputed.train, 
                cost = tune.out$best.parameter$cost)

# output a summary
summary(svm.linear)

# make a prediction on test dataset
svm.pred <- as.numeric(predict(svm.linear, df.imputed.test))
```


    
    Call:
    svm(formula = Result ~ ATT1 + ATT2 + ATT3 + ATT4 + ATT5 + ATT6 + 
        ATT7 + ATT8 + ATT9 + ATT10, data = df.imputed.train, kernel = "linear", 
        cost = tune.out$best.parameter$cost)
    
    
    Parameters:
       SVM-Type:  C-classification 
     SVM-Kernel:  linear 
           cost:  10 
          gamma:  0.1 
    
    Number of Support Vectors:  227
    
     ( 14 56 116 41 )
    
    
    Number of Classes:  4 
    
    Levels: 
     0 1 2 3
    
    
    



```R
multiclass.roc(df.imputed.test$Result, as.vector(as.integer(svm.pred)),
               levels=base::levels(df.imputed.test$Result), percent=TRUE)
```


    
    Call:
    multiclass.roc.default(response = df.imputed.test$Result, predictor = as.vector(as.integer(svm.pred)),     levels = base::levels(df.imputed.test$Result), percent = TRUE)
    
    Data: as.vector(as.integer(svm.pred)) with 4 levels of df.imputed.test$Result: 0, 1, 2, 3.
    Multi-class area under the curve: 99.59%


The AUC score is 99.59% which is the fourth best prediction score out of all of the models. The SVM uses a one-against-one classification which may provide a better strategy in adjusting the decision boundary (while allowing some misclassifications due to overlapping regions). However, from previous models there does not seem to have datapoints that are not well separated.   

Note: The graph is not plotted since the PCA does not show the same shape previously shown due to sampling a portion of the original data.

# Conclusion

The cancer dataset contains inbalanced class, where two types of cancer share similar attributes. In addition, some classes' attributes are not set for a particular class (i.e. Attributes that classifies node cancer, but true label is a tumor class). Although the graphical representation and statistical information of the dataset seems to indicate spillover between classes, models that separate with a linear decision boundry perform quite well compared to gaussian and random forest methods.  

List of models sorted by descending test AUC score:
- Multinominal Logistic Regression
- QDA
- LDA
- SVM(linear)
- SVM(radial)
- Random Forest

---

The notebook can be found [here](https://github.com/lawko698/notebooks/tree/master/other/)
