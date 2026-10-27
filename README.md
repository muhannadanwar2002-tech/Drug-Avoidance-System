# Drug-Avoidance-System
# Drug Side Effects and Medical Conditions EDA Dashboard

## Overview

This project explores a dataset containing information about drugs, medical conditions, side effects, drug classes, Rx/OTC status, pregnancy categories, user ratings, and number of reviews.

The project uses Exploratory Data Analysis (EDA) to identify patterns and relationships within the dataset and presents the results through an interactive Streamlit dashboard.

The main purpose is to organize drug-related information and make patterns in the dataset easier to explore and understand.

## Problem

Healthcare professionals and patients may find it difficult to compare medications for different medical conditions because medications can have different characteristics, ratings, and reported side effects.

This project explores the available information in the dataset to identify:

* Which medical conditions have the largest number of available drugs.
* Which side effects are mentioned most frequently.
* Which drugs have the fewest extracted side-effect terms.
* Which drugs frequently contain potential warning terms in their descriptions.
* How user ratings vary across medical conditions.
* Whether the number of reviews is related to user ratings.

## Project Goals

The project aims to:

1. Explore the relationship between medical conditions and available drugs.
2. Analyze and organize reported side-effect information.
3. Identify potential warning terms in drug descriptions.
4. Compare user ratings across medical conditions.
5. Investigate the relationship between ratings and number of reviews.
6. Present the findings through an interactive dashboard.

## Dataset

The project uses the **Drugs, Side Effects & Medical Conditions** dataset.

The dataset contains information such as:

* Drug name
* Medical condition
* Side effects
* Brand names
* Generic name
* Drug classes
* Rx/OTC status
* Pregnancy category
* Related drugs
* User rating
* Number of reviews

The project currently uses one dataset rather than combining multiple datasets.

## Data Cleaning

The original dataset is inspected for:

* Missing values
* Duplicate records
* Data types
* Unique values
* Irrelevant columns
* Potential outliers

The following cleaning decisions were made:

* Rows without side-effect descriptions were removed for the side-effect analysis.
* The alcohol column was removed because it contained very limited useful information.
* Missing categorical information such as brand names, generic names, drug classes, and pregnancy categories was represented using values such as Unknown or NA
* Missing ratings and review counts were not replaced with zero because zero would represent false information.
* Potential outliers in ratings and number of reviews were investigated using the IQR method.
* Legitimate unusual values were retained rather than automatically removed.
(important note the cleaned data is cleaned_drug_2) 
## Exploratory Data Analysis

### Q1. Which medical conditions have the largest number of available drugs?

The number of distinct drugs associated with each medical condition is calculated and visualized.

The analysis shows that **Colds & Flu** has the largest number of distinct listed drugs in this dataset, followed by **Pain** and **Acne**.

This result represents the number of drugs recorded in this dataset and does not indicate how common a medical condition is in the real world.

### Q2. What are the most common side effects?

The side-effect descriptions are separated into individual terms so that their frequency can be analyzed.

The most frequently mentioned extracted terms are visualized using bar charts.

A limitation of this analysis is that the original side-effect descriptions contain long text. Splitting the descriptions using punctuation can sometimes produce incomplete terms. Therefore, the extracted terms should be interpreted as an initial text-based analysis rather than a complete medical classification of side effects.

The dataset also does not contain a gender column, so gender-based side-effect comparisons cannot be performed.

### Q3. Which drugs frequently contain potential warning terms?

The side-effect descriptions are searched for predefined warning-related terms.

The dashboard calculates the percentage of reviewed descriptions for each drug that contain selected warning terms.

These results are intended to identify descriptions that contain potentially important warnings for further review.

They should not be interpreted as proof that a drug is unsafe or that it should be avoided.

### Q4. How do user ratings vary across drugs and medical conditions?

User ratings are analyzed across medical conditions using statistical summaries and boxplots.

The analysis allows the user to compare:

* Mean ratings
* Rating distributions
* Number of ratings available for each condition

The number of ratings is considered when comparing average ratings because a mean based on very few ratings may not be as representative as a mean based on many ratings.

### Rating vs Number of Reviews

The project also investigates whether drugs with more reviews tend to have higher or lower ratings.

The Pearson correlation between rating and number of reviews was approximately:

`-0.043`

This represents a **very weak negative linear relationship**.

The scatter plot also shows that drugs with very different numbers of reviews can have similar ratings.

Therefore, having more reviews does not appear to strongly indicate having a higher or lower rating in this dataset.

## Outlier Analysis

Potential outliers were identified using the Interquartile Range (IQR) method.

The project investigates outliers in:

* User ratings
* Number of reviews

Outliers were not automatically removed.

For example, some drugs have substantially more reviews than most other drugs. These values can represent genuine differences in the amount of user feedback rather than incorrect data.

Therefore, the approach used in this project is:

**Detect → Visualize → Investigate → Decide**

rather than automatically deleting every statistical outlier.

## Dashboard

The Streamlit application provides an interactive interface for exploring the dataset.

The dashboard includes:

* Dataset preview
* Summary statistics
* Original vs cleaned dataset comparison
* Missing-value analysis
* Outlier analysis
* Medical-condition analysis
* Side-effect analysis
* Warning-term analysis
* Rating analysis
* Rating vs number-of-reviews analysis
* Interactive filters

The dashboard also allows users to explore different medical conditions and dataset versions.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Streamlit

## Notebook

The notebook contains the main EDA workflow, including:

1. Data loading
2. Dataset structure inspection
3. Missing-value analysis
4. Data cleaning
5. Outlier detection
6. Data visualization
7. Relationship analysis
8. Side-effect analysis
9. Medical-condition analysis
10. Rating analysis


## Future Improvements

Possible future improvements include:

* More advanced natural-language processing of side-effect descriptions.
* Better grouping of similar side effects.
* Additional validated medical datasets.
* Improved drug-condition matchings.
* Additional validation using reliable medical sources.
* Improved recommendation logic based on validated clinical information.
