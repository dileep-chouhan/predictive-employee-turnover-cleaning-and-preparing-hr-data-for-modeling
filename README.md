# Predictive Employee Turnover: Cleaning and Preparing HR Data for Modeling

## Overview

This project focuses on cleaning and preparing a messy HR dataset to create a standardized and usable dataset for building a machine learning model to predict employee attrition.  The analysis involves handling missing values, dealing with inconsistent data formats, and transforming categorical variables to prepare the data for effective model training. This improved dataset will then enable more accurate prediction of employee turnover, allowing for the development of proactive retention strategies.

## Technologies Used

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3 installed.  Then, navigate to the project directory in your terminal and install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print key statistics and summaries of the data cleaning process to the console.  This includes information on missing values handled, data transformations performed, and the final shape of the cleaned dataset.  No plot files are generated in this data preparation stage; visualization will be part of a subsequent modeling project.  The cleaned dataset will be saved as a CSV file named `cleaned_hr_data.csv` in the project directory.