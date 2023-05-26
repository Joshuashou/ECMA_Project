import numpy as np
import pandas as pd
import csv

import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

def cmd():
    df_2010 = clean2010("personsx2010.dat")
    df_2012 = clean2012("personsx2012.dat") # cleaned data for 2012
    df_2014 = clean2014("personsx2014.dat")

    df_2014['MedDelay'] = df_2014['MedDelay'].apply(lambda x: 1 if x == 1 else 0)

    df_2012['MedDelay'] = df_2012['MedDelay'].apply(lambda x: 1 if x == 1 else 0)

    df_2010['MedDelay'] = df_2010['MedDelay'].apply(lambda x: 1 if x == 1 else 0)

    df_2014['MedCancel'] = df_2014['MedCancel'].apply(lambda x: 1 if x == 1 else 0)

    df_2012['MedCancel'] = df_2012['MedCancel'].apply(lambda x: 1 if x == 1 else 0)

    df_2010['MedCancel'] = df_2010['MedCancel'].apply(lambda x: 1 if x == 1 else 0)

    df_2014['CostTooHigh'] = df_2014['CostTooHigh'].apply(lambda x: 1 if x == 1 else 0)

    df_2012['CostTooHigh'] = df_2012['CostTooHigh'].apply(lambda x: 1 if x == 1 else 0)

    df_2010['CostTooHigh'] = df_2010['CostTooHigh'].apply(lambda x: 1 if x == 1 else 0)

    df_2014['InsuranceRefused'] = df_2014['InsuranceRefused'].apply(lambda x: 1 if x == 1 else 0)

    df_2012['InsuranceRefused'] = df_2012['InsuranceRefused'].apply(lambda x: 1 if x == 1 else 0)

    df_2010['InsuranceRefused'] = df_2010['InsuranceRefused'].apply(lambda x: 1 if x == 1 else 0)

    Y_estimators = ['MedDelay', 'MedCancel', 'CostTooHigh', 'InsuranceRefused']

    print("Running Difference in Differences for CHIP")
        
    Pre_experiment = df_2012.loc[(df_2012['CHIP'] == 1) | (df_2012['CHIP'] == 2)]
    Pre_control = df_2012.loc[(df_2012['CHIP'] == 3) & (df_2012['Age'] < 24) & (df_2012['Age'] > 20)]
    Post_experiment = df_2014.loc[(df_2014['CHIP'] == 1) | (df_2014['CHIP'] == 2)]
    Post_control = df_2014.loc[(df_2014['CHIP'] == 3) & (df_2014['Age'] < 24) & (df_2012['Age'] > 20)]


    for Y in Y_estimators:
        Pre_control_mean = Pre_control[Y].mean()
        Pre_experiment_mean = Pre_experiment[Y].mean()
        Post_control_mean = Post_control[Y].mean()
        Post_experiment_mean = Post_experiment[Y].mean()
        
        chip_did = (Post_experiment_mean - Pre_experiment_mean) - (Post_control_mean - Pre_control_mean)
        
        print("Causal Coefficient for " + str(Y) + "is " + str(chip_did))


    print("Running Placebo Test for comparing 2010 versus 2011, acting as if treatment is 2011")

    Pre_experiment = df_2010.loc[(df_2010['CHIP'] == 1) | (df_2010['CHIP'] == 2)]
    Pre_control = df_2010.loc[((df_2012['CHIP'] ==3)) & (df_2010['Age'] < 24) & (df_2010['Age'] > 20)]
    Post_experiment = df_2012.loc[(df_2012['CHIP'] == 1) | (df_2012['CHIP'] == 2)]
    Post_control = df_2012.loc[((df_2012['CHIP'] ==3)) & (df_2012['Age'] > 20)]


    for Y in Y_estimators:
        Pre_control_mean = Pre_control[Y].mean()
        Pre_experiment_mean = Pre_experiment[Y].mean()
        Post_control_mean = Post_control[Y].mean()
        Post_experiment_mean = Post_experiment[Y].mean()
        
        chip_did = (Post_experiment_mean - Pre_experiment_mean) - (Post_control_mean - Pre_control_mean)
        
        print("Causal Coefficient for " + str(Y) + "is " + str(chip_did))

    #Build INdicator variables for Fuzzy Difference in Discontinuity
    df_2012['ACA'] = 0
    df_2014['ACA'] = np.where((df_2014['Age'] <= 18) & ((df_2014['CHIP'] == 1) | (df_2014['CHIP'] == 2)), 1, 0)

    df_2012['Year_indicator'] = 0
    df_2014['Year_indicator'] = 1

    fuzzy_diff_df = pd.concat([df_2012, df_2014])
    fuzzy_diff_df['Age_indicator'] = np.where(fuzzy_diff_df['Age']<= 18, 1, 0)

    fuzzy_diff_df['CHIP_indicator'] = np.where(fuzzy_diff_df['CHIP'] == 3, 0, 1)#CHIP value of 1,2 means you are in experimental group

    fuzzy_diff_df['Age*Year'] = fuzzy_diff_df['Year_indicator'] * fuzzy_diff_df['Age_indicator']

    #Now, we want this to work for Fuzzy Difference in Discontinuity 

    chip_x =sm.add_constant(fuzzy_diff_df[['Age_indicator', 'Year_indicator', 'Age*Year']])

    fuzzy_diff_df['Predicted_CHIP'] = sm.OLS(fuzzy_diff_df['CHIP_indicator'], chip_x).fit().predict(chip_x)

    aca_x = sm.add_constant(fuzzy_diff_df[['Age_indicator', 'Year_indicator', 'Age*Year']])

    fuzzy_diff_df['Predicted_ACA'] = sm.OLS(fuzzy_diff_df['ACA'], aca_x).fit().predict(aca_x)


    #Now, we run the final regression 
    print("Running Fuzzy Diff in Disc Estimator")


    Y_estimators = ['MedDelay', 'MedCancel', 'CostTooHigh', 'InsuranceRefused']

    fuzzy_diff_df['Pred_CHIP * Pred_ACA'] = fuzzy_diff_df['Predicted_CHIP'] * fuzzy_diff_df['Predicted_ACA']

    fuzzy_diff_df['Year * Pred_CHIP * Pred_ACA'] = fuzzy_diff_df['Predicted_CHIP'] * fuzzy_diff_df['Predicted_ACA'] * fuzzy_diff_df['Year_indicator']

    fuzzy_diff_x = sm.add_constant(fuzzy_diff_df[['Predicted_CHIP', 'Predicted_ACA', 'Year_indicator', 'Pred_CHIP * Pred_ACA', 'Year * Pred_CHIP * Pred_ACA']])

    for response in Y_estimators:
        fuzzy_diff_y = fuzzy_diff_df[response]
        regression = sm.OLS(fuzzy_diff_y, fuzzy_diff_x)
        fit = regression.fit()
        print("Effect of response" + str(response) + " is " + str(fit.params['Year * Pred_CHIP * Pred_ACA']) 
            + " with p_value " + str(fit.pvalues['Year * Pred_CHIP * Pred_ACA']))

    adult_df = pd.read_csv('RDD_data/adult21.csv')
    child_df = pd.read_csv('RDD_data/child21.csv')

    adult_income_df = pd.read_csv('RDD_data/adultinc21.csv')
    child_income_df = pd.read_csv('RDD_data/childinc21.csv')

    #RATCAT_A 

    #Merge income with Adult and Child dataframes based on 'HHX' 

    adult_merged = pd.merge(adult_df, adult_income_df, on = 'HHX')
    child_merged = pd.merge(child_df, child_income_df, on = 'HHX')

    #Get rows we want


    #HISTOP_COST_A is Cost Increase

    #HISTOPJOB_A is Number of months without coverage

    #PAYBLL12M_A is problems paying medical bills, past 12 months

    #MEDDL12M_A is Delayed medical care due to cost, past 12m

    #MEDNG12M_A is Needed medical care but did not get it due to cost, past 12m

    adult_data = adult_df[['HHX', 'AGEP_A', 'SEX_A', 'HOSPONGT_A', 'RATCAT_A',
                    'CHIP_A','PAYBLL12M_A',
                    'MEDDL12M_A','MEDNG12M_A', ]]
    child_data = child_df[['HHX', 'AGEP_C', 'SEX_C', 'HOSPONGT_C', 'RATCAT_C',
                    'CHIP_C', 'PAYBLL12M_C',
                    'MEDDL12M_C','MEDNG12M_C', ]]


    child_data = child_data.loc[(child_data['AGEP_C'] >= 14)]
    #df['new_column'] = df['a'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    child_data['Treatment'] = child_df['CHIP_C'].apply(lambda x: 1 if x == 1 or x == 2 else 0)

    adult_data = adult_data.loc[adult_data['AGEP_A'] <= 24] #Experimental group should not be much higher than 23. 



    #Rename Columns
    adult_data.rename(columns = {'AGEP_A':'Age', 'SEX_A':'Sex', 'HOSPONGT_A': 'Hospitalized', 
                            'RATCAT_A': 'Income_Ratio', 'CHIP_A': 'CHIP',
                            'PAYBLL12M_A': 'PayBill', 'MEDDL12M_A': 'DelayedCareCost',
                                'MEDNG12M_A':'NoCareCost'},inplace=True)

    child_data.rename(columns = {'AGEP_C':'Age', 'SEX_C':'Sex', 'HOSPONGT_C': 'Hospitalized', 
                            'RATCAT_C': 'Income_Ratio', 'CHIP_C': 'CHIP',
                            'PAYBLL12M_C': 'PayBill', 'MEDDL12M_C': 'DelayedCareCost',
                            'MEDNG12M_C':'NoCareCost'}, inplace=True)

    adult_data['Treatment'] = 0

    #RUN RDD ON above target variable to be 'PAYBLL12M_C'
    rdd_data = pd.concat([child_data, adult_data])

    # Create the running variable for the first stage: whether the age is over the threshold (18)
    rdd_data['Age_over_18'] = np.where(rdd_data['Age'] > 18, 1, 0)

    # Add an interaction term between the running variable and the cutoff indicator
    rdd_data['Age_over_18_times_Age'] = rdd_data['Age_over_18'] * rdd_data['Age']

    #Delayed Care due to Cost
    rdd_data['DelayCareCost'] = rdd_data['DelayedCareCost'].apply(lambda x: 1 if x == 1 else 0)

    #Inability to Pay hospital Bill
    rdd_data['PayBill'] = rdd_data['PayBill'].apply(lambda x: 1 if x == 1 else 0)

    #Didn't get care due to Cost
    rdd_data['NoCareCost'] = rdd_data['NoCareCost'].apply(lambda x: 1 if x == 1 else 0)



    # First stage regression: Treatment on Age_over_18, Age, Age_over_18_times_Age, Income_Ratio
    X_first = sm.add_constant(rdd_data[['Age_over_18', 'Age', 'Age_over_18_times_Age', 'Income_Ratio']])
    Y_first = rdd_data['Treatment']
    first_stage = sm.OLS(Y_first, X_first).fit()
    rdd_data['predicted_treatment'] = first_stage.predict(X_first)

    response_variables = ['NoCareCost', 'DelayedCareCost', 'PayBill']

    print("Running Regression Discontinuity Design for ACA")

    for response in response_variables:
        # Second stage regression using 2SLS: response on predicted treatment, Age, Age_over_18_times_Age, Income_Ratio
        Y_second = rdd_data[response]
        X_second = sm.add_constant(rdd_data[['Treatment', 'Age', 'Age_over_18_times_Age', 'Income_Ratio']])
        model = sm.OLS(Y_second, X_second, instrument = X_first)
        second_stage = model.fit()
        print(second_stage.summary())


def clean2010(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    age_p = []
    household = []
    fam = []
    indiv = []
    schip = []
    pdmed = []
    med_cancel = []
    in_hos = []
    cost_too_high = []
    insurance_coverage = []

    for line in lines:
        age = int(line[65:67])

        if age >= 15 and age <= 24:
            hhx = int(line[6:12])
            family = int(line[15:17])
            individual = int(line[17:19])
            if line[622] == ' ':
                chip = 0
            else:
                chip = int(line[622])
            m_cancel = int(line[526])
            med_delay = int(line[525])
            hos = ''
            
            if line[643] == ' ':
                cost = 0
            else:
                cost = int(line[643])
            
            if line[647] == ' ':
                insurance = 0
            else:
                insurance = int(line[647])

            age_p.append(age)
            household.append(hhx)
            fam.append(family)
            indiv.append(individual)
            schip.append(chip)
            pdmed.append(med_delay)
            med_cancel.append(m_cancel)
            in_hos.append(hos)
            cost_too_high.append(cost)
            insurance_coverage.append(insurance)

    d = {'Age': age_p, 'Household': household, 'Family': family, 'Individual Within Family' : indiv,
         'CHIP': schip, 'MedDelay': pdmed, 'MedCancel':med_cancel,
         'BeenInHospital': in_hos, 'CostTooHigh': cost_too_high, 
         'InsuranceRefused': insurance_coverage}
    
    df = pd.DataFrame.from_dict(d)
    df.fillna(0, inplace=True)
    return df

def clean2012(filename):
    """
    'PDMED12M', 'PHOSPYR2', 'HISTOP5','HISTOP6'
    """
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    age_p = []
    household = []
    fam = []
    indiv = []
    schip = []
    pdmed = []
    med_cancel = []
    in_hos = []
    cost_too_high = []
    insurance_coverage = []

    for line in lines:
        age = int(line[65:67])

        if age >= 15 and age <= 24:
            hhx = int(line[6:12])
            family = int(line[15:17])
            individual = int(line[17:19])
            chip = int(line[674])
            med_delay = int(line[526])
            m_cancel = int(line[527])
            hos = int(line[528])
            
            if line[698] == ' ':
                cost = 0
            else:
                cost = int(line[698])
                
            if line[699] == ' ':
                insurance = 0
            else:
                insurance = int(line[699])

            age_p.append(age)
            household.append(hhx)
            fam.append(family)
            indiv.append(individual)
            schip.append(chip)
            pdmed.append(med_delay)
            med_cancel.append(m_cancel)
            in_hos.append(hos)
            cost_too_high.append(cost)
            insurance_coverage.append(insurance)

    d = {'Age': age_p, 'Household': household, 'Family' : fam, 'Individual Within Family' : indiv, 
         'CHIP': schip, 'MedDelay': pdmed, 'MedCancel':m_cancel,
         'BeenInHospital': in_hos, 'CostTooHigh': cost_too_high,
         'InsuranceRefused': insurance_coverage}
    
    df = pd.DataFrame.from_dict(d)
    df.fillna(0, inplace=True)
    return df

def clean2014(filename):
    """
    'PDMED12M', 'PHOSPYR2', 'HISTOP5','HISTOP6'
    """
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    age_p = []
    household = []
    fam = []
    indiv = []
    schip = []
    pdmed = []
    med_cancel = []
    in_hos = []
    cost_too_high = []
    insurance_coverage = []

    for line in lines:
        age = int(line[65:67])

        if age >= 15 and age <= 24:
            hhx = int(line[6:12])
            family = int(line[15:17])
            individual = int(line[17:19])
            chip = int(line[691])
            m_cancel = int(line[525])
            med_delay = int(line[524])
            hos = int(line[526])
            
            if line[729] == ' ':
                cost = 0
            else:
                cost = int(line[729])
            
            if line[730] == ' ':
                insurance = 0
            else:
                insurance = int(line[730])

            age_p.append(age)
            household.append(hhx)
            fam.append(family)
            indiv.append(individual)
            schip.append(chip)
            pdmed.append(med_delay)
            med_cancel.append(m_cancel)
            in_hos.append(hos)
            cost_too_high.append(cost)
            insurance_coverage.append(insurance)

    d = {'Age': age_p, 'Household': household, 'Family': family, 'Individual Within Family' : indiv,
         'CHIP': schip, 'MedDelay': pdmed, 'MedCancel':med_cancel,
         'BeenInHospital': in_hos, 'CostTooHigh': cost_too_high, 
         'InsuranceRefused': insurance_coverage}
    
    df = pd.DataFrame.from_dict(d)
    df.fillna(0, inplace=True)
    return df



if __name__ == "__main__":
    cmd()