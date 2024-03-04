
import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
  
select_page = st.sidebar.radio('Select page', ['Introduction','Analysis', 'Model Classification'])

if select_page == 'Introduction':
    
    
    def main():
        
        st.title('Customer Churn in Banking Sector')

        st.image('bank.jpg')
        
        
        st.write('### Introduction to my data:')
        
        st.write('''
        The dataset "Churn_Modelling (1).csv" focuses on the analysis of customer retention (churn) within a business context. It contains demographic and transactional information about customers, along with indicators related to loyalty and satisfaction.

        The dataset provides an opportunity to explore several key aspects such as the overall churn rate, variations in churn based on different demographic characteristics, and the impact of customer interactions on the likelihood of churn. It also allows for an analysis of the correlation between customer satisfaction, loyalty, and the consumption of specific products or services.

        Furthermore, the dataset can be used to build predictive models aimed at anticipating customer churn based on historical data. Analysis questions can cover topics ranging from customer segmentation to evaluating the effectiveness of loyalty programs, and identifying the most influential factors on churn.

        In summary, "Churn_Modelling (1).csv" provides a rich foundation to explore the dynamics of customer retention, using descriptive and predictive analysis techniques to extract crucial insights for strategic decision-making in the field of customer management.
        
        ''')
        
        st.header('Dataset Features Overview')
        
        st.write('''
                *CustomerID*: A unique identifier assigned to each customer.
                
                *Surname*: The customer's last name.
                 
                *CreditScore*: The customer's credit score, which may be used to assess creditworthiness.
                 
                *Geography*: The geographical location where the customer is based.
                 
                *Gender*: The customer's gender (male, female, other).
                  
                *Age*: The customer's age.
                 
                *Tenure*: The number of years the customer has been a client of the company.
                 
                *Balance*: The customer's account balance.
                
                *NumOfProducts*: The number of products held by the customer with the company.
                 
                *HasCrCard*: Indicates whether the customer has a credit card (1 for Yes, 0 for No).
                
                *IsActiveMember*: Indicates whether the customer is an active member (1 for Yes, 0 for No).
                
                *EstimatedSalary*: The estimated salary of the customer.
                
                *Exited*: The target variable, indicating whether the customer has churned (1 for Yes, 0 for No).
                 ''')
        
        

    if __name__=='__main__':
         main()

            
elif select_page == 'Analysis':
    
    def main():
        st.title('Customer Churn Models ')
        st.image('churn2.jpg')
        cleaned_df = pd.read_csv('cleaned_df.csv')
        st.write('### Head of DataFrame')
        st.dataframe(cleaned_df.head(9))
        tab1, tab2, tab3 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
        
        for col in cleaned_df.columns:
            tab1.plotly_chart(px.histogram(cleaned_df, x= col))
            
            
        tab2.write('### Does age affects the chance of creditscore?')
        tab2.plotly_chart(px.scatter(cleaned_df, x='Age', y= 'CreditScore'))
        
        
        tab2.write('### Is there a relationship between the estimated salary of customers and their likelihood to  Exited?')
        tab2.plotly_chart(px.box(cleaned_df, x='Exited', y='EstimatedSalary'))
                    
        
        
        tab2.write('### Which gender has the highest creditscore?')
        tab2.plotly_chart(px.histogram(cleaned_df, x= 'Gender', y= 'CreditScore'))
        
        
        tab2.write('### How does the distribution of age differ between customers who Exited and those who do not?')
        tab2.plotly_chart(px.box(cleaned_df, x= 'Exited', y= 'Age'))
        

        tab2.write('### Is there a correlation between credit scores and the likelihood of Exited?')
        tab2.plotly_chart(px.box(cleaned_df, x='Exited', y='CreditScore'))
        
        
        tab2.write('### Do customers with longer tenure exhibit lower exited rates compared to those with shorter tenure?')
        tab2.plotly_chart(px.box(cleaned_df, x='Exited', y= 'Tenure'))
        
        tab2.write('### How does the number of products a customer has with the bank relate to their likelihood of Exited?')
        tab2.plotly_chart(px.box(cleaned_df, x='Exited', y='NumOfProducts'))
        
        
        tab2.write('### How does Estimatedsalary differ bassed on Gender ?')
        tab2.plotly_chart(px.box(cleaned_df, x= 'Gender', y= 'EstimatedSalary' ))
        
        
        tab2.write('### Is there a relationship between the tenure of a customer with the company and the number of products they hold?')
        tab2.plotly_chart(px.box(cleaned_df, x= 'Tenure', y= 'NumOfProducts' ))
        
        
        tab2.write('### Do gender-based patterns emerge in the usage of different products?')
        tab2.plotly_chart(px.box(cleaned_df, x= 'Gender', y= 'NumOfProducts' ))
        
        
        tab2.write('### Is there a correlation between the tenure of customers with the company and their credit scores?')
        tab2.plotly_chart(px.box(cleaned_df, x='Tenure', y= 'CreditScore'))
        
        
        tab2.write('###  How does the tenure of customers with the company relate to their churn rates?')
        tab2.plotly_chart(px.box(cleaned_df, x='Exited', y='Tenure' ))
        
        
        
                                
        
        tab3.write('### How does the distribution of Exited vary across diffrent demographics(Age,Gender,Geography) ?')
        tab3.plotly_chart(px.box(cleaned_df, x='Geography', y='Age', color= 'Gender'))
        
        tab3.write('### Are there specific time intervals during which exited rates are higher?')
        tab3.plotly_chart(px.box(cleaned_df, x='Exited', y='Tenure', color = 'Gender'))
                          
                          
        tab3.write('### What are the key variables that have a significant impact on customer exited?')
        tab3.plotly_chart(px.imshow(cleaned_df.corr(), text_auto=True, width= 1000, height = 800))
        
        
        tab3.write('### How do interactions between creditescore, account balance, and customer tenure impact the exited rate?')
        tab3.plotly_chart(px.box(cleaned_df, x='Tenure', y='Balance', color = 'Exited'))
        
        
        
        tab3.write('### Is there a correlation between customers estimated income and their likelihood to cancel their subscription?')
        tab3.plotly_chart(px.box(cleaned_df, x='Exited', y='EstimatedSalary', color = 'Gender'))
        
        
        tab3.write('### How does the tenure of the relationship with the company and the status of active membership impact the churn rate ?')
        tab3.plotly_chart(px.box(cleaned_df, x='IsActiveMember', y='Tenure', color = 'Exited'))
        
        
        tab3.write('### Are there geographic patterns influencing customer loyalty and, consequently, the exited rate?')
        tab3.plotly_chart(px.histogram(cleaned_df, x='Geography', y='Exited', color= 'Gender'))
        
        
        tab3.write('###  How does the customers usage of a diverse range of products interact with the exited rate?')
        tab3.plotly_chart(px.histogram(cleaned_df, x='NumOfProducts', y='Exited', color= 'Gender'))

        
        tab3.write('###  How do age groups and gender influence the types of products customers prefer?')
        tab3.plotly_chart(px.box(cleaned_df, x='NumOfProducts', y='Age', color= 'Gender'))
        
        
        
        
        
        
        
        
        
        
        
                          
        
        

        
    if __name__=='__main__':
         main()
            
            
            
            
elif select_page =='Model Classification':
     
    def main():
        
        st.title('Churn Modelling ')

        st.image('churn1.jpg')
        
        cleaned_df = pd.read_csv('cleaned_df.csv')

        st.radio('### who has the most remaining races in the company ?',['Male','Female'])


        st.multiselect('Geographic location of customers:',['France', 'Spain', 'Germany'])


        st.slider('Customers ages:', 19,88)



        st.write('min_CreditScore')
        min_CreditScore= cleaned_df['CreditScore'].min()
        st.write(min_CreditScore)



        st.write('Age_Creditscore')
        Age_Creditscore = cleaned_df.groupby('Age').max()['CreditScore'].nsmallest(3)
        st.write(Age_Creditscore)
                                                                                      
                                                                                     
                                                                                      
        
        
        # first step:load pkl file
        my_pipeline = joblib.load('RF_pipeline.pkl')

        # second step:create dataframe from input data
        def Prediction(CreditScore, Geography, Gender, Age, Tenure, Balance,
                  NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):

            df = pd.DataFrame(columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
               'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Es timatedSalary'])


            df.at[0, 'CreditScore'] = CreditScore
            df.at[0, 'Geography'] = Geography
            df.at[0, 'Gender'] = Gender
            df.at[0, 'Age'] = Age
            df.at[0, 'Tenure'] = Tenure
            df.at[0, 'Balance'] = Balance
            df.at[0, 'NumOfProducts'] = NumOfProducts
            df.at[0, 'HasCrCard'] = HasCrCard
            df.at[0, 'IsActiveMember'] = IsActiveMember
            df.at[0, 'EstimatedSalary'] = EstimatedSalary


            # make prediction of dataframe using pipeline
            result = my_pipeline.predict(df)[0]

            return result


        Geography =st.selectbox('Please provide where the majority of customers are concentrated', ['France','Germany','Spain'] )   
        Age = st.sidebar.slider('Enter customers age',18,92)  
        Tenure = st.sidebar.slider('Please provide the longest period the customer has apent with',0,10)
        Gender = st.sidebar.radio('Gender', ['Male','Female'])
        EstimatedSalary = st.sidebar.slider('Please provide estimated salary of the customers',11,199992)
        CreditScore = st.sidebar.slider('Please enter highest credit score',350,850)
        Balance = st.sidebar.slider('Please provide the remaining amount is in the customers account',0,250898)
        NumOfProducts = st.selectbox('Please provide customers products',[0,1,2,3])
        HasCrCard = st.sidebar.radio('Please provide whether the customer has credit card or not',[1, 0])
        IsActiveMember = st.selectbox('Please provide whether the customer is an active member or no',[1, 0])




        if st.button('Predict'):
            result= Prediction(CreditScore, Geography, Gender, Age, Tenure, Balance,
                               NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)


            if result == 0:
                st.write('the customer is not likely to leave the bank')
            elif result == 0:
                st.write('the customer is likely to leave the abnk')
                                                                                      
                                                                    
    if __name__=='__main__':
         main()
