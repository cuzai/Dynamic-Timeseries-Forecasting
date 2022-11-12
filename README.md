# Smartphone Sales Forecast
## 1. Objective
  - Smartphone sales prediction using historical smartphone sales data
  - When testing, the length of the time sequence of the input can vary (from zero month to any month)
  
## 2. Model
  - ### Smartphone Spec AutoEncoder
    - This model vectorizes the specs of individual smartphones (Spec example below)
    ![image](https://user-images.githubusercontent.com/13309017/201269233-422b5c78-e26f-4a85-b5ea-39b16a820d90.png)
    - Architecture  
    ![image](https://user-images.githubusercontent.com/13309017/201280363-feeab2fa-970c-4358-87b9-80b3e50806ce.png)
 
   ###  
   ###
 
  - ### Time sequence generation model applying the concept of CharRNN  
    - This model dynamically generates time series by taking advantage of the dataset structure below 
      ![image](https://user-images.githubusercontent.com/13309017/201481240-69497e23-7626-4550-9af2-5f5a5334569f.png)

    - Architecture  
      <img src="https://user-images.githubusercontent.com/13309017/201481260-b3bd2e21-f374-47e2-877f-d07a89c2a2aa.png" alt="" width="350"/>
