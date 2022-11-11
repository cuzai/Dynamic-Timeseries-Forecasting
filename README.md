# Smartphone-Sales-Forecast
## 1. Objective
  - Smartphone sales prediction using historical smartphone sales data
  - When testing, the length of the time sequence of the input can vary (from zero month to any months)
  
## 2. Model
  - Smartphone Spec AutoEncoder  
    - This model vectorizes the specs of individual smartphones (Spec example below)
    ![image](https://user-images.githubusercontent.com/13309017/201269233-422b5c78-e26f-4a85-b5ea-39b16a820d90.png)
    - Architecture  
    ![image](https://user-images.githubusercontent.com/13309017/201280363-feeab2fa-970c-4358-87b9-80b3e50806ce.png)
  
  
  - Time sequence generation model applying the concept of CharRNN  
    - This model dynamically generates time series by taking advantage of the dataset structure below 
    - Architecture  
  ![image](https://user-images.githubusercontent.com/13309017/198940558-9d3a8593-ca5d-49b0-aa33-1c9e0e158760.png)
