# Quality-Inspection 
# Dataset
This dataset is of casting manufacturing product. <br />
Please refer to the dataset from:- https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
# Sample of the dataset
![image](https://github.com/user-attachments/assets/10c7dc0c-fd2c-4a4a-bead-b82b8a770c1c)
# Installation of important libraries
```python
# install tensorflow
!pip install tensorflow

# install streamlit
!pip install streamlit

# for nav-bar
!pip install streamlit-option-menu
```
# Output
![download](https://github.com/user-attachments/assets/9a70f132-da7b-4130-b141-03648b3b8cfb)

# Steps for streamlit deployment:
1.
 ```python
   %%writefile app.py
   # write the code here
```
2.
```python
# we've used local tunnel to make your local server accessible to internet.
!streamlit run app.py & npx localtunnel --port 8501
```
