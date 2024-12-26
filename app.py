import streamlit as st
import runpy

# Створення словника для навігації
pages = {
    "Train Model": "train.py",  # файл для тренування моделі
    "Test Model": "test_model.py",    # файл для тестування моделі
}

# Створення бокового меню для вибору сторінки
page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

if page == "Train Model":
    st.title("Training the Model")
    runpy.run_path(pages[page]) 
elif page == "Test Model":
    st.title("Testing the Model")
    runpy.run_path(pages[page]) 

