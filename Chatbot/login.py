import streamlit as st

c=st.container(border=1)
c.markdown("<h1 style='text-align: center; color: Blue;'>Login</h1>", unsafe_allow_html=True)
text_input = c.text_input("Enter yout Username")
password = c.text_input("Enter your password", type="password")
col1, col2, col3 = c.columns([10, 10, 5])
button=col2.button("Submit", type="primary")

if button:
    st.page_link("pages/main.py", label="main")

    







