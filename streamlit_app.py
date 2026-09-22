import streamlit as st

st.set_page_config(
    page_title="Building Human-Centered AI Applications",
    page_icon="🤖"
)

# Create pages

lab2 = st.Page(
    "Lab2.py",
    title="Lab 2",
    icon="📝"
)

lab3 = st.Page(
    "Lab3.py",
    title="Lab 3",
    icon="📝"
)

lab4 = st.Page(
    "Lab4.py",
    title="Lab 4",
    icon="📝"
    
)

lab5 = st.Page(
    "Lab5.py",
    title="Lab 5",
    icon="📝",
    default=True
)

# Navigation
pg = st.navigation(
    [ lab2, lab3, lab4, lab5]
)

# Run selected page
pg.run()