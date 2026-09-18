import streamlit as st


st.set_page_config(
    page_title="Building Human-Centered AI Applications",
    page_icon=":material/edit:"
)

lab1 = st.Page(
    "Lab1.py", 
    title="Lab  1",
    icon = "✏️"
)

lab2 = st.Page(
    "Lab2.py",
    title="Lab 2",
    icon = "✏️",
)

lab3 = st.Page(
    "Lab3.py",
    title="Lab 3",
    icon="✏️"       
)

lab4 = st.Page(
        "Lab4.py",
    title= "Lab 4",
    icon="✏️",
    default=True

)

#Creare 4A collection
#Cost oney so make ut seperate
pg = st.navigation([
    lab1,
    lab2,
    lab3,
    lab4
])

pg.run()