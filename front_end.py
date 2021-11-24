import streamlit as st
from dim_reduction import dr
from load_data import over_view
from PIL import Image

if __name__ == '__main__':
    st.title('Data Challenge: Team 6')
    st.text("Group members: Wenxu Li; Xiaoyan Feng")
    option = st.sidebar.selectbox(
        'VIEW selection ',
        ['Data Overview', 'Dimensionality Reduction', 'Subgroup_Gender', 'Subgroup_Age', 'Data Imputation'])
    st.header(option)

    if option == "Data Overview":
        over_view()
    elif option == "Dimensionality Reduction":
        img = Image.open('//image/originalsetA.png')
        st.image(img, caption='DR on original dataset A')

    elif option == "Subgroup_Gender":
        img = Image.open('//image/gender0.png')
        img_2 = Image.open('//image/gender1.png')
    elif option == "Subgroup_Age":
        img_0 = Image.open('//image/20<age<40.png')

    elif option == "Data Imputation":
        img = Image.open('//image/originalsetA.png')
