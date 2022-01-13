import streamlit as st
from dim_reduction import dr
from load_data import over_view
from PIL import Image
import pandas as pd

if __name__ == '__main__':
    st.title('Data Challenge: Team 6')
    st.text("Group members: Wenxu Li; Xiaoyan Feng")
    option = st.sidebar.selectbox(
        'VIEW selection ',
        ['Data Overview', 'Dimensionality Reduction', 'Subgroup_Gender', 'Subgroup_Age', 'Data Imputation',
         'Time Series'])
    st.header(option)

    if option == "Data Overview":
        over_view()
    elif option == "Dimensionality Reduction":
        img = Image.open('/Users/wenxu/PycharmProjects/DataChallenge/image/originalsetA.png')
        st.image(img, caption='DR on original dataset A')

    elif option == "Subgroup_Gender":
        img = Image.open('/Users/wenxu/PycharmProjects/DataChallenge/image/gender0.png')
        img_2 = Image.open('/Users/wenxu/PycharmProjects/DataChallenge/image/gender1.png')
        st.image(img, caption='DR on male dataset')
        st.image(img_2, caption='DR on female dataset')
    elif option == "Subgroup_Age":
        img_0 = Image.open('/Users/wenxu/PycharmProjects/DataChallenge/image/40<age<70.png')
        st.image(img_0, caption='DR on age subgroup dataset between 40 and 70')
    elif option == "Data Imputation":
        img = Image.open('/Users/wenxu/PycharmProjects/DataChallenge/image/40<mean<70.png')
        st.image(img, caption='DR on age subgroup dataset between 40 and 70 by mean imputation')
    elif option == "Time Series":
        ts_data = pd.read_csv('/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/40<mean<70.csv')
        hr = ts_data['HR']
        st.line_chart(hr)
        O2Sat = ts_data['O2Sat']
        st.line_chart(O2Sat)
        Temp = ts_data['Temp']
        st.line_chart(Temp)
