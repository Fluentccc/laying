import streamlit as st
from tensorflow.python.keras.models import load_model
import joblib
import pandas as pd

model_dic = {
	'RandomForest': joblib.load('models/rfr.pkl'),
	'RNN': load_model('models/rnn.h5'),
	'CNN': load_model('models/cnn.h5')
}
scaler = joblib.load('models/x_transform.pkl')
scaler_ = joblib.load('models/y_inverse.pkl')

st.write("<h1 style='font-style:italic;font-family: Times New Roman'>Prediction of laying rate of laying hens</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='font-family: Times New Roman'>Please enter the following information to make a prediction</h4>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 5])
with col1:
	day = st.text_input('day')
	sun = st.text_input('sun')
	temp = st.text_input('temp')
	model_name = st.selectbox(
		'Please select a model 👇',
		('RandomForest', 'RNN', 'CNN'))
	
with col2:
	food = st.text_input('food')
	water = st.text_input('water(kg)')
	hen = st.text_input('hen')

if st.button('Predict'):
	test_data = pd.DataFrame([float(day),float(sun),float(temp),float(hen),float(food),float(water)/1000], 
		index=['day', 'sun', 'temp', 'hen', 'food', 'water'])
	# 归一化处理
	scaled_pred_data = scaler.transform(test_data.T)
	with st.spinner('Prediction is in progress. Please wait...'):
		if model_name == 'RandomForest':
			# 重塑数据形状以适应CNN模型的要求
			scaled_pred_data = scaled_pred_data.reshape((scaled_pred_data.shape[0], scaled_pred_data.shape[1]))

			pred = int(model_dic[model_name].predict(scaled_pred_data))/test_data.T['hen'].values[0]
		else:
			# 重塑数据形状以适应CNN模型的要求
			scaled_pred_data = scaled_pred_data.reshape((scaled_pred_data.shape[0], scaled_pred_data.shape[1], 1))
			prediction = model_dic[model_name].predict(scaled_pred_data)

			if scaler_.inverse_transform(prediction)[0][0] <= 0:
				pred = 0
			else:
				pred = int(scaler_.inverse_transform(prediction)[0][0])/test_data.T['hen'].values[0]
	st.subheader('Laying rate: {}%'.format(pred*100))
