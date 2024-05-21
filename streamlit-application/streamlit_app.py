# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:36:40 2022

@author: javil
"""

import streamlit as st
import pandas as pd
import cufflinks as cf
from sklearn.ensemble import StackingRegressor
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error


cf.set_config_file(sharing='private',theme='white',offline=True)

st.title('Energetic Consumption Prediction')

st.markdown("""
Select the dates and reveal how much you will spend
""")

# Descarga datos
@st.cache_data
def load_data():
    with open('features.pkl', 'rb') as f:
            X = pickle.load(f)
    with open('objetivos.pkl', 'rb') as f:
            y = pickle.load(f)
    with open('dates.pkl', 'rb') as f:
            dates = pickle.load(f)

    return X, y, dates
df_features, df_objetivos, dates = load_data()

st.sidebar.title("Select appliance/Room")

appl = st.sidebar.selectbox("Appliance/Room", ["Overall", "Dishwasher", "Office", "Fridge", "Wine Cellar", "Garage Door", "Barn", "Well", "Microwave", "Living Room"])

appls = dict(zip(["Overall", "Dishwasher", "Office", "Fridge", "Wine Cellar", "Garage Door", "Barn", "Well", "Microwave", "Living Room"],
                 ['House.overall..kW.', 'Dishwasher..kW.', 'Home.office..kW.',
        'Fridge..kW.', 'Wine.cellar..kW.', 'Garage.door..kW.', 'Barn..kW.',
        'Well..kW.', 'Microwave..kW.', 'Living.room..kW.']))

st.sidebar.title("Select the period")

#pre_trained = st.sidebar.checkbox("Use pre-trained-model")

# Usando el widget st.datetime_input para obtener la fecha y hora del usuario
initial_day = st.sidebar.date_input("Initial day", value=datetime(2016, 1, 1), min_value = datetime(2016, 1, 1), max_value = datetime(2016, 12, 15))
initial_hour = st.sidebar.time_input("Initial hour (exact hour will be considered)")

i_m = initial_day.month
i_d = initial_day.day
i_h = initial_hour.hour

check  = datetime.combine(datetime.today(), initial_hour)
check = check + timedelta(hours=1)
check = check.time()

final_day = st.sidebar.date_input("Final day (max difference: 30 days)", value=initial_day, min_value = initial_day, max_value = initial_day + timedelta(days=30))
final_hour = st.sidebar.time_input("Final hour (exact hour will be considered)", value = check)

f_m = final_day.month
f_d = final_day.day
f_h = final_hour.hour

# Crear botón para actualizar
button_pressed = st.sidebar.button("Update")

# Verificar si se presionó el botón
if button_pressed:

    initial_index = dates.index[(dates['Mes'] == i_m) & (dates['Hora'] == i_h) & (dates['Dia'] == i_d)].tolist()[0]
    final_index = dates.index[(dates['Mes'] == f_m) & (dates['Hora'] == f_h) & (dates['Dia'] == f_d)].tolist()[0]

    dates_test = pd.to_datetime({
        'year': 2016,
        'month': dates.loc[initial_index:final_index, 'Mes'],
        'day': dates.loc[initial_index:final_index, 'Dia'],
        'hour': dates.loc[initial_index:final_index, 'Hora']
    })

    X = np.array(df_features)
    y = np.array(df_objetivos[[appls[appl]]]).flatten()

    X_test = X[initial_index:final_index+1]
    y_test = y[initial_index:final_index+1]

    if False:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        X_train = np.concatenate([X[:initial_index], X[final_index+1:]])
        y_train = np.concatenate([y[:initial_index], y[final_index+1:]])

        m1 = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree= 0.8, learning_rate= 0.1,
                            max_depth= 5, min_child_weight= 5, n_estimators= 200, subsample= 0.6)
        m2 = SVR(C=100, kernel='poly')
        m3 = GradientBoostingRegressor(max_depth=4, n_estimators=200)
        m = StackingRegressor(estimators=[("xgb", m1),
                                            ("svm", m2),
                                            ("gbr", m3)],
                            final_estimator = RandomForestRegressor(max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=300))
        model = m.fit(X_train,y_train)

    yp = model.predict(X_test)

    st.markdown(f"RMSE: {np.round(root_mean_squared_error(yp,y_test),3)}")
    st.markdown(f"MAPE: {np.round(mean_absolute_percentage_error(yp,y_test)*100,3)}%")

    if len(dates_test) > 60:

        df_p = pd.DataFrame({'fecha': dates_test, 'valor': yp})
        df_pr = pd.DataFrame(columns=["fecha","valor","suma_acumulada"])
        s = 0
        size = len(df_p)
        n = size // 30
        st.markdown(f"Aggregation of {n} hours")
        while s+n < size:
            portion = df_p.iloc[s:s+n]
            s += n
            portion['suma_acumulada'] = portion['valor'].cumsum()
            df_pr = pd.concat([df_pr, portion.iloc[[-1]]])
        portion = df_p.iloc[s:]
        s += n
        portion['suma_acumulada'] = portion['valor'].cumsum()
        df_pr = pd.concat([df_pr, portion.iloc[[-1]]])

        df_r = pd.DataFrame({'fecha': dates_test, 'valor': y_test})
        df_rr = pd.DataFrame(columns=["fecha","valor","suma_acumulada"])
        s = 0
        size = len(df_r)
        n = size // 30
        while s+n < size:
            portion = df_r.iloc[s:s+n]
            s += n
            portion['suma_acumulada'] = portion['valor'].cumsum()
            df_rr = pd.concat([df_rr, portion.iloc[[-1]]])
        portion = df_r.iloc[s:]
        s += n
        portion['suma_acumulada'] = portion['valor'].cumsum()
        df_rr = pd.concat([df_rr, portion.iloc[[-1]]])

        fig, ax = plt.subplots()

        ax.plot(df_pr["fecha"], df_pr["suma_acumulada"], label='Predicho')
        ax.plot(df_rr["fecha"], df_rr["suma_acumulada"], label='Real')
        ax.set_xlabel('Date')
        ax.set_ylabel('kW', rotation=0)
        ax.set_title('Compartion between real and predicted')
        ax.yaxis.set_label_coords(-0.1,0.5)
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

    else:
        st.markdown(f"No aggregation")

        fig, ax = plt.subplots()

        ax.plot(dates_test, yp, label='Predicho')
        ax.plot(dates_test, y_test, label='Real')
        ax.set_xlabel('Date')
        ax.set_ylabel('kW', rotation=0)
        ax.set_title('Compartion between real and predicted')
        ax.yaxis.set_label_coords(-0.1,0.5)
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

    initial_datetime = datetime.combine(initial_day, initial_hour)
    final_datetime = datetime.combine(final_day, final_hour) + timedelta(hours=1)

    def api(i,f):

        i = i.strftime('%Y-%m-%d %H:%M:%S')
        f = f.strftime('%Y-%m-%d %H:%M:%S')

        i = str(i).replace(" ","T")
        f = str(f).replace(" ","T")

        url = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
        params = {
            "start_date": i,
            "end_date": f,
            "time_trunc": "hour"
        }

        # Realizar la solicitud GET
        response = requests.get(url, params=params)

        # Verificar si la solicitud fue exitosa (código de estado 200)
        if response.status_code == 200:
            # Extraer los datos en formato JSON
            data = response.json()
            # Aquí puedes procesar los datos según tus necesidades
            #print(data)
        else:
            # En caso de error, imprimir el código de estado
            st.error(f"Error: {response.status_code}")

        data = json.loads(json.dumps(data))
        pvpc_values = data['included'][0]['attributes']['values']
        #print("Precios de mercado peninsular en tiempo real (PVPC):")
        precios = []
        for value in pvpc_values:
            precios.append(value['value'])

        return precios
    
    precios = api(initial_datetime, final_datetime)

    precios = np.array(precios)
    costes = precios*yp/1000
    c_ascendente = costes.cumsum()
    c_total = costes.sum()

    fig, (ax1,ax2) = plt.subplots(1,2)

    if len(dates_test) > 60:

        df_r = pd.DataFrame({'fecha': dates_test, 'valor': costes})
        df_rr = pd.DataFrame(columns=["fecha","valor","suma_acumulada"])
        s = 0
        size = len(df_r)
        n = size // 30
        while s+n < size:
            portion = df_r.iloc[s:s+n]
            s += n
            portion['suma_acumulada'] = portion['valor'].cumsum()
            df_rr = pd.concat([df_rr, portion.iloc[[-1]]])
        portion = df_r.iloc[s:]
        s += n
        portion['suma_acumulada'] = portion['valor'].cumsum()
        df_rr = pd.concat([df_rr, portion.iloc[[-1]]])

        ax1.plot(df_rr["fecha"], df_rr["suma_acumulada"])
    else:
        ax1.plot(dates_test, costes)
    ax2.plot(dates_test, c_ascendente)
    ax1.set_xlabel('Date')
    ax2.set_xlabel('Date')
    ax1.set_ylabel('€')
    ax2.set_ylabel('€')
    ax1.set_title('Predicted Cost')
    ax2.set_title('Accumulated Predicted Cost')
    fig.subplots_adjust(wspace=0.5)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    st.markdown(f"COSTE TOTAL: {c_total}€")
