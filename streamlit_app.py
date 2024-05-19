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

    return df_features, df_objetivos, dates
df_features, df_objetivos, dates = load_data()

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
    y = np.array(df_objetivos[['House.overall..kW.']]).flatten()

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
    final_datetime = datetime.combine(final_day, final_hour)

    mes_inicio = initial_datetime.month
    mes_final = final_datetime.month
    dia_inicio = initial_datetime.day
    dia_final = final_datetime.day

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

    fig, ax = plt.subplots()

    ax.plot(dates_test, costes, label='Cost')
    ax.plot(dates_test, c_ascendente, label='Cumulative')
    ax.set_xlabel('Date')
    ax.set_ylabel('€', rotation=0)
    ax.set_title('Cost Predicted')
    ax.yaxis.set_label_coords(-0.1,0.5)
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    # # Sidebar - Selección de años
    # st.sidebar.header('Selecciones del usuario')
    # ano_min = st.sidebar.selectbox('Año comienzo', list(range(1950,2020)))
    # ano_max = st.sidebar.selectbox('Año final', list(reversed(range(1950,2020))))
    # ran_anual = list(range(ano_min, ano_max+1))

    # # Sidebar - Selección variable
    # dic_var = {'Horas anuales trabajadas por ocupados':'avh',
    #            'Horas anuales trabajadas por semana':'avh_week',
    #            'GDP del lado del gasto (millones)':'rgdpe',
    #            'GDP del lado del gasto per capita':'rgdpe_cap',
    #            'GDP del lado de la producción (millones)':'rgdpo',
    #            'GDP del lado de la producción per capita':'rgdpo_cap',
    #            'Población (millones)':'pop',
    #            'Personas contratadas (millones)':'emp',
    #            'Índice de capital humano':'hc',
    #            'Nivel de ingresos o consumo diario':'inc_con',
    #            'Días de vacaciones y festivos':'days_vac',
    #            'Productividad':'prod',
    #            'Índice de desarrollo humano':'idh',
    #            'Posición en el World Happiness Report':'happ',
    #            'PIB por hora trabajada':'GDP_hour',
    #            'GDP per capita':'eco',
    #            'Esperanza de vida':'life_exp',
    #            'Libertad':'freed',
    #            'Confianza en el Gobierno':'trust',
    #            'Generosidad':'gen'}
    # selected_var = st.sidebar.selectbox('Variables', dic_var.keys())

    # # Acote años y variable
    # df_anos = df[df.year.isin(ran_anual)]
    # ev = df_anos.loc[:,['continent','country','year', dic_var[selected_var]]]
    # ev.dropna(axis='rows', how='any', inplace=True)

    # #Sidebar - Autoselección país por continente
    # st.sidebar.markdown('Puedes elegir todos los países de uno o más continentes')
    # sorted_unique_cont = sorted(ev.continent.unique())
    # selected_continent = st.sidebar.multiselect('Continentes', sorted_unique_cont, [])
    # ev_cont = ev[ev.continent.isin(selected_continent)]

    # # Sidebar - Selección país
    # st.sidebar.markdown('O escogerlos uno a uno')
    # sorted_unique_country = sorted(ev.country.unique())
    # al_selected_country = sorted(ev_cont.country.unique())
    # selected_country = st.sidebar.multiselect('Países', sorted_unique_country, al_selected_country)

    # # Acote país
    # ev = ev[ev.country.isin(selected_country)]

    # # Ploteo gráfico de líneas
    # ev_lin = ev.pivot(index='year', columns='country', values=dic_var[selected_var])
    # st.subheader('Gráfico de líneas')
    # fig = ev_lin.iplot(asFigure=True, kind='line', title=selected_var)
    # st.plotly_chart(fig)

    # # Selección año para gráfico de barras
    # st.subheader('Gráfico de barras')
    # st.markdown('Elige el año en el que quieres comparar los países')
    # selected_year = st.selectbox('Año para comparar', list(reversed(sorted(ev.year.unique()))))

    # # Ploteo gráfico de barras
    # ev_mask = ev['year']==selected_year
    # ev_comp = ev[ev_mask]
    # ev_bar = ev_comp.loc[:,['country', dic_var[selected_var]]]
    # ev_bar = ev_bar.sort_values(by=dic_var[selected_var],ascending=False)
    # ev_bar = ev_bar.set_index('country')
    # bar = ev_bar.iplot(asFigure=True, kind='bar', xTitle='Países',yTitle=selected_var,color='blue')
    # st.plotly_chart(bar)
    # st.markdown('*Si al añadir un país, este no aparece, se debe a que no hay datos para ese año')

    # # Selección variable para comparar
    # st.subheader('Comparación de variables')
    # st.markdown('Elige la variable con la que quieres comparar la variable seleccionada anteriormente en el año seleccionado')
    # dic2 = dic_var.copy()
    # del dic2[selected_var]
    # var_comp = st.selectbox('Variable para comparar', dic2.keys())

    # # Ploteo scatter
    # ev_scat = df_anos.loc[:,['continent','country','year',dic_var[selected_var],dic_var[var_comp]]]
    # ev_mask_scat = ev_scat['year']==selected_year
    # ev_scat = ev_scat[ev_mask_scat]
    # ev_scat = ev_scat[ev_scat.country.isin(selected_country)]
    # ev_scat.dropna(axis='rows', how='any', inplace=True)
    # scat = px.scatter(ev_scat,x=dic_var[selected_var],y=dic_var[var_comp],color='continent',hover_name='country')
    # st.plotly_chart(scat)
    # st.markdown('*Si al añadir un país, este no aparece, se debe a que no hay datos para ese año')
