# Smart House: A Digital Twin

This is a project carried out by students of the Degree of Data Science (GCD) of the Universitat Politènica de València (UPV) during our third year: Daniel Garijo, Ángel López, Javier Luque, Claudia Martínez, Pablo Parrilla and Andrea Sánchez.
The project consists on using data from a house to predict its energetic consumption. The whole project is implemented with python.

___

- **M2_T01**: Brief report done halfway through the project.
- **G1_T01**: Memory of the project that includes a complete report on the problem and the solutions provided. The origin of the data and all the model used can be consulted here.

---

- **requirements.txt**: Versions of the libraries of python needed to deploy the streamlit application.
- **streamlit_app.py**: Base code of the streamlit application developed with python, including inserts of HTML and CSS (https://smarthouse-proyiii.streamlit.app/).
- **streamlit-application**: Folder with pickle files that contain the data used.
  - dates.pkl: numpy array with the dates.
  - features.pkl: numpy array with the independent variables.
  - objetivos.pkl: numpy array with the dependent variables.

---

## data-transformations

Includes the code used in the preprocess stage to transform the original dataset.

## time-series-training

Training of models that deal with the data as a time series. The aggregation for this models is daily.

## machine-learning-training

Training of machine learning models, including embedings. The aggregation for this models is hourly.

## Presentation

https://www.canva.com/design/DAGFv8_S75w/pK_T9-W--0SMqK-kQCPPXw/edit?utm_content=DAGFv8_S75w&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
