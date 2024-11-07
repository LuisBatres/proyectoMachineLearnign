import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

st.set_page_config(page_title="PROYECTO ML", page_icon="💼", layout="wide")

# Función para convertir el dataframe a excel
def df_a_excel(df):
    # Crear un buffer en memoria
    output = io.BytesIO()
    # Escribir el DataFrame a Excel en el buffer
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
    # Volver al inicio del buffer para que pueda ser descargado
    output.seek(0)
    return output.read()

# Estilo personalizado
st.markdown(
    """
    <style>

        /* Se asegura de que la aplicación ocupe toda la ventana */
        .stApp {
            background-color: #082338;
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100vh;  /* 100% de la altura de la ventana */
            display: flex;
            flex-direction: column;
        }

        /* Elimina los márgenes y bordes de la barra lateral */
        .stSidebar { 
            background-color: #2C3E50; 
            padding: 0;
            margin: 0;
        }

        /* Estilo de los botones para hacerlos más atractivos */
        .stButton > button {
            background-color: #2980B9;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
            box-sizing: border-box;
        }

        .stButton > button:hover {
            background-color: #3498DB;
        }

        /* Pie de página */
        .footer {
            position: fixed; 
            bottom: 0; 
            left: 0px;
            right: 0px;
            width: 100%; 
            text-align: center; 
            font-size: 12px; 
            color: #777;
            background-color: #2C3E50;
            padding: 10px;
            z-index: 999990;
            display: block;
        }

        /* Aseguramos que el área de contenido principal no tenga márgenes */
        .main {
            padding: 0;
            margin: 0;
            flex: 1;
            overflow: hidden;  /* Asegura que no haya barras de desplazamiento innecesarias */
        }

        /* También podemos aplicar el color negro a títulos y subtítulos */
        .css-ffhzg2, .css-1y2b25n, .css-1b6t5f2, .css-17p6f8y, .css-1r6a8v4 {
            color: black !important;
        }

        /* Evita el espacio adicional en los elementos del cuerpo de la aplicación */
        .css-1p1n4r9, .css-12ttj6s, .css-16hu7wv {
            margin: 0 !important;
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True
)


# Menú de opciones
menu = ["Inicio", "Cargar Datos", "Preprocesamiento de Datos", "Análisis Estadístico", "Visualización de Datos", "Modelado", "Exportar Resultados"]
choice = st.sidebar.selectbox("Menú", menu)

# Manejo de sesión para manipular la información en cada módulo
if 'data' not in st.session_state:
    st.session_state.data = None

# 0. Inicio
if choice == "Inicio":
    st.subheader("Análisis de Datos y Machine Learning")

    st.image("tec_logo.png", width=200)
    st.title("Sistema de Análisis Financiero para Empresas y Negocios 📊")
    st.header("PROYECTO FINAL")
    st.text("Luis Enrique Batres Martinez - 20130806")
    st.markdown("**Objetivo:** Analizar el desempeño financiero de empresas mediante técnicas de Machine Learning y estadística.")

# 1. Cargar Datos
if choice == "Cargar Datos":
    st.subheader("Carga de Datos")
    
    # Cargar archivo
    uploaded_file = st.file_uploader("Subir archivo CSV o Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Leer CSV o Excel
            if uploaded_file.name.endswith(".csv"):
                st.session_state.data = pd.read_csv(uploaded_file)
            else:
                st.session_state.data = pd.read_excel(uploaded_file)

            st.success("Datos cargados exitosamente!")
            st.dataframe(st.session_state.data.head())  # Mostrar las primeras filas de los datos
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

# 2. Preprocesamiento de Datos
elif choice == "Preprocesamiento de Datos":
    st.subheader("Preprocesamiento de Datos")
    
    if st.session_state.data is not None:
        data = st.session_state.data.copy()  # Copia de datos para mantener el original

        st.write("Opciones:")
        
        # Limpiar valores nulos
        if st.checkbox("Eliminar valores nulos"):
            data = data.dropna()
            st.success("Valores nulos eliminados.")
            st.dataframe(data.head())
        
        if st.checkbox("Reemplazar valores nulos por 0"):
            data = data.fillna(0)
            st.success("Valores nulos reemplazados por 0.")
            st.dataframe(data.head())
        
        # Normalización de datos
        if st.checkbox("Normalizar datos"):
            num_cols = data.select_dtypes(include=['float64', 'int64']).columns
            data[num_cols] = (data[num_cols] - data[num_cols].min()) / (data[num_cols].max() - data[num_cols].min())
            st.success("Datos normalizados.")
            st.dataframe(data.head())
        
        # Actualizar los datos en session_state
        st.session_state.data = data
    else:
        st.warning("Primero cargue un archivo de datos en la sección 'Cargar Datos'.")

# 3. Análisis Estadístico
elif choice == "Análisis Estadístico":
    st.subheader("Análisis Estadístico")
    
    if st.session_state.data is not None:
        st.write("Estadísticas Descriptivas:")
        st.write(st.session_state.data.describe())

        st.write("Correlación entre variables financieras:")
        st.write(st.session_state.data.corr())
    else:
        st.warning("Primero cargue un archivo de datos en la sección 'Cargar Datos'.")

# 4. Visualización de Datos
elif choice == "Visualización de Datos":
    st.subheader("Visualización de Datos")

    if st.session_state.data is not None:
        # Visualización interactiva con Plotly
        st.write("Distribución de Ventas por Mes (Gráfico Interactivo)")
        fig = px.bar(st.session_state.data, x="Mes", y="Ventas", title="Distribución de Ventas por Mes")
        st.plotly_chart(fig)

        st.write("Mapa de Calor de Correlación entre Variables")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(st.session_state.data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Primero cargue un archivo de datos en la sección 'Cargar Datos'.")

# 5. Modelado de Machine Learning
elif choice == "Modelado":
    st.subheader("Modelado de Machine Learning")

    if st.session_state.data is not None:
        # Separar características y variable objetivo
        X = st.session_state.data[['Ingresos', 'Publicidad']]
        y = st.session_state.data['Ventas']

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenar modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Métrica de evaluación
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Error cuadrático medio (MSE): {mse:.2f}")
    else:
        st.warning("Primero cargue un archivo de datos en la sección 'Cargar Datos'.")

# 6. Exportar Resultados
elif choice == "Exportar Resultados":
    st.subheader("Exportación de Resultados")

    if st.session_state.data is not None:
        # Botón para exportar datos procesados
        st.download_button(
            label="📥 Descargar CSV",
            data=st.session_state.data.to_csv(index=False).encode('utf-8'),
            file_name="datos_analizados.csv",
            mime='text/csv'
        )

        st.download_button(
            label="📥 Descargar Excel",
            data=df_a_excel(st.session_state.data),  # Llamar a la función que convierte los datos a Excel
            file_name="datos_procesados.xlsx",  # Nombre del archivo de salida
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # MIME type para Excel
        )
    else:
        st.warning("Primero cargue un archivo de datos en la sección 'Cargar Datos'.")

# Pie de página con información del autor
st.markdown(
    """
    <div class="footer">
        <p>&copy; 2024 Luis Enrique Batres Martinez - Proyecto de Análisis Financiero</p>
    </div>
    """, unsafe_allow_html=True)
