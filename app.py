import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Mapa de calor del coeficiente de Pearson")

# Descripción breve
st.markdown(
    """Esta aplicación genera un diagrama de calor basado en los coeficientes de correlación de Pearson
    para un conjunto de datos con 10 atributos. Puedes usar el ejemplo por defecto o cargar tu propio archivo CSV."""
)

# Opción para usar datos de ejemplo o cargar un archivo CSV
data_option = st.radio("Selecciona una opción:", ("Datos de ejemplo", "Cargar archivo CSV"))

# Función para generar datos de ejemplo
def generate_example_data():
    np.random.seed(42)
    data = np.random.rand(100, 10)  # 100 filas, 10 atributos
    columns = [f"Atributo {i+1}" for i in range(10)]
    return pd.DataFrame(data, columns=columns)

# Obtener los datos según la opción seleccionada
if data_option == "Datos de ejemplo":
    df = generate_example_data()
    st.write("### Datos de ejemplo:")
    st.dataframe(df)
else:
    uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Datos cargados:")
        st.dataframe(df)
    else:
        st.warning("Por favor, carga un archivo CSV para continuar.")
        st.stop()

# Verificar que el DataFrame tiene al menos 2 columnas
if df.shape[1] < 2:
    st.error("El conjunto de datos debe tener al menos 2 columnas para calcular las correlaciones.")
    st.stop()

# Calcular la matriz de correlación
correlation_matrix = df.corr(method='pearson')

# Generar el diagrama de calor
st.write("### Mapa de calor del coeficiente de Pearson:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
plt.title("Mapa de calor de las correlaciones")
st.pyplot(fig)

# Pie de página
st.markdown("Hecho con ❤️ usando Streamlit, Seaborn y Matplotlib.")
