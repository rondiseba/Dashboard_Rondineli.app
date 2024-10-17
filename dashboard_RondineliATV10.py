import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium
from geopy.geocoders import Nominatim
import datetime
import os
import pickle

# Configuração inicial do Streamlit
st.set_page_config(page_title="Dashboard Rondineli ATV10", layout="wide", initial_sidebar_state="expanded", page_icon="/path/to/your/icon.png")
st.title("Analise e previsão de Aluguel de imóveis")
st.markdown("### Análise de dados para suporte à tomada de decisão - Criado por Rondineli Seba | Python | Streamlit | CSS e HTML")

# Adicionar seleção de tema
tema = st.sidebar.radio('Selecione o Tema:', ['Claro', 'Escuro'])

if tema == 'Claro':
    st.markdown(
        """
        <style>
            .main {
                background-color: #ffffff;
                color: #000000;
            }
            .stButton>button {
                background-color: #007bff;
                color: white;
            }
            .stMetric {
                color: #007bff;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
elif tema == 'Escuro':
    st.markdown(
        """
        <style>
            .main {
                background-color: #2c2c2c;
                color: #ffffff;
            }
            .stButton>button {
                background-color: #000000;
                color: white;
            }
            .stMetric {
                color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Carregar os dados
df = pd.read_csv('houses_to_rent_v2.csv')

# Criar novas features
df['area_per_room'] = df['area'] / df['rooms']

# Adicionar coluna de mês para análise de sazonalidade
if 'date' in df.columns:
    df['rental_month'] = pd.to_datetime(df['date']).dt.month

# Adicionar coordenadas geográficas usando o Geopy com cache
cache_file = 'city_coordinates.pkl'
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        city_coordinates = pickle.load(f)
else:
    city_coordinates = {}

geolocator = Nominatim(user_agent="geoapiExercises")

latitudes = []
longitudes = []

for city in df['city'].unique():
    if city in city_coordinates:
        lat, lon = city_coordinates[city]
    else:
        try:
            location = geolocator.geocode(city)
            if location:
                lat, lon = location.latitude, location.longitude
                city_coordinates[city] = (lat, lon)
            else:
                lat, lon = None, None
        except Exception as e:
            lat, lon = None, None
    latitudes.append(lat)
    longitudes.append(lon)

# Salvar as coordenadas em cache
with open(cache_file, 'wb') as f:
    pickle.dump(city_coordinates, f)

# Mapear as coordenadas para o DataFrame
df['latitude'] = df['city'].map(dict(zip(df['city'].unique(), latitudes)))
df['longitude'] = df['city'].map(dict(zip(df['city'].unique(), longitudes)))

# Layout principal do dashboard
col1, col2 = st.columns([1, 2])

# Seção da barra lateral
st.sidebar.header("Opções de Filtro")
st.sidebar.markdown("Utilize os filtros abaixo para personalizar a visualização dos dados.")
cidade = st.sidebar.selectbox("Selecione a Cidade:", options=df['city'].unique())
df_filtrado = df[df['city'] == cidade]

test_size = st.sidebar.slider("Tamanho do conjunto de teste (em %):", 10, 50, 20)

# Seção de Métricas principais (à esquerda)
with col1:
    st.markdown(f"### Métricas para {cidade}")
    
    st.metric("Média de Aluguel (R$)", f"R$ {df_filtrado['rent amount (R$)'].mean():.2f}")
    st.metric("Máximo de Aluguel (R$)", f"R$ {df_filtrado['rent amount (R$)'].max():.2f}")
    st.metric("Mínimo de Aluguel (R$)", f"R$ {df_filtrado['rent amount (R$)'].min():.2f}")
    st.metric("Desvio Padrão do Aluguel (R$)", f"R$ {df_filtrado['rent amount (R$)'].std():.2f}")
    st.metric("Média do Condomínio (HOA) (R$)", f"R$ {df_filtrado['hoa (R$)'].mean():.2f}")
    st.metric("Média do IPTU (Taxa de Propriedade) (R$)", f"R$ {df_filtrado['property tax (R$)'].mean():.2f}")
    
    st.sidebar.markdown("### Simule um Aluguel")
    input_area = st.sidebar.number_input("Área (m²):", min_value=0, max_value=1000, value=100)
    input_rooms = st.sidebar.number_input("Número de Quartos:", min_value=0, max_value=10, value=2)
    input_bathroom = st.sidebar.number_input("Número de Banheiros:", min_value=0, max_value=10, value=1)
    input_parking = st.sidebar.number_input("Número de Vagas de Garagem:", min_value=0, max_value=5, value=1)
    input_area_per_room = input_area / input_rooms if input_rooms > 0 else 0

    previsao = RandomForestRegressor().fit(df[['area', 'rooms', 'bathroom', 'parking spaces', 'area_per_room']], df['rent amount (R$)']).predict([[input_area, input_rooms, input_bathroom, input_parking, input_area_per_room]])[0]
    st.sidebar.write(f"Previsão do Aluguel: R$ {previsao:.2f}")

# Treinamento de um modelo de machine learning para previsão de aluguel
st.markdown("### Previsão de Aluguel Utilizando Machine Learning")
X = df[['area', 'rooms', 'bathroom', 'parking spaces', 'area_per_room']]
y = df['rent amount (R$)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Modelo RandomForest
modelo_rf = RandomForestRegressor()
modelo_rf.fit(X_train, y_train)
predicoes_rf = modelo_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, predicoes_rf))
st.write(f"RMSE do Modelo Random Forest: {rmse_rf:.2f}")

# Modelo Gradient Boosting
modelo_gb = GradientBoostingRegressor()
modelo_gb.fit(X_train, y_train)
predicoes_gb = modelo_gb.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, predicoes_gb))
st.write(f"RMSE do Modelo Gradient Boosting: {rmse_gb:.2f}")

# Validação Cruzada com Ridge Regression
modelo_ridge = Ridge(alpha=1.0)
scores = cross_val_score(modelo_ridge, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_ridge = np.sqrt(-scores.mean())
st.write(f"RMSE Médio do Modelo Ridge (Validação Cruzada): {rmse_ridge:.2f}")

# Gráficos de análise e importância das variáveis (alinhados lado a lado)
with col2:
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Importância das Variáveis para o Modelo Random Forest")
        importancias = modelo_rf.feature_importances_
        features = X.columns
        fig3, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=importancias, y=features, ax=ax, palette='Blues')
        ax.set_title("Importância das Variáveis")
        ax.set_xlabel("Importância")
        ax.set_ylabel("Variáveis")
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"### Distribuição dos Preços de Aluguel na cidade de {cidade}")
        fig1 = px.histogram(df_filtrado, x='rent amount (R$)', nbins=30, title='Distribuição dos Preços de Aluguel', color_discrete_sequence=['#636EFA'], barmode='overlay')
        fig1.update_traces(marker_line_width=1.5, marker_line_color='white')
        st.plotly_chart(fig1)

# Gráficos de análise mais detalhados
st.markdown(f"### Análise Detalhada para {cidade}")
col5, col6 = st.columns(2)
with col5:
    fig2 = px.scatter(df_filtrado, x='area', y='rent amount (R$)', title='Relação entre Área e Aluguel', color='rent amount (R$)', color_continuous_scale=px.colors.sequential.Blues)
    st.plotly_chart(fig2)
with col6:
    fig4 = px.box(df, x='city', y='rent amount (R$)', title='Comparação de Aluguel Entre Cidades', color='city', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig4)

col7, col8 = st.columns(2)
with col7:
    fig5 = px.box(df, x='rooms', y='rent amount (R$)', title='Relação Entre Número de Quartos e Aluguel', color='rooms', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig5)
with col8:
    fig6 = px.box(df, x='animal', y='rent amount (R$)', title='Impacto da Permissão de Animais no Valor do Aluguel', color='animal', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig6)

# Gráfico de Análise de Preço por Mobiliado
st.markdown("### Comparação de Preços por Mobiliado")
fig_mobiliado = px.box(df, x='furniture', y='rent amount (R$)', title='Comparação de Preço por Mobiliado', color='furniture', color_discrete_sequence=px.colors.qualitative.Set3)
st.plotly_chart(fig_mobiliado)

# Clusterização detalhada
st.markdown("### Clusterização Detalhada dos Imóveis")
kmeans = KMeans(n_clusters=4)
df_filtrado['cluster'] = kmeans.fit_predict(df_filtrado[['area', 'rooms', 'bathroom', 'parking spaces']])
fig_cluster = px.scatter(df_filtrado, x='area', y='rent amount (R$)', color='cluster', title='Clusterização Detalhada dos Imóveis', color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig_cluster)

# Gráficos de análise comparativa adicionais
st.markdown("### Análise Comparativa Adicional")
col9, col10 = st.columns(2)

# Comparação entre cidades por média de aluguel e média do condomínio
with col9:
    df_numeric = df.select_dtypes(include=[np.number])
    df_grouped = df_numeric.join(df['city']).groupby('city').mean().reset_index()
    fig7 = px.bar(df_grouped, x='city', y=['rent amount (R$)', 'hoa (R$)'], title='Comparação de Média de Aluguel e Condomínio por Cidade', barmode='group', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig7)

# Comparação entre número de banheiros e valor de aluguel
with col10:
    fig8 = px.box(df, x='bathroom', y='rent amount (R$)', title='Relação Entre Número de Banheiros e Aluguel', color='bathroom', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig8)

# Comparação de sazonalidade dentro da mesma cidade e entre cidades
st.markdown("### Comparação de Sazonalidade do Aluguel")
col11, col12 = st.columns(2)
with col11:
    if 'rental_month' in df_filtrado.columns:
        fig_sazonal_cidade = px.line(df_filtrado, x='rental_month', y='rent amount (R$)', title=f'Variação Sazonal do Aluguel em {cidade}', markers=True)
        st.plotly_chart(fig_sazonal_cidade)

with col12:
    if 'rental_month' in df.columns:
        df_sazonal = df.groupby(['city', 'rental_month']).mean().reset_index()
        fig_sazonal_cidades = px.line(df_sazonal, x='rental_month', y='rent amount (R$)', color='city', title='Comparação de Sazonalidade do Aluguel Entre Cidades', markers=True)
        st.plotly_chart(fig_sazonal_cidades)

# Mapa interativo com coordenadas
st.markdown("### Mapa Interativo dos Aluguéis por Cidade")
if 'latitude' in df.columns and 'longitude' in df.columns:
    mapa = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
    for _, row in df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Cidade: {row['city']}\nValor do Aluguel: R$ {row['rent amount (R$)']:.2f}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(mapa)
    folium_static(mapa)

# Estilo e Design
st.markdown("""
<style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .css-18e3th9 {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 5px;
    }
    .stMetric {
        font-size: 1.2em;
        font-weight: bold;
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)