import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import streamlit.components.v1 as components

# Carregar os dados
file_path = 'houses_to_rent_v2.csv'  # Atualize o caminho do arquivo para o local correto
df = pd.read_csv(file_path)

# Configurações de Estilo
st.set_page_config(page_title='Análise de Aluguel de Imóveis', layout='wide')
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .main {
        background-color: #f0f2f6;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2E86C1;
        font-family: 'Roboto', sans-serif;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Roboto', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Título do Dashboard
st.title('🏠 Análise de Aluguel de Imóveis')
st.write('### Explore o mercado imobiliário de forma visual e intuitiva')

# Resumo dos dados
st.header('📊 Resumo dos Dados Estatísticos')
st.dataframe(df.describe(), use_container_width=True, height=500)

# Filtro por Cidade
st.sidebar.header('Filtros de Pesquisa')
st.sidebar.write('Utilize os filtros abaixo para explorar os dados disponíveis')
city = st.sidebar.selectbox('Selecione a cidade:', df['city'].unique())
filtered_data = df[df['city'] == city]

# Gráfico de Distribuição de Aluguel
st.header(f'📈 Distribuição dos Valores de Aluguel em {city}')
fig = px.histogram(filtered_data, x='rent amount (R$)', nbins=20, color_discrete_sequence=['#1f77b4'])
fig.update_layout(
    xaxis_title='Valor do Aluguel (R$)',
    yaxis_title='Quantidade',
    template='plotly_dark',
    title_text='Distribuição dos Valores de Aluguel',
    title_x=0.5,
    font=dict(family='Roboto, sans-serif', size=18),
    xaxis_side='top',
    yaxis_side='left'
)
fig.update_traces(
    marker_line_width=2,
    marker_line_color='black',
    texttemplate='%{y}',
    textposition='outside',
    marker=dict(opacity=0.75)
)
fig.add_vline(x=filtered_data['rent amount (R$)'].mean(), line_width=3, line_dash='dash', line_color='red', annotation_text='Valor Médio', annotation_position='top right')
fig.add_vline(x=filtered_data['rent amount (R$)'].median(), line_width=3, line_dash='dot', line_color='yellow', annotation_text='Valor Mediano', annotation_position='bottom right')
st.plotly_chart(fig)

# Comparação de Custo Total por Mobiliário
st.header('🛋️ Comparação do Custo Total por Tipo de Mobília')
furniture_group = df.groupby('furniture')['total (R$)'].mean().reset_index()
fig = px.bar(furniture_group, y='furniture', x='total (R$)', color='furniture', orientation='h', color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(
    xaxis_title='Custo Médio Total (R$)',
    yaxis_title='Tipo de Mobília',
    template='plotly_dark',
    title_text='Custo Médio Total por Tipo de Mobília',
    title_x=0.5,
    font=dict(family='Roboto, sans-serif', size=18),
    xaxis_side='top',
    yaxis_side='left'
)
fig.update_traces(
    marker_line_width=2,
    marker_line_color='black',
    texttemplate='%{x:.2f}',
    textposition='inside',
    marker=dict(opacity=0.85)
)
st.plotly_chart(fig)

# Análise do Impacto do Número de Quartos no Aluguel
st.header('🛏️ Impacto do Número de Quartos no Valor do Aluguel')
rooms_group = df.groupby('rooms')['rent amount (R$)'].mean().reset_index()
fig = px.line(rooms_group, x='rooms', y='rent amount (R$)', markers=True, color_discrete_sequence=['#2ca02c'])
fig.update_layout(
    xaxis_title='Quantidade de Quartos',
    yaxis_title='Valor Médio do Aluguel (R$)',
    template='plotly_dark',
    title_text='Valor Médio do Aluguel por Quantidade de Quartos',
    title_x=0.5,
    font=dict(family='Roboto, sans-serif', size=18),
    xaxis_side='top',
    yaxis_side='left'
)
fig.update_traces(
    line=dict(width=4),
    marker=dict(size=15, line=dict(width=3, color='black')),
    texttemplate='%{y:.2f}',
    textposition='top center'
)
st.plotly_chart(fig)

# Filtro para visualizar detalhes específicos
st.header('🔍 Detalhes Específicos dos Imóveis Selecionados')
selected_index = st.slider('Selecione o índice do imóvel:', 0, len(filtered_data) - 1, 0)
st.write(filtered_data.iloc[selected_index])

# Conclusão
st.header('📝 Considerações Finais')
st.write('Este dashboard permite explorar os padrões de aluguel de imóveis em diferentes cidades, analisando fatores como mobília, número de quartos, e custos adicionais. Assim, é possível identificar o que mais impacta o custo total do aluguel.')

# Nota de Acessibilidade
st.sidebar.header('Opções de Acessibilidade')
st.sidebar.write('Este dashboard foi desenvolvido com cores amigáveis para pessoas com daltonismo, utilizando paletas de alto contraste para uma melhor visualização.')