import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Carregar os dados
file_path = 'houses_to_rent_v2.csv'  # Atualize o caminho do arquivo para o local correto
df = pd.read_csv(file_path)

# Título do Dashboard
st.title('Análise de Aluguéis de Casas')

# Resumo dos dados
st.header('Resumo dos Dados')
st.write(df.describe())

# Filtro por Cidade
city = st.selectbox('Selecione a cidade:', df['city'].unique())
filtered_data = df[df['city'] == city]

# Gráfico de Distribuição de Aluguel
st.header(f'Distribuição dos Valores de Aluguel em {city}')
fig, ax = plt.subplots()
ax.hist(filtered_data['rent amount (R$)'], bins=20, color=mcolors.TABLEAU_COLORS['tab:blue'], edgecolor='black')
ax.set_xlabel('Valor do Aluguel (R$)')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição dos Valores de Aluguel')
st.pyplot(fig)

# Comparação de Custo Total por Mobiliário
st.header('Comparação do Custo Total por Tipo de Mobiliário')
furniture_group = df.groupby('furniture')['total (R$)'].mean().reset_index()
fig, ax = plt.subplots()
ax.bar(furniture_group['furniture'], furniture_group['total (R$)'], color=mcolors.TABLEAU_COLORS['tab:orange'])
ax.set_xlabel('Tipo de Mobiliário')
ax.set_ylabel('Custo Total Médio (R$)')
ax.set_title('Custo Total Médio por Tipo de Mobiliário')
st.pyplot(fig)

# Análise do Impacto do Número de Quartos no Aluguel
st.header('Impacto do Número de Quartos no Valor do Aluguel')
rooms_group = df.groupby('rooms')['rent amount (R$)'].mean().reset_index()
fig, ax = plt.subplots()
ax.plot(rooms_group['rooms'], rooms_group['rent amount (R$)'], marker='o', linestyle='-', color=mcolors.TABLEAU_COLORS['tab:green'])
ax.set_xlabel('Número de Quartos')
ax.set_ylabel('Valor Médio do Aluguel (R$)')
ax.set_title('Valor Médio do Aluguel por Número de Quartos')
st.pyplot(fig)

# Filtro para visualizar detalhes específicos
st.header('Detalhes Específicos dos Imóveis')
selected_index = st.slider('Selecione um índice de imóvel:', 0, len(filtered_data) - 1, 0)
st.write(filtered_data.iloc[selected_index])

# Conclusão
st.header('Conclusão')
st.write('Este dashboard permite explorar os padrões de aluguel de imóveis em diferentes cidades, analisando fatores como mobília, número de quartos, e custos adicionais. Assim, é possível identificar o que mais impacta o custo total do aluguel.')

# Nota de Acessibilidade
st.sidebar.header('Acessibilidade')
st.sidebar.write('Este dashboard foi desenvolvido com cores que são amigáveis para pessoas com daltonismo, usando a paleta de cores TABLEAU, que possui contrastes adequados para melhorar a visualização.')