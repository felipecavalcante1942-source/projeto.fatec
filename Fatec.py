import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-Commerce Sales Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / "basket_details.csv"

    df = pd.read_csv(CSV_PATH)
    df["basket_date"] = pd.to_datetime(df["basket_date"], errors="coerce")
    return df.dropna(subset=["basket_date", "basket_count"])

df = load_data()

st.title("📊 E-Commerce Sales Dashboard")
st.caption("Análise exploratória da base E-Commerce Sales Dataset")

st.sidebar.header("Filtros")
min_date = df["basket_date"].min().date()
max_date = df["basket_date"].max().date()
date_range = st.sidebar.date_input("Período", (min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = df[
    (df["basket_date"].dt.date >= start_date) &
    (df["basket_date"].dt.date <= end_date)
].copy()

product_options = sorted(filtered["product_id"].dropna().unique())
selected_products = st.sidebar.multiselect("Produtos", product_options)
if selected_products:
    filtered = filtered[filtered["product_id"].isin(selected_products)]

customer_options = sorted(filtered["customer_id"].dropna().unique())
selected_customers = st.sidebar.multiselect("Clientes", customer_options)
if selected_customers:
    filtered = filtered[filtered["customer_id"].isin(selected_customers)]

if filtered.empty:
    st.warning("Nenhum registro encontrado com os filtros selecionados.")
    st.stop()

transactions = len(filtered)
total_items = int(filtered["basket_count"].sum())
avg_basket = filtered["basket_count"].mean()
customers = filtered["customer_id"].nunique()
products = filtered["product_id"].nunique()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transações", f"{transactions:,}".replace(",", "."))
c2.metric("Itens vendidos", f"{total_items:,}".replace(",", "."))
c3.metric("Média por cesta", f"{avg_basket:.2f}")
c4.metric("Clientes", f"{customers:,}".replace(",", "."))
c5.metric("Produtos", f"{products:,}".replace(",", "."))

st.divider()

st.subheader("📋 Dataset Filtrado")
st.dataframe(filtered, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    daily = filtered.groupby("basket_date", as_index=False).agg(
        transacoes=("basket_count", "size"),
        itens=("basket_count", "sum")
    )
    fig_daily = px.line(
        daily,
        x="basket_date",
        y=["transacoes", "itens"],
        markers=True,
        title="Evolução diária de transações e itens"
    )
    fig_daily.update_layout(xaxis_title="Data", yaxis_title="Quantidade")
    st.plotly_chart(fig_daily, use_container_width=True)

with col2:
    basket_counts = filtered["basket_count"].value_counts().sort_index().reset_index()
    basket_counts.columns = ["basket_count", "frequencia"]
    fig_hist = px.bar(
        basket_counts,
        x="basket_count",
        y="frequencia",
        title="Distribuição da quantidade de itens por cesta",
        text_auto=True
    )
    fig_hist.update_layout(xaxis_title="Itens por cesta", yaxis_title="Frequência")
    st.plotly_chart(fig_hist, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    top_products = (
        filtered.groupby("product_id", as_index=False)["basket_count"]
        .sum()
        .sort_values("basket_count", ascending=False)
        .head(10)
    )
    fig_products = px.bar(
        top_products.sort_values("basket_count"),
        x="basket_count",
        y="product_id",
        orientation="h",
        title="Top 10 produtos por quantidade de itens",
        text_auto=True
    )
    fig_products.update_layout(xaxis_title="Itens", yaxis_title="Produto")
    st.plotly_chart(fig_products, use_container_width=True)

with col4:
    top_customers = (
        filtered.groupby("customer_id", as_index=False)["basket_count"]
        .sum()
        .sort_values("basket_count", ascending=False)
        .head(10)
    )
    fig_customers = px.bar(
        top_customers.sort_values("basket_count"),
        x="basket_count",
        y="customer_id",
        orientation="h",
        title="Top 10 clientes por quantidade de itens",
        text_auto=True
    )
    fig_customers.update_layout(xaxis_title="Itens", yaxis_title="Cliente")
    st.plotly_chart(fig_customers, use_container_width=True)

st.divider()
st.subheader("Resumo estatístico")
summary = filtered["basket_count"].describe().rename({
    "count": "Quantidade",
    "mean": "Média",
    "std": "Desvio padrão",
    "min": "Mínimo",
    "25%": "1º quartil",
    "50%": "Mediana",
    "75%": "3º quartil",
    "max": "Máximo"
})
st.dataframe(summary.to_frame("Valor"), use_container_width=True)

st.subheader("Amostra dos dados")
st.dataframe(filtered.head(100), use_container_width=True)

# --- MÓDULO DBSCAN (OTIMIZADO) ---
st.header("Agrupamento de Dados (DBSCAN)")

# 1. Filtrar apenas colunas de comportamento de compra (ignorando IDs)
colunas_comportamento = [c for c in ['item_count', 'price', 'order_score', 'total_amount'] if c in df.columns]

# Caso não encontre essas colunas com os nomes exatos, pega as numéricas ignorando IDs
if len(colunas_comportamento) < 2:
    colunas_comportamento = [c for c in df.select_dtypes(include=[np.number]).columns if 'id' not in c.lower()]

X = df[colunas_comportamento].dropna()

# 2. Padronização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN com eps maior (1.2) e min_samples (15) para gerar menos clusters
dbscan = DBSCAN(eps=1.2, min_samples=15)
df_clean = df.loc[X.index].copy()
df_clean['Cluster'] = dbscan.fit_predict(X_scaled)

# Mapear nome dos clusters para ficar legível
df_clean['Cluster_Nome'] = df_clean['Cluster'].apply(lambda x: 'Ruído / Outlier' if x == -1 else f'Grupo {x}')

# Exibição do gráfico do cotovelo lado a lado com métricas
col_img, col_info = st.columns([1, 1])
with col_img:
    st.image("grafico_cotovelo_dbscan.png", caption="K-Distance Plot (Escolha do eps)")
with col_info:
    st.subheader("Resumo dos Agrupamentos")
    st.write(f"**Total de Clusters identificados:** {len(set(df_clean['Cluster'])) - (1 if -1 in df_clean['Cluster'] else 0)}")
    st.write(f"**Atributos analisados:** {', '.join(colunas_comportamento)}")

# 4. Gráfico de dispersão limpo e customizado (Dark Theme combinando com o Streamlit)
eixo_x = colunas_comportamento[0]
eixo_y = colunas_comportamento[1]

fig = px.scatter(
    df_clean, 
    x=eixo_x, 
    y=eixo_y, 
    color='Cluster_Nome',
    title=f"Distribuição de Clusters: {eixo_x.capitalize()} vs {eixo_y.capitalize()}",
    color_discrete_sequence=px.colors.qualitative.Bold,
    template='plotly_dark'  # Garante fundo escuro combinando com a interface
)

fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(
    legend_title_text='Agrupamentos',
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)
