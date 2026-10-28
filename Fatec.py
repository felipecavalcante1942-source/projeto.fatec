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

# --- MÓDULO DBSCAN ---
st.divider()
st.header("Agrupamento de Dados (DBSCAN)")

try:
    # Agrupar por cliente para criar métricas de comportamento reais
    cluster_df = filtered.groupby("customer_id").agg(
        total_itens=("basket_count", "sum"),
        total_pedidos=("basket_count", "count")
    ).reset_index()

    X = cluster_df[["total_itens", "total_pedidos"]]

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Execução do DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    cluster_df["Cluster"] = dbscan.fit_predict(X_scaled)
    cluster_df["Agrupamento"] = cluster_df["Cluster"].apply(
        lambda x: "Outliers / Ruído" if x == -1 else f"Grupo {x}"
    )

    col_img, col_info = st.columns([1, 1])

    with col_img:
        try:
            st.image("grafico_cotovelo_dbscan.png", caption="K-Distance Plot (Escolha do eps)")
        except Exception:
            st.info("Imagem 'grafico_cotovelo_dbscan.png' não encontrada na pasta raiz.")

    with col_info:
        st.subheader("Resumo dos Agrupamentos")
        num_clusters = len(set(cluster_df["Cluster"])) - (1 if -1 in cluster_df["Cluster"] else 0)
        st.write(f"**Total de Clusters identificados:** {num_clusters}")
        st.write("**Métricas analisadas:** Total de Itens Comprados vs. Frequência de Pedidos por Cliente")

    # Gráfico de dispersão limpo
    fig_dbscan = px.scatter(
        cluster_df,
        x="total_pedidos",
        y="total_itens",
        color="Agrupamento",
        title="Perfil dos Clientes: Total de Pedidos vs Total de Itens",
        labels={"total_pedidos": "Total de Pedidos", "total_itens": "Total de Itens Comprados"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    st.plotly_chart(fig_dbscan, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao processar o agrupamento DBSCAN: {e}")
