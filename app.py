import streamlit as st
import pandas as pd
import torch
from torch_geometric.utils import to_networkx

import os

# Imposta sempre tema chiaro creando/modificando il config.toml
config_dir = os.path.expanduser("~/.streamlit")
os.makedirs(config_dir, exist_ok=True)
with open(os.path.join(config_dir, "config.toml"), "w") as f:
    f.write("[theme]\nbase=\"light\"\n")



# CSS per sfondo
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)),
        url("https://wallpapers.com/images/hd/data-science-concepts-collage-415i6eboc97470w6.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* 🔹 Testi in nero */
body, p, div, h1, h2, h3, h4, h5, h6, span {
    color: black !important;
}


/* 🔹 Box bianco SOLO per selectbox */
.stSelectbox {
    background-color: rgba(255,255,255,1);  /* bianco pieno */
    padding: 8px;
    border-radius: 8px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
}


</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# Caricare i dati
data = torch.load("/Users/vincenzocamerlengo/Desktop/tesi/GAT_2/19_08/graph_data.pt")

# Caricare le predizioni salvate
@st.cache_data
def load_predictions():
    df = pd.read_csv("/Users/vincenzocamerlengo/Desktop/tesi/GAT_2/19_08/top_link_predictions_vicini.csv")
    df["recommended_nodes"] = df["recommended_nodes"].apply(eval)  # Converte stringhe in liste di tuple
    return df

# Caricare il grafo originale
@st.cache_data
def load_graph(data):
    return to_networkx(data, to_undirected=True)

# Titolo dell'app
#st.title("Link Prediction")
st.markdown(
    """
    <h1 style='text-align: center; 
               font-size: 50px; 
               font-family: Georgia; 
               color: black;'>
        LINK PREDICTION
    </h1>
    """,
    unsafe_allow_html=True
)


# Caricare i dati delle predizioni
df = load_predictions()

# Selezione del nodo
node_list = df["node_id"].tolist()
selected_node = st.selectbox("Select a node:", node_list)



# Mostrare i risultati
if selected_node in df["node_id"].values:
    recommendations = df[df["node_id"] == selected_node]["recommended_nodes"].values[0]
    
    st.subheader(f"🔍 Recommendation for node {selected_node}:")
    for rec in recommendations:
        #st.write(f"🔹 Nodo {rec[0]} → Probabilità: {rec[1]:.2f}")
        st.write(f"🔹 Node {rec[0]}")
