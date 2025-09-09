import streamlit as st
import pandas as pd
import torch
import gdown
import urllib.request
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


# File 1: graph_data.pt
FILE1_ID = "1bjPOKNI40uSPTWYQzmZ7KqRG-IbW7e6G"
URL1 = f"https://drive.google.com/uc?export=download&id={FILE1_ID}"
FILE1_NAME = "graph_data.pt"

# File 2: top_link_predictions_vicini.csv
FILE2_ID = "1bW-_zIBABLp_oDDSsOjsQfLSkAh0pjUw"
URL2 = f"https://drive.google.com/uc?export=download&id={FILE2_ID}"
FILE2_NAME = "top_link_predictions_vicini.csv"

# Funzione helper per scaricare se non esiste
def fetch_if_not_exists(url: str, filename: str):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

# Scarica i file (solo la prima volta)
fetch_if_not_exists(URL1, FILE1_NAME)
fetch_if_not_exists(URL2, FILE2_NAME)


# Caricare i dati
data = torch.load(FILE1_NAME, map_location=torch.device("cpu"))

# Caricare le predizioni salvate
@st.cache_data
def load_predictions():
    df = pd.read_csv(FILE2_NAME)
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
