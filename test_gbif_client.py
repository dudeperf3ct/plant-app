import pandas as pd
import logging
import plotly.express as px
import calendar
from zipfile import ZipFile
import os
import streamlit as st
from pygbif import occurrences as occ

logger = logging.getLogger()
logger.disabled = True

from gbif_client import GbifClient


def main(scientific_name, family_name, genus_name, limit=1):
    gb = GbifClient(scientific_name, family_name, genus_name, limit)
    taxonomy = gb.get_taxonomy()
    gb_df = pd.DataFrame(
        {
            "Occurences": gb.get_count_occurences(),
            "Scientific bame": taxonomy["scientificName"],
            "Kingdom": taxonomy["kingdom"],
            "Phylum": taxonomy["phylum"],
            "Class": taxonomy["class"],
            "Order": taxonomy["order"],
            "Family": taxonomy["family"],
            "Genus": taxonomy["genus"],
        },
        index=[0],
    )
    print(gb_df)
    print(gb.get_similar_images())
    data = gb.download_data()
    filename = "occurrence.txt"
    folder_name = "0335536-210914110416597"
    data = pd.read_csv(
        folder_name + "/" + filename,
        sep="\t",
    )
    row = data.iloc[33]
    print(row.values)
    # for i, r in enumerate(row):
    #     if r == -26.056688:
    #         print(i)
    col = "level0Name"
    sub_df = data[col].value_counts().reset_index()
    sub_df.columns = [col, "count"]
    # sub_df = sub_df.sort_values(by=[col], ascending=False)
    sub_df = sub_df[:10]
    fig = px.bar(sub_df, y="count", x=col)
    fig.show()

    lats = data["decimalLatitude"].values
    lons = data["decimalLongitude"].values
    sub_df = pd.DataFrame()
    sub_df["lat"] = lats
    sub_df["lon"] = lons
    sub_df = sub_df.dropna()
    # st.map(sub_df)
    request = "taxonKey = 306404"
    download_key, _ = occ.download(
        queries=request,
        user=st.secrets["GBIF_USER"],
        pwd=st.secrets["GBIF_PWD"],
        email=st.secrets["GBIF_EMAIL"],
    )
    req = occ.download_get(download_key)
    filename = "occurrence.txt"
    folder_name = req["key"]
    zip = ZipFile(f"{folder_name}.zip")
    zip.extractall(f"./{folder_name}")
    zip.close()
    os.remove(f"{folder_name}.zip")
    data = pd.read_csv(
        folder_name + "/" + filename,
        sep="\t",
    )


if __name__ == "__main__":
    scientific_name = "Ipomoea purpurea (L.) Roth"
    family_name = "Convolvulaceae"
    genus_name = "Ipomoea"
    main(scientific_name, family_name, genus_name, limit=1)
