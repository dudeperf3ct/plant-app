import pandas as pd
from zipfile import ZipFile
import os
import logging
import streamlit as st

logger = logging.getLogger()
logger.disabled = False

from pygbif import species
from pygbif import occurrences as occ


class GbifClient:
    def __init__(self, scientific_name, family_name, genus_name, limit=1) -> None:
        self.scientific_name = scientific_name
        self.family_name = family_name
        self.genus_name = genus_name
        self.limit = limit
        self.x = species.name_backbone(
            name=self.scientific_name,
            family=self.family_name,
            genus=self.genus_name,
            limit=self.limit,
        )
        self.y = occ.search(taxonKey=self.get_species_key(), limit=1)
        self.results = self.y["results"][0]
        print(self.x)
        print(self.y)
        print(self.results)

    def get_key(self) -> int:
        return self.results["key"]

    def get_dataset_key(self):
        return self.results["datasetKey"]

    def get_similar_images(self):
        media = self.results["media"]
        urls, extra_data = [], []
        if len(media) > 0:
            for m in media:
                if "jpeg" in m["identifier"]:
                    urls.append(m["identifier"])
                    extra_data.append(
                        {
                            "publisher": m["publisher"],
                            "creator": m["creator"],
                            "references": m["references"],
                        }
                    )
            extra_data_df = pd.DataFrame({"name": extra_data})
            return urls, extra_data_df
        else:
            return [], []

    def get_species_key(self) -> int:
        return self.x["speciesKey"]

    def get_taxonomy(self) -> dict:
        taxonomy = dict()
        taxonomy["kingdom"] = self.x.get("kingdom", None)
        taxonomy["phylum"] = self.x.get("phylum", None)
        taxonomy["order"] = self.x.get("order", None)
        taxonomy["family"] = self.x.get("family", None)
        taxonomy["genus"] = self.x.get("genus", None)
        taxonomy["species"] = self.x.get("species", None)
        taxonomy["class"] = self.x.get("class", None)
        taxonomy["scientificName"] = self.x.get("scientificName", None)
        return taxonomy

    def get_count_occurences(self):
        return occ.count(self.get_species_key())

    def download_dataset(self):
        request = f"taxonKey = {self.get_species_key()}"
        download_key, _ = occ.download(
            queries=request,
            user=st.secrets["GBIF_USER"],
            pwd=st.secrets["GBIF_PWD"],
            email=st.secrets["GBIF_EMAIL"],
        )
        try:
            req = occ.download_get(download_key)
        except Exception as e:
            return []
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
        return data
