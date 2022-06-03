import streamlit as st
import requests
import json
import pandas as pd
import base64
from io import BytesIO
import calendar
import plotly.express as px
import logging

logger = logging.getLogger()
logger.disabled = False

from gbif_client import GbifClient


def plot_grid(filter_df, filtered_images):
    idx = 0
    for _ in range(len(filtered_images) - 1):
        col1, col2, col3, col4 = st.columns(4)

        if idx < len(filtered_images):
            with col1:
                st.write(idx)
                st.image(
                    filtered_images[idx],
                    width=150,
                    caption=f"{filter_df['name'].iloc[idx]}",
                )
                idx += 1
        if idx < len(filtered_images):
            with col2:
                st.write(idx)
                st.image(
                    filtered_images[idx],
                    width=150,
                    caption=f"{filter_df['name'].iloc[idx]}",
                )
                idx += 1
        if idx < len(filtered_images):
            with col3:
                st.write(idx)
                st.image(
                    filtered_images[idx],
                    width=150,
                    caption=f"{filter_df['name'].iloc[idx]}",
                )
                idx += 1
        if idx < len(filtered_images):
            with col4:
                st.write(idx)
                st.image(
                    filtered_images[idx],
                    width=150,
                    caption=f"{filter_df['name'].iloc[idx]}",
                )
                idx += 1
        else:
            break


def select_apis(base_image, key):
    menu = ["None", "Pl@ntNet API", "Plant.id API"]
    st.sidebar.header("API Selection")
    choice = st.sidebar.selectbox("Select any API", menu, key=key)

    if choice == "None":
        return ""

    if choice == "Pl@ntNet API" and base_image is not None:
        st.write("Querying Pl@ntNet API")
        api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={st.secrets['plantnet_key']}"

        files = [("images", (base_image))]

        req = requests.Request("POST", url=api_endpoint, files=files)
        prepared = req.prepare()

        s = requests.Session()
        response = s.send(prepared)
        json_result = json.loads(response.text)

        if response.status_code == 200:
            scores, common_names, family_names, genus_names, sci_names = (
                [],
                [],
                [],
                [],
                [],
            )
            st.write(f"Retriving top {len(json_result['results'])} results")
            for i in range(len(json_result["results"])):
                scores.append(json_result["results"][i]["score"])
                common_names.append(json_result["results"][i]["species"]["commonNames"])
                family_names.append(
                    json_result["results"][i]["species"]["family"]["scientificName"]
                )
                genus_names.append(
                    json_result["results"][i]["species"]["genus"]["scientificName"]
                )
                sci_names.append(json_result["results"][i]["species"]["scientificName"])
            df = pd.DataFrame(
                {
                    "scores": scores,
                    "commonNames": common_names,
                    "scientificName": sci_names,
                    "family": family_names,
                    "genus": genus_names,
                }
            )
            st.table(df)
            return df
        else:
            st.write(
                f"Querying failed with response status code {response.status_code}"
            )

    elif choice == "Plant.id API" and base_image is not None:
        st.write("Querying Plant.id API")

        params = {
            "api_key": st.secrets["plantid_key"],
            "images": [base64.b64encode(base_image).decode("ascii")],
            "modifiers": ["crops_fast"],
            "plant_language": "en",
            "plant_details": [
                "common_names",
                "edible_parts",
                "gbif_id" "name_authority",
                "propagation_methods",
                "synonyms",
                "structured_name",
                "taxonomy",
                "url",
                "wiki_description",
                "wiki_image",
            ],
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            "https://api.plant.id/v2/identify", json=params, headers=headers
        )

        json_result = response.json()

        if response.status_code == 200:
            (
                scores,
                common_names,
                wiki_url,
                genus_names,
                family_names,
                sci_names,
                taxo,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            st.write(f"Retriving top {len(json_result['suggestions'])} results")
            for i in range(len(json_result["suggestions"])):
                scores.append(json_result["suggestions"][i]["probability"])
                common_names.append(
                    json_result["suggestions"][i]["plant_details"]["common_names"]
                )
                sci_names.append(
                    json_result["suggestions"][i]["plant_details"]["scientific_name"]
                )
                genus_names.append(
                    json_result["suggestions"][i]["plant_details"]["structured_name"][
                        "genus"
                    ]
                )
                wiki_url.append(json_result["suggestions"][i]["plant_details"]["url"])
                # taxo.append(json_result["suggestions"][i]["plant_details"]["taxonomy"])
                family_names.append(
                    json_result["suggestions"][i]["plant_details"]["taxonomy"]["family"]
                )

            df = pd.DataFrame(
                {
                    "scores": scores,
                    "commonNames": common_names,
                    "scientificName": sci_names,
                    "family": family_names,
                    "genus": genus_names,
                    "Wikipedia URL": wiki_url,
                    # "taxonomy": taxo,
                }
            )
            st.table(df)
            return df
        else:
            st.write(
                f"Querying failed with response status code {response.status_code}"
            )


def add_seprator():
    st.markdown(
        """<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True,
    )


def center_image(img):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        st.image(img)
    with col3:
        st.write(" ")


def plot_basisrecord(data):
    st.write("Occurrences per Basis of record")
    sub_df = data["basisOfRecord"].value_counts().reset_index()
    sub_df.columns = ["baisOfRecord", "count"]
    fig = px.pie(sub_df, values="count", names="baisOfRecord")
    st.plotly_chart(fig, use_container_width=True)


def plot_stats(data, col):
    labels = None
    if col == "level0Name":
        st.write(f"Occurrences per top 10 countries")
    else:
        st.write(f"Occurrences per {col}")
    sub_df = data[col].value_counts().reset_index()
    sub_df.columns = [col, "count"]
    if col != "level0Name":
        sub_df = sub_df.sort_values(by=[col])
    if col == "month":
        sub_df[col] = sub_df[col].apply(lambda x: calendar.month_abbr[int(x)])
    if col == "level0Name":
        sub_df = sub_df[:10]
        labels = {"level0Name": "Country"}
    if labels is not None:
        fig = px.bar(sub_df, y="count", x=col, labels=labels)
    st.plotly_chart(fig, use_container_width=True)


def plot_map(data):
    st.write("Map of occurrences")
    lats = data["decimalLatitude"].values
    lons = data["decimalLongitude"].values
    sub_df = pd.DataFrame()
    sub_df["lat"] = lats
    sub_df["lon"] = lons
    sub_df = sub_df.dropna()
    st.map(sub_df)


def gbif_client(row):
    add_seprator()
    st.markdown(
        "<h3 style='text-align: center; color: red;'>GBIF</h3>",
        unsafe_allow_html=True,
    )
    st.write(f"Querying GBIF API for more information about {row['scientificName']}")
    st.write("")
    st.write("")
    st.write("")
    st.write("Taxonomy and Occurence count")
    scientific_name = row["scientificName"]
    family_name = row["family"]
    genus_name = row["genus"]
    gb = GbifClient(scientific_name, family_name, genus_name, limit=1)
    taxonomy = gb.get_taxonomy()
    gb_df = pd.DataFrame(
        {
            "Occurences": gb.get_count_occurences(),
            "Scientific Name": taxonomy["scientificName"],
            "Kingdom": taxonomy["kingdom"],
            "Phylum": taxonomy["phylum"],
            "Class": taxonomy["class"],
            "Order": taxonomy["order"],
            "Family": taxonomy["family"],
            "Genus": taxonomy["genus"],
        },
        index=[0],
    )
    st.table(gb_df)
    st.write("Similar Images")
    urls, titles = gb.get_similar_images()
    if len(urls) > 0 and len(titles) > 0:
        success = True
    else:
        success = False
        st.write("No similar images found")
    if success:
        plot_grid(titles, urls)
    with st.spinner(f"Working some magic..."):
        data = gb.download_dataset()
    if len(data) > 0:
        plot_basisrecord(data)
        plot_stats(data, "year")
        plot_stats(data, "month")
        plot_stats(data, "level0Name")
        plot_map(data)
    else:
        st.write("Oh no!! Download from GBIF unsuccessful. Try Again!")


def main():

    st.markdown(
        "<h2 style='text-align: center; color: red;'>Plant species classification</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: red;'>Choose any plant image and get the species of the plant</h4>",
        unsafe_allow_html=True,
    )

    base_image = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])
    if base_image is not None:
        base_image = base_image.read()
        st.image(base_image)

    menu = ["None", "YOLO Model", "No YOLO Model"]
    st.sidebar.header("Choose Models")
    choice = st.sidebar.selectbox("Select method for inference", menu)

    if choice == "YOLO Model":
        import io
        import torch
        from PIL import Image
        import numpy as np

        model = torch.hub.load(
            "ultralytics/yolov5", "custom", "models/yolov5s_weights.pt"
        )
        if base_image is not None:
            THRESH = st.slider("Threshold for YOLO detection", 0.0, 1.0, 0.5)
            with st.spinner(f"Applying AI magic..."):
                img = Image.open(io.BytesIO(base_image))
                img_arr = np.array(img)
                results = model(img)
            if results is not None:
                add_seprator()
                st.markdown(
                    "<h3 style='text-align: center; color: red;'>YOLO Model Detection</h3>",
                    unsafe_allow_html=True,
                )
                results.render()
                st.image(results.imgs)

                df = results.pandas().xyxy[0]
                filter_df = df[df["confidence"] > THRESH]
                if len(filter_df) > 0:
                    add_seprator()
                    st.write(f"Detection threshold set at {THRESH}")
                    st.write(f"Unique classes detected: {filter_df['name'].unique()}")
                    st.write(
                        f"Number of detections per unique class: {dict(filter_df['name'].value_counts())}"
                    )
                    filtered_images = []
                    for _, row in filter_df.iterrows():
                        xmin, xmax, ymin, ymax = (
                            int(row["xmin"]),
                            int(row["xmax"]),
                            int(row["ymin"]),
                            int(row["ymax"]),
                        )
                        extract_img = img_arr[ymin:ymax, xmin:xmax]
                        # extract_img_bin =  bytes(Image.fromarray(extract_img).tobytes())
                        filtered_images.append(extract_img)
                    add_seprator()
                    plot_grid(filter_df, filtered_images)
                    add_seprator()
                    img_options = np.arange(len(filtered_images))
                    st.sidebar.header("Identify")
                    choice = st.sidebar.selectbox("Select image", img_options)
                    st.write("")
                    st.markdown(
                        "<h3 style='text-align: center; color: red;'>Selected image for identification</h3>",
                        unsafe_allow_html=True,
                    )
                    center_image(filtered_images[choice])
                    pil_image = Image.fromarray(
                        np.uint8(filtered_images[choice])
                    ).convert("RGB")
                    buf = BytesIO()
                    pil_image.save(buf, format="JPEG")
                    df = select_apis(buf.getvalue(), "select_api")
                    if len(df) > 0:
                        gbif_client(df.iloc[0])
                else:
                    st.write(
                        f"No detection at confidence = {THRESH}. Lower the threshold"
                    )

    elif choice == "No YOLO Model":
        df = select_apis(base_image, "select_api")
        if len(df) > 0:
            gbif_client(df.iloc[0])


if __name__ == "__main__":
    main()
