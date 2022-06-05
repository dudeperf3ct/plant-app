import requests
import streamlit as st
from pprint import pprint
from wolframclient.language import wlexpr
from wolframclient.evaluation import SecuredAuthenticationKey, WolframCloudSession

from wolfram_client import WolframClient


# def parse_name(output):
#     start = -1
#     for i, o in enumerate(output):
#         if o == "," and start == -1:
#             start = i
#             break
#     scientific_name = output[start + 3 : -2]
#     return scientific_name


# url = st.secrets["WOLFRAM_URL"]

# payload = {}
# files = [
#     (
#         "image",
#         (
#             "Lion_waiting_in_Namibia.jpg",
#             open("test_images/Lion_waiting_in_Namibia.jpg", "rb"),
#             "image/jpeg",
#         ),
#     )
# ]

# response = requests.request("POST", url, data=payload, files=files)

# print(response.text, response.status_code)
# parsed_name = parse_name(response.text)

# sak = SecuredAuthenticationKey(
#     st.secrets["ConsumerKey"],
#     st.secrets["ConsumerSecret"],
# )
# session = WolframCloudSession(credentials=sak)
# session.start()
# print(session.authorized())
# find_name = session.function(
#     wlexpr(
#         """
#     EntityValue[Entity["Concept", #1], "Name"] &
#     """
#     )
# )
# species_name = find_name(parsed_name)
# print(species_name)

# appid = st.secrets["APP_ID"]
# query_url = f"https://api.wolframalpha.com/v2/query?input={species_name}&format=plaintext&output=JSON&appid={appid}"

# r = requests.get(query_url).json()
# # print(r.text)
# pprint(r)
# print(r)


def main(img_path):
    # wl = WolframClient()
    # # species = wl.identify_image(img_path)
    # # print(species)
    # parsed_name = wl.find_entities(img_path)
    # sak = SecuredAuthenticationKey(
    #     st.secrets["ConsumerKey"],
    #     st.secrets["ConsumerSecret"],
    # )
    # session = WolframCloudSession(credentials=sak)
    # session.start()
    # print(session.authorized())
    # find_name = session.function(
    #     wlexpr(
    #         """
    #     EntityValue[Entity["Concept", #1], "Name"] &
    #     """
    #     )
    # )
    # species_name = find_name(parsed_name)
    # print(species_name)

    species_name = "lion"

    appid = st.secrets["APP_ID"]
    query_url = f"https://api.wolframalpha.com/v2/query?input={species_name}&format=plaintext&output=JSON&appid={appid}"

    json_result = requests.get(query_url).json()
    # print(r.text)
    if json_result["queryresult"]["success"]:
        sci_names, kingdom, phylum, kls, order, family, genus, species = (
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        for i in range(len(json_result["queryresult"]["pods"])):
            if (
                json_result["queryresult"]["pods"][i]["title"]
                == "Scientific name"
            ):
                # fmt: off
                sci_names = json_result["queryresult"]["pods"][i]["subpods"][0]["plaintext"]
            if (
                "Taxonomy" in json_result["queryresult"]["pods"][i]["title"]
            ):
                taxonomy = json_result["queryresult"]["pods"][i]["subpods"][0]["plaintext"]
                kingdom, phylum, kls, order, family, genus, species = taxonomy.split("\n")
                kingdom, phylum, kls, order, family, genus, species = kingdom.split("|")[-1].strip("\n"), phylum.split("|")[-1].strip("\n"), kls.split("|")[-1].strip("\n"), order.split("|")[-1].strip("\n"), family.split("|")[-1].strip("\n"), genus.split("|")[-1].strip("\n"), species.split("|")[-1].strip("\n")
        print(sci_names)
        print(kingdom, phylum, kls, order, family, genus, species)

if __name__ == "__main__":
    main("test_images/Lion_waiting_in_Namibia.jpg")