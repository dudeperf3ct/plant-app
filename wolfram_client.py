import requests
import streamlit as st


def parse_name(output):
    start = -1
    for i, o in enumerate(output):
        if o == "," and start == -1:
            start = i
            break
    scientific_name = output[start + 3 : -2]
    return scientific_name


class WolframClient:
    def __init__(self) -> None:
        self.url = st.secrets["WOLFRAM_URL"]

    def find_entities(self, img_path):
        payload = {}
        files = [
            (
                "image",
                (
                    "tmp.{}".format(img_path.split(".")[-1]),
                    open(f"{img_path}", "rb"),
                    "image/*",
                ),
            )
        ]
        response = requests.request("POST", self.url, data=payload, files=files)
        print(response.text, response.status_code)
        parsed_name = parse_name(response.text)
        print(parsed_name)
        return parsed_name
