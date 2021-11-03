import streamlit as st
import requests
import json
import pandas as pd
from pprint import pprint
import http.client
import urllib.request, urllib.parse, urllib.error
import base64


def main():

    st.write("Plant species classification:")
    st.write("Choose any plant Image and get the species of the plant:")

    Image = st.file_uploader("Choose an Image...", type=['jpg','jpeg','png'])
    if Image is not None:
        Image = Image.read()
        st.image(Image)

    menu = ['None', 'Pl@ntNet API', 'Plant.id API', 'Azure API']
    st.sidebar.header('API Selection')
    choice = st.sidebar.selectbox('Select any API', menu)

    if choice == 'Pl@ntNet API' and Image is not None:
        st.write('Querying Pl@ntNet API')
        api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={st.secrets['plantnet_key']}"

        st.sidebar.header('Select organ for identification')
        menu1 = ['leaf', 'flower']
        sub_choice = st.sidebar.selectbox('Select any of the following', menu1)

        if sub_choice is not None:
            data = {
                'organs': sub_choice
            }

            files = [
                ('images', (Image))
            ]
            
            req = requests.Request('POST', url=api_endpoint, files=files, data=data)
            prepared = req.prepare()

            s = requests.Session()
            response = s.send(prepared)
            json_result = json.loads(response.text)

            if response.status_code == 200:
                scores, common_names, family_names, genus_names, sci_names = [], [], [], [], []
                st.write(f"Retriving top {len(json_result['results'])} results")
                for i in range(len(json_result['results'])):
                    scores.append(json_result['results'][i]['score'])
                    common_names.append(json_result['results'][i]['species']['commonNames'])
                    family_names.append(json_result['results'][i]['species']['family']['scientificName'])
                    genus_names.append(json_result['results'][i]['species']['genus']['scientificName'])
                    sci_names.append(json_result['results'][i]['species']['scientificName'])
                df = pd.DataFrame(
                    {
                        'scores' : scores,
                        'commonNames': common_names,
                        'scientificName': sci_names,
                        'family': family_names,
                        'genus': genus_names
                    })
                st.table(df)
            else:
                st.write(f'Querying failed with response status code {response.status_code}')

    elif choice == 'Azure API' and Image is not None:
        st.write('Waiting for api key')
        topK, predictMode = 5, 'classifyAndDetect'

        headers = {
            # Request headers
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': '{subscription key}',
        }

        params = urllib.parse.urlencode({
            # Request parameters
            'topK': topK,
            'predictMode': predictMode,
        })

        try:
            conn = http.client.HTTPSConnection('aiforearth.azure-api.net')
            conn.request("POST", "/species-classification/v2.0/predict?%s" % params, "{Image}", headers)
            response = conn.getresponse()
            data = response.read()
            print(data)
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))

    elif choice == 'Plant.id API' and Image is not None:
        st.write("Querying Plant.id API")

        params = {
            "api_key": st.secrets['plantid_key'],
            "images": [base64.b64encode(Image).decode("ascii")],
            "modifiers": ["crops_fast"],
            "plant_language": "en",
            "plant_details": ["common_names",
                                "edible_parts",
                                "gbif_id"
                                "name_authority",
                                "propagation_methods",
                                "synonyms",
                                "structured_name",
                                "taxonomy",
                                "url",
                                "wiki_description",
                                "wiki_image",
                                ]
            }

        headers = {
            "Content-Type": "application/json"
            }

        response = requests.post("https://api.plant.id/v2/identify",
                                json=params,
                                headers=headers)

        json_result = response.json()

        if response.status_code == 200:
                scores, common_names, spec_names, genus_names, sci_names, taxo = [], [], [], [], [], []
                st.write(f"Retriving top {len(json_result['suggestions'])} results")
                for i in range(len(json_result['suggestions'])):
                    scores.append(json_result['suggestions'][i]['probability'])
                    common_names.append(json_result['suggestions'][i]['plant_details']['common_names'])
                    sci_names.append(json_result['suggestions'][i]['plant_details']['scientific_name'])
                    genus_names.append(json_result['suggestions'][i]['plant_details']['structured_name']['genus'])
                    spec_names.append(json_result['suggestions'][i]['plant_details']['structured_name']['species'])
                    taxo.append(json_result['suggestions'][i]['plant_details']['taxonomy'])

                df = pd.DataFrame(
                    {
                        'scores' : scores,
                        'commonNames': common_names,
                        'scientificName': sci_names,
                        'genus': genus_names,
                        'species': spec_names,
                        'taxonomy': taxo
                    })
                st.table(df)
        else:
            st.write(f'Querying failed with response status code {response.status_code}')

if __name__ == '__main__':
    main()