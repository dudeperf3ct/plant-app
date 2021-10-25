import streamlit as st
import requests
import json
import pandas as pd

def main():

    st.write("Plant species classification:")
    st.write("Choose any plant Image and get the species of the plant:")

    Image = st.file_uploader("Choose an Image...", type=['jpg','jpeg','png'])
    if Image is not None:
        Image = Image.read()
        st.image(Image)

    menu = ['None', 'Pl@ntNet API']
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
                st.write('Querying failed...')



if __name__ == '__main__':
    main()