FROM python:3.8-slim-bullseye
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8051
COPY . /app/
ENTRYPOINT [ "streamlit", "run"]
CMD [ "app.py" ]