# syntax=docker/dockerfile:1
FROM python:3.10
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# download the pretrained model
# change here to download your pretrained model
# RUN curl -O "https://drive.google.com/uc?export=download&id=1Ls4ao5xHiMIasqzGDgI7fAFxVOjzvgOp&confirm=t&uuid=06cfacda-73b8-4ae8-8eca-4b30c7f37659&at=AKKF8vzyVfx5U_gwzfnYlkREbkPk:1683118319248"

EXPOSE 80
COPY . .
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]