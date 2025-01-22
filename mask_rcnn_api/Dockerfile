FROM tensorflow/tensorflow:2.4.0-gpu

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade --use-feature=2020-resolver -r  /app/requirements.txt 
RUN pip install uvicorn

COPY ./main.py /app/

CMD ["uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0"]