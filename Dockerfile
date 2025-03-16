FROM python:3.9

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./main.py /app/

CMD ["fastapi", "run", "main.py", "--port", "4000", "--host", "0.0.0.0"]