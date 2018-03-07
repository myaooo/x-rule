FROM python:3.5
ADD . /code
WORKDIR /code

# Install dependencies
#RUN pip install git+https://github.com/myaooo/pysbrl.git@master

# Add our code
ADD . /iml
WORKDIR /iml

RUN pip install --no-cache-dir /iml/vendors/pyfim/

RUN pip install --no-cache-dir  -r requirements.txt


# Expose is NOT supported by Heroku
# EXPOSE 5000

# Run the image as a non-root user
#RUN adduser -D myuser
#USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT iml.server:app