FROM amaksimov/python_data_science
RUN apt-get update
RUN apt-get install -y ffmpeg

WORKDIR /home
COPY support_files/* ./

RUN chmod +x download_da-tacos.sh
RUN chmod +x download_covers80.sh

RUN pip3 install --upgrade pip
RUN pip3 install wget gdown
RUN pip3 install git+https://github.com/arthurtofani/acoss.git@readme_encoding
RUN pip3 install librosa==0.7.2
RUN pip3 install numba deepdish psutil
RUN pip3 install cython
RUN pip3 install madmom essentia


#CMD ["jupyter", "lab", "--allow-root"]
