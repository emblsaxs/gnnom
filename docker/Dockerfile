FROM python:3.8
WORKDIR /docker
COPY env.yml env.yml
# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      python=3.8
      
RUN conda env update --file env.yml --name base

COPY . .

RUN python setup.py sdist

RUN tar -xvf dist/gnnom-1.0.tar.gz
RUN cd gnnom-1.0

CMD [ "python", "gnnom/apply_nn.py", "p", "mw", "testdata/SASDCF8.dat", "44.66", "2.74", "--n", "1000", "--mode", "DEBUG", "--units", "nanometer"]#, "--host=0.0.0.0"]
