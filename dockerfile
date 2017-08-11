FROM centos:latest
MAINTAINER CVUSK

ENV container docker
RUN (cd /lib/systemd/system/sysinit.target.wants/; for i in *; do [ $i == systemd-tmpfiles-setup.service ] || rm -f $i; done); \
rm -f /lib/systemd/system/multi-user.target.wants/*;\
rm -f /etc/systemd/system/*.wants/*;\
rm -f /lib/systemd/system/local-fs.target.wants/*; \
rm -f /lib/systemd/system/sockets.target.wants/*udev*; \
rm -f /lib/systemd/system/sockets.target.wants/*initctl*; \
rm -f /lib/systemd/system/basic.target.wants/*;\
rm -f /lib/systemd/system/anaconda.target.wants/*;
VOLUME [ "/sys/fs/cgroup" ]

# yum install packages
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
RUN yum -y groupinstall "Development Tools"
RUN yum -y install epel-release && \
    yum -y install python-devel python-pip python-dev python-virtualenv zip && \
    yum -y install wget bzip2 java-1.8.0-openjdk java-1.8.0-openjdk-devel tar unzip libXdmcp && \
    yum -y install vi vim sudo yum-utils policycoreutils selinux-policy-targeted openssh PyQt4 && \
    yum -y install openmpi openmpi-devel 
RUN yum -y update
RUN yum clean all
ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0
ENV PATH=/usr/lib64/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH


WORKDIR /opt/
RUN wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
RUN bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p /opt/anaconda3
ENV PATH="/opt/anaconda3/bin:$PATH"
RUN conda install -c http://conda.binstar.org/menpo opencv
RUN conda create -n py27 python=2.7 anaconda
RUN source activate py27
RUN conda install notebook ipykernel
RUN ipython kernel install --user
RUN conda install -c http://conda.binstar.org/menpo opencv
RUN source deactivate
RUN ipython kernel install --user

RUN conda install --channel https://conda.anaconda.org/conda-forge pyqt
RUN conda install -c anaconda pyqt=4.11.4

# pip install python libraries
RUN pip install --upgrade pip
RUN pip install pandas scipy jupyter tensorflow keras flask wtforms && \
    pip install scikit-learn matplotlib Pillow && \
    pip install graphviz requests py4j seaborn GPy && \
    pip install gpyopt pulp ortoolpy 
RUN conda update scipy
RUN conda install --channel https://conda.anaconda.org/cachemeorg funcdesigner openopt
RUN pip install cvxopt gensim
RUN pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp35-cp35m-linux_x86_64.whl
RUN python -m cntk.sample_installer
RUN pip install dask --upgrade
RUN yum -y update
RUN yum clean all

# generate jupyter notebook config
# BE SURE to change NotebookApp.token variable to your password
RUN jupyter notebook --generate-config  --allow-root && \
    ipython profile create
RUN echo "c.NotebookApp.ip = '*'" >>/root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >>/root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >>/root/.jupyter/jupyter_notebook_config.py && \
    echo "c.InteractiveShellApp.matplotlib = 'inline'" >>/root/.ipython/profile_default/ipython_config.py && \
    echo "c.NotebookApp.notebook_dir = '/opt/files/python/'" >>/root/.jupyter/jupyter_notebook_config.py

RUN mkdir -p /opt/
RUN echo 'nohup jupyter notebook --allow-root &' >> /opt/run_jupyter.sh

WORKDIR /opt/
EXPOSE 8888
