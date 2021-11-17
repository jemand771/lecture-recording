FROM bandi13/gui-docker:1.01

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:obsproject/obs-studio && \
    apt-get update && \
    apt-get install -y --no-install-recommends obs-studio && \
    apt-get clean -y

RUN echo "?package(bash):needs=\"X11\" section=\"DockerCustom\" title=\"OBS Screencast\" command=\"obs\"" >> /usr/share/menu/custom-docker && update-menus
