# recording-image

This folder contains configuration files for the recording container.

## Container configuration

### Quickstart

````
docker run --rm -it -p 1234:5901 -e VNC_PASSWD=123456 -v ~/obs-container:/root --shm-size=1g ghcr.io/jemand771/docker-obs
````

Navigate to http://localhost:1234 to open the web-based vnc viewer.

### Container config
When starting, make sure to adjust the:
* vnc password
* shm size
* host port
* mounted host folder

### OBS config

OBS stores its configuration files under `~/.config/obs-studio` which lies in `/root` for the root user (default).

While I could spend some time fine-tuning a default config for this container, it's easier to just configure it by hand once and mount in the config folder on every consecutive run.

## Credit
The Dockerfile is based on [bandi13](https://github.com/bandi13 )'s awesome work: [gui-docker](https://github.com/bandi13/gui-docker) (base image) and [docker-obs] (https://github.com/bandi13/docker-obs)