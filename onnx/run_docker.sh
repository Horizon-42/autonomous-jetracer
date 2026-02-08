#!/bin/bash

sudo docker run -it --rm -v $(pwd):/export -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python vae_bridge /bin/bash
