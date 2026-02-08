#!/bin/bash

rsync -avzP ./test_images/ jetson@192.168.0.103:/home/jetson/Documents/dev/resnet
