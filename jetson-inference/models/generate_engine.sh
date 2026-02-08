#!/bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=mb1-ssd.onnx --saveEngine=mb1-ssd.engine --fp16 --explicitBatch --workspace=1028 --verbose
