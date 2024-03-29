#!/bin/sh

# for model in "ResNetRS50" "ResNetRS101" "ResNetRS152" "ResNetRS200" "ResNetRS270" "ResNetRS350" "ResNetRS420" "InceptionV3" "DenseNet121" "DenseNet169" "DenseNet201" "MobileNetV2"; do
for model in "ResNetRS50" "ResNetRS101" "ResNetRS152" "ResNetRS200" "ResNetRS270" "ResNetRS350" "ResNetRS420" "DenseNet121" "DenseNet169" "DenseNet201" "MobileNetV2"; do
	python main.py --model "$model"
done
