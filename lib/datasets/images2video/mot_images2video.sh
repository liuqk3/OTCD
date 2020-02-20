#!/usr/bin/env bash

path=/data0/liuqk/MOTChallenge/

# change the following two var to the corresponding subset and stage.

dataset=MOT17
stage=test

for seq_name in $(ls ${path}/${dataset}/${stage})
do
    seq_path=${path}/${dataset}/${stage}/${seq_name}
    frame_path=${seq_path}/img1/%6d.jpg
    output_path=${seq_path}/${seq_name}.mp4

    echo frame path: ${frame_path}
    echo output path: ${output_path}

    ffmpeg -i ${frame_path} -c:v mpeg4 -f rawvideo ${output_path}
    #ffmpeg -f image2 -i ${frame_path} -q:v 1 -c:v mpeg4 -f rawvideo ${output_path}
done