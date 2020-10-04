#!/bin/sh

fileId=MjqKTJvnxRZ4w0lHlxFeqY2ix4nmILNt
fileName=tweets.7z
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

7z x tweets.7z
#rm tweets.7z