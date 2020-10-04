#!/bin/sh

fileId=1MjqKTJvnxRZ4w0lHlxFeqY2ix4nmILNt
#fileId = 1iIJSBXy_quz5tE-HfIkKAfGxwSySwulx

fileName=tweets.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

7z x tweets.7z
#rm tweets.7z