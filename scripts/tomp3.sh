#!/bin/bash
i=0
for filename in ../midi/*.mid; do
    echo $filename
    timidity "$filename" ".mid" -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k "../mp3/$i.mp3"
    ((i++))
done
