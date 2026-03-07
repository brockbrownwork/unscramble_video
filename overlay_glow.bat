@echo off
REM Overlay cab_ride_trimmed_glow.mp4 on top of cab_ride_trimmed.mkv
REM with black chromakeyed out of the glow layer.

ffmpeg -i cab_ride_trimmed.mkv -i cab_ride_trimmed_glow.mp4 ^
  -filter_complex "[1:v]colorkey=black:0.1:0.2[glow];[0:v][glow]overlay=0:0:shortest=1" ^
  -c:v libx264 -crf 18 -preset medium ^
  -c:a copy ^
  -y cab_ride_trimmed_combined.mp4

pause
