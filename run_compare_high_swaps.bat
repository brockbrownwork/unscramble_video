@echo off
echo === Running compare_metrics with 10000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 10000 -o compare_10000.png

echo === Running compare_metrics with 20000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 20000 -o compare_20000.png

echo === Running compare_metrics with 50000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 50000 -o compare_50000.png

echo === Done! Output files: compare_10000.png, compare_20000.png, compare_50000.png ===
pause
