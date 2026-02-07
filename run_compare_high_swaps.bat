@echo off
echo === Running compare_metrics with 10 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 10 -o compare_10.png

echo === Running compare_metrics with 100 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 100 -o compare_100.png

echo === Running compare_metrics with 1000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 1000 -o compare_1000.png

echo === Running compare_metrics with 10000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 10000 -o compare_10000.png

echo === Running compare_metrics with 20000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 20000 -o compare_20000.png

echo === Running compare_metrics with 50000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 50000 -o compare_50000.png

echo === Running compare_metrics with 100000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 100000 -o compare_100000.png

echo === Running compare_metrics with 1000000 swaps ===
python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 1000000 -o compare_1000000.png

echo === Done! Output files: compare_10.png, compare_100.png, compare_1000.png, compare_10000.png, compare_20000.png, compare_50000.png, compare_100000.png, compare_1000000.png ===
pause
