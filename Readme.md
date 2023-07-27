# Open Vocabulary Object Retrieval
## Data Preparation
To prepare the Hong Kong airport video, first download this [video](https://www.youtube.com/watch?v=VRmh2gGeBiE) with `data_provider.py`: `python data_provider.py`; Then, devide it into three minute chunks with ffmpeg: 
```
ffmpeg -i hong_kong_airport.mp4 -c copy -map 0 -segment_time 00:03:00 -f segment -reset_timestamps 1 hong_kong_airport%01d.mp4
```
You can also convert a video into frame with certain fps with the following command
```
ffmpeg -i hong_kong_airport_3.mp4 -vf fps=5 img%04d.jpg
```

