# Open Vocabulary Object Retrieval
## Data Preparation
To prepare the Hong Kong airport video, first download this [video](https://www.youtube.com/watch?v=VRmh2gGeBiE) with `data_provider.py`: `python data_provider.py`; Then, devide it into three minute chunks with ffmpeg: 
```
ffmpeg -i hong_kong_airport.mp4 -c copy -map 0 -segment_time 00:03:00 -f segment -reset_timestamps 1 hong_kong_airport%01d.mp4
```
We have chosen the part between 8:00 to 9:40 as our test data, you can either generate it with ffmpeg, or download
from [google drive link](https://drive.google.com/file/d/1NW670p5VUBKKNl2oZfA80VCNNgit7kcY/view?usp=drive_link)

To convert a video into frame with certain fps with the following command
```
ffmpeg -i hong_kong_airport_3.mp4 -vf fps=5 img%04d.jpg
```

