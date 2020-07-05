# Stereo_Project

## Requirements

only need python(>=3.5), and install OpenCV-Python

```Bash
pip install opencv-python
```

## Get started

calibrate camera use opencv API. You can find undistorted images in output
folder after running following commands

```Bash
python calibrate.py left # calibrate camera using images in left folder
python calibrate.py right # calibrate camera using images in right folder
```

calibrate camera use reimplemtented version

```Bash
python my_calibrate.py left # calibrate camera using images in left folder
python my_calibrate.py right # calibrate camera using images in right folder
```

Using images in left and right folders to do stereo calibration

```Bash
python stereo_calib.py
```

do rectification (note you must run `stereo_calib.py` before you running following command)

```Bash
python rectify.py $(img_id) # e.g. python rectify.py 01
```

stereo matching using OpenCV API

```Bash
python stereo_match.py $(left_img_dir) $(right_img_dir)
# e.g. python stereo_match.py match_input/left_rectified.png match_input/right_rectified.png
```
