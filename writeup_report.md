Behavioral Cloning
==================

This document is the writeup summary of "Project 3: Behavioral Cloning". This
contains what my network architecture lokks like and how I trained my network.

Contents
--------

Besides this writeup, this repository contains `model.py`, `model.h5`,
`model.h5.o`, `drive.py`, `video.mp4` and `video_fast.mp4`. `model.h5.o` is
the model generated from the network without dropout layers. `video_fast.mp4`
is the recorded video where the speed in `drive.py` is changed to the larger
value, 20.

Network Arechitecture
---------------------

I tried few network models, from very simple one convolutional layer model to
LeNet like model. After several trials I ended up with employing Nvidia's
network that is borrowed from their paper,
[End to End Learning for Self-Driving Cars][nvidia-e2e], since it is a battle
tested model used on the real road.

It is comprised of five convolutional layers and the three color channels are
preserved in the first layer. Then it's flattened and three fully conntected
layers folow. The output is the single value, which correponds to the steering
angle.

In addition to the original Nvidia network, I added a cropping layer and a
normalization layer. The cropping layer crops the image and extract the road
image data. The normalization layer normalizes the values of the image matrix
data to fit into the reage from `-0.5` to `0.5`.

Finally, I added the dropout layers after each convolutional layer as well. This
made the driving smooth but the higher dropout probabilities let the car go off
the road. So I set the dropout rate to `0.1`.


[nvidia-e2e]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Training Approach
-----------------

### Modular Data Separation

I had hard time on training my network with the appropriate data. Besides the
fact I was a terrible driver, if I chose the existing directory that contains
the driving data from the previous record, the new record data overwrite the
existing ones. So if I wanted to train the network with the specific driving
data that enhances the learning of the specific part of the course, the specific
curve point where the car tends to go off the course for instance, I needed to
create a new directory and save driving data there.

To leverage the maximum data benefit, I separated the data retrieval of the CSV
file from the image data retrieval. The driving data from the CSV file is much
smaller than the image data. So my computer can hold all text data retrieved
from multiple CSV files stored in the different directories. I defined the
constant list value that contains the name of the directories from which the
data is loaded and I modified this value to change the datasets for the
training. This enalbes the modular data separation based on the directory and I
could give the specific dataset to the network. This improved the agility for
the training and I could just add the new recorded data when the car indicated
non disired behaviour.

### Data Separation and Generator

In the beggining all the lines of the multiple CSV files are loaded, shuffled
and separated into the training and validation datasets. Then some data is
picked up depends on the batch size and the image data is retrieved with the
file name contaiend in the line. This image data loading lazily happnes in the
generator and used image data is garbage collected. So this is very memory
space efficient as well as it is modular. A line contained the locations of the
three image files and they are corresponded to the steering data with the
correction.

### Basic Training Data

I collected the training data following the instruciton of the project. I
prepare the two laps of the regular driving, one lap of the recovery driving,
another lap of the smooth curing driving. I added the two laps of the clean
driving data on the track two as well. They are all in the differenet
directories and recorded separately.

### More Training Data

The initial collected data was not enough. The driving was not stable and
occasionally the car went off the road or failed to turn the curves
appropriately. I introduced the augumented data for the three images taken
in the one data collection frame. They are flipped and the corresponded
steering angles are also negated. This contributed to the stability at some
level but the car still went off the road.

I ended up with adding much more data, for the counter-clockwise course driving,
the quicker recovery from the lane edge, more stable driving at the center of
the road and more smooth curving. These data improved the driving stability much
and the car rarely went off the road. The quick data collection and the training
based on them was achieved by the modular data separation described above.

The recovery data is mandatory but the stable driving data is more important to
stabilize the driving. The car should follow the stable path normally and it
should correct its course when it likely goes off the road, which happnes less
frequently than the normal case.

Summary
-------

I constructed the stable network for driving the track 1. It can drive the car
very well on the track 2 as well. At the default speed and the higher speed,
i.e., `15` in the unit of `driving.py`, the driving is stable but at the maximum
speed, i.e., `30`, the car gets unstable especially once it is off the course
since it tires to correct its course with the big angle that leads the car set
its path nearly vertical against the lanes. To fix this issue, I believe the
resolution of the steering control time frame should be increased. The current
time frame is low compared with the actual one where we drvie the car manually.
With the better framerate resolution for the driving control, finer grain
steering control can be achieved.
