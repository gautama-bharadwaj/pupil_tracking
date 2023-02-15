from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Opnening a dialog box for selecting the csv file
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
if (filename.split("/")[-1] != "pupil_positions.csv"):
    print("Wrong file selected. The selected file must be pupil_positions.csv")
    quit()

# Reading csv file
df = pd.read_csv(filename)

initial_timestamp = df['pupil_timestamp'].iloc[0]
pupil_start = 101
pupil_end = 107
time_interval = 6

df['pupil_timestamp'] = df['pupil_timestamp'].apply(lambda x : x - initial_timestamp)
df['norm_pos_x'] = df['norm_pos_x'].round(decimals = 2)
df['norm_pos_y'] = df['norm_pos_y'].round(decimals = 2)
# Filtering out data with low confidence score and filtering out the 3D modelled data
right_eye_data = df.loc[(df['pupil_timestamp'] > pupil_start) & (df['pupil_timestamp'] < pupil_end) & (df['eye_id'] == 0) & (df['confidence'] > 0.8) & (df['method'] == '2d c++')]
left_eye_data = df.loc[(df['pupil_timestamp'] > pupil_start) & (df['pupil_timestamp'] < pupil_end) & (df['eye_id'] == 1) & (df['confidence'] > 0.8) & (df['method'] == '2d c++')]

right_eye_data['pupil_timestamp'] = right_eye_data['pupil_timestamp'].apply(lambda x : x - pupil_start)
left_eye_data['pupil_timestamp'] = left_eye_data['pupil_timestamp'].apply(lambda x : x - pupil_start)

# right_eye_data = right_eye_data.loc[right_eye_data['norm_pos_x']< 0.45]

# Calculating the variance of the normalized positions
right_norm_pos_x = right_eye_data['norm_pos_x'].var()
right_norm_pos_y = right_eye_data['norm_pos_y'].var()

left_norm_pos_x = left_eye_data['norm_pos_x'].var()
left_norm_pos_y = left_eye_data['norm_pos_y'].var()

right_min_x_var = 1
left_min_x_var = 1
comp = 1

for iter in range(0, ((pupil_end-pupil_start+1)-time_interval)):

    # Calculating variance
    interval_right_x_var = right_eye_data.loc[(right_eye_data['pupil_timestamp'] > iter) & (right_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_x'].var()
    interval_right_y_var = right_eye_data.loc[(right_eye_data['pupil_timestamp'] > iter) & (right_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_y'].var()
    interval_left_x_var = left_eye_data.loc[(left_eye_data['pupil_timestamp'] > iter) & (left_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_x'].var()
    interval_left_y_var = left_eye_data.loc[(left_eye_data['pupil_timestamp'] > iter) & (left_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_y'].var()

    # Calculating min
    interval_right_x_min = right_eye_data.loc[(right_eye_data['pupil_timestamp'] > iter) & (right_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_x'].min()
    interval_right_y_min = right_eye_data.loc[(right_eye_data['pupil_timestamp'] > iter) & (right_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_y'].min()
    interval_left_x_min = left_eye_data.loc[(left_eye_data['pupil_timestamp'] > iter) & (left_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_x'].min()
    interval_left_y_min = left_eye_data.loc[(left_eye_data['pupil_timestamp'] > iter) & (left_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_y'].min()

    # Calculating max
    interval_right_x_max = right_eye_data.loc[(right_eye_data['pupil_timestamp'] > iter) & (right_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_x'].max()
    interval_right_y_max = right_eye_data.loc[(right_eye_data['pupil_timestamp'] > iter) & (right_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_y'].max()
    interval_left_x_max = left_eye_data.loc[(left_eye_data['pupil_timestamp'] > iter) & (left_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_x'].max()
    interval_left_y_max = left_eye_data.loc[(left_eye_data['pupil_timestamp'] > iter) & (left_eye_data['pupil_timestamp'] < (iter + time_interval))]['norm_pos_y'].max()

    # Right eye
    if (interval_right_x_var <= right_min_x_var):
        # Variance
        right_min_x_var = interval_right_x_var
        right_min_y_var = interval_right_y_var

        # Min
        right_min_x_min = interval_right_x_min
        right_min_y_min = interval_right_y_min

        # Max
        right_min_x_max = interval_right_x_max
        right_min_y_max = interval_right_y_max

        # Timestamp
        right_interval = iter


    # Left eye
    if (interval_left_x_var <= left_min_x_var):
        # Variance
        left_min_x_var = interval_left_x_var
        left_min_y_var = interval_left_y_var

        # Min
        left_min_x_min = interval_left_x_min
        left_min_y_min = interval_left_y_min

        # Max
        left_min_x_max = interval_left_x_max
        left_min_y_max = interval_left_y_max

        # Timestamp
        left_interval = iter


right_eye_data.plot(x = 'pupil_timestamp', y = ['norm_pos_x', 'norm_pos_y'], linewidth=4)
plt.figure(1)
plt.title("Center ball - Right eye")
plt.xlabel("Timestamps (s)")
plt.ylabel("Normalized pixel position")
plt.ylim([0, 1])
plt.show()

left_eye_data.plot(x = 'pupil_timestamp', y = ['norm_pos_x', 'norm_pos_y'], linewidth=4)
plt.figure(1)
plt.title("Center ball - Left eye")
plt.xlabel("Timestamps (s)")
plt.ylabel("Normalized pixel position")
plt.ylim([0, 1])
plt.show()


print("Right eye norm_pos_x variance: " + str(right_min_x_var))
print("Right eye norm_pos_y variance: " + str(right_min_y_var))
print("Right eye min var timestamp: " + str(right_interval))
print("Left eye norm_pos_x variance: " + str(left_min_x_var))
print("Left eye norm_pos_y variance: " + str(left_min_y_var))
print("Left eye min var timestamp: " + str(left_interval))

print("Right eye - x min: " + str(right_min_x_min))
print("Right eye - y min: " + str(right_min_y_min))
print("Left eye - x min: " + str(left_min_x_min))
print("Left eye - y min: " + str(left_min_y_min))

print("Right eye - x max: " + str(right_min_x_max))
print("Right eye - y max: " + str(right_min_y_max))
print("Left eye - x max: " + str(left_min_x_max))
print("Left eye - y max: " + str(left_min_y_max))