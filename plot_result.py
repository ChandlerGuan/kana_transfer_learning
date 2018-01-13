# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:19:09 2018

@author: chandler
"""
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('loss')
    return ax1, ax2


history = np.load('result12.npz')
# Create some mock data
t = np.arange(1,101,1)
s1 = history['acc2']
s2 = history['loss2']

# Create axes
fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, t, s1, s2, 'r', 'b')


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None
color_y_axis(ax1, 'r')
color_y_axis(ax2, 'b')
plt.savefig('stage2.png',transparent=True)

#history = np.load('result12.npz')
#
#plt.plot(history['acc2'])
#plt.plot(history['loss2'])
#plt.title('model accuracy')
#plt.ylabel('accuracy/loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='lower right')
#plt.savefig('acc.png',transparent=True);
#
#plt.show()
# "Loss"
#plt.plot(history['loss'])
#plt.plot(history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.savefig('loss.png',transparent=True)