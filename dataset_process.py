# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:08:17 2017

@author: chandler
"""

from PIL import Image
import os
import cv2

cvt_table = {'a':0,'i':1,'u':2,'e':3,'o':4}

def load_png():
    for filename in os.listdir('dataset'):
        img = Image.open(os.path.join(os.getcwd(),'dataset',filename))
        img = img.convert("RGBA")
        pixdata = img.load()
        
        for y in xrange(img.size[1]):
            for x in xrange(img.size[0]):
                
                if (pixdata[x,y][0]>200 and pixdata[x,y][1]>200 and pixdata[x,y][2]>200) and pixdata[x,y][3]>200:
                    pixdata[x,y]=(0,0,0,255)
                else:
                    pixdata[x,y]=(255,255,255,255)
        img.save(os.path.join(os.getcwd(),'dataset','result_'+filename),'PNG')
        
def cvt_png():
    x = [];
    y = [];
    for filename in os.listdir('dataset'):
        if (not filename.split('_')[0]=='result'):
            continue;
        img = cv2.imread(os.path.join('dataset',filename))
        img = cv2.resize(img,(28,28));
        x.append(img);
        y.append(cvt_table[os.path.splitext(filename.split('_')[1])[0]]);
        cv2.imwrite(os.path.join('dataset','final_'+filename),img)

if __name__ == "__main__":
    cvt_png();
        