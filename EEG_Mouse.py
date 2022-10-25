#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from NeuroPy import NeuroSkyPy
from time import sleep
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
import datetime


# In[ ]:


neuropy = NeuroSkyPy.NeuroSkyPy("COM3", 57600)


# In[ ]:


neuropy.start()


# In[ ]:


neuropy.stop()


# In[ ]:


import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from NeuroPy import NeuroSkyPy
import numpy as np

neuropy = NeuroSkyPy.NeuroSkyPy("COM3", 57600)

class eegSignal(QThread):
    timeout = pyqtSignal(int, int)    # 사용자 정의 시그널

    def __init__(self):
        super().__init__()
        self.attention = 0
        self.meditation = 0
        
    def run(self):
        while True:
            self.timeout.emit(self.attention, self.meditation)     # 방출
            self.attention = neuropy.attention
            self.meditation = neuropy.meditation
            self.sleep(1)

        
class MyWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        neuropy.start()
        
        self.signal = eegSignal()
        self.signal.start()
        self.signal.timeout.connect(self.printValue())   # 시그널 슬롯 등록

        self.edit = QLineEdit(self)
        self.edit.move(10, 10)

    @pyqtSlot(int)
    def printValue(self, attention, meditation):
        print(attention)
        print(meditation)


app = QApplication(sys.argv)
mywindow = MyWindow()
mywindow.show()
app.exec_()
neuropy.stop()


# In[ ]:


import numpy as np


# In[ ]:


a = np.array([1, 2])


# In[ ]:


print(a[0])


# In[ ]:


def sumab(array):
    return array[0] + array[1]

a = np.array([1, 2])
print(sumab(a))


# In[ ]:




