#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
#from MindWaveSignal import *
from time import sleep
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
from PyQt5.QtGui import*
# from Cmodel import *
import numpy as np
import pyautogui as pg


# In[ ]:


Mw = MindWave("COM3", 57600)


# In[2]:


class eegSignal(QThread):
    timeout = pyqtSignal(int)    # 사용자 정의 시그널

    def __init__(self):
        super().__init__()
        self.attention = 0 # 변수 초기화
        self.meditation = 0
        self.delta = 0
        self.theta = 0
        self.lowAlpha = 0
        self.highAlpha = 0
        self.lowBeta = 0
        self.highBeta = 0
        self.lowGamma = 0
        self.midGamma = 0
        
    def run(self):
        while True:
            self.timeout.emit(self.attention)     # 방출
            self.attention = Mw.attention # MindWaveSignal에서 작동되는 스레드로부터 신호를 받아온다
            self.meditation = Mw.meditation
            self.delta = Mw.delta
            self.theta = Mw.theta
            self.lowAlpha = Mw.lowAlpha
            self.highAlpha = Mw.highAlpha
            self.lowBeta = Mw.lowBeta
            self.highBeta = Mw.highBeta
            self.lowGamma = Mw.lowGamma
            self.midGamma = Mw.midGamma
            
            self.sleep(1)            


class EEG_Mouse(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        ######### tab ###############
        tab1 = QWidget()
        tab2 = QWidget()

        tabs = QTabWidget()
        tabs.addTab(tab1, 'Mouse')
        tabs.addTab(tab2, 'KeyBoard')
        #############################
        

        ######### Mouse 이미지 ##########
        mouseIMG = QPixmap('mouse.png')
        lbl_img.setPixmap(mouseIMG)
        lbl_size = QLabel('Width: '+str(mouse.width())+', Height: '+str(mouse.height()))
        lbl_size.setAlignment(Qt.AlignCenter)    
        #################################
        
        
          
        
        ######## set Layout #############
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        
        hbox.addLayout(vbox)
        hbox.addWidget(mouseIMG)

        self.setLayout(hbox)
        ################################
        
        
        
        
        ######### title, Icon ################
        self.setWindowTitle('EEG_Mouse')
        self.setWindowIcon(QIcon('brain.png'))
        ######################################
        
        self.resize(500, 500)
        self.center()
        self.show()
    
    def center(self): # 중앙배치 함수
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def mouseINPUT(input)
        if input == 0:
            pg.click(button='left')
            return 0
        elif input == 1:
            pg.click(button='right')
            return 1
        elif input == 3:
            return 3
        
        
    def keyboardINPUT(input)
        pg.press(input)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EEG_Mouse()
    sys.exit(app.exec_())


# In[ ]:





# In[ ]:




