#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from NeuroPy import *
from time import sleep
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
from PyQt5.QtGui import*
from Cmodel import *
import numpy as np
import pyautogui as pg
import random
import pickle5 as pickle

Mw = NeuroSkyPy("COM3", 57600)
timeCount = 0
label = 0
isconnected = 0
state = np.array(['RED', 'GREEN', 'NEUTRAL'])

def get_data():
    try:
        with open("data.pickle", 'rb') as f: 
            return pickle.load(f)
    except FileNotFoundError:
        data = np.zeros(13, int)
            
def save_data():
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def get_model():
    with open("model.pickle", 'rb') as f: 
            return pickle.load(f)

data = get_data()
eegNet = get_model()
        
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 600, 300)
        
        self.connectAction = QAction(QIcon('icon/connect'), '연결', self)
        self.connectAction.setShortcut('Shift+C')
        self.connectAction.setStatusTip('장비와 연결을 시도합니다.')
        self.connectAction.triggered.connect(EEG_Mouse.Mw_start)
        
        self.saveAction = QAction(QIcon('icon/save'), '저장', self)
        self.saveAction.setShortcut('crtl+s')
        self.saveAction.setStatusTip('학습 데이터를 저장합니다.')
        self.saveAction.triggered.connect(save_data)
        
        self.exitAction = QAction(QIcon('icon/exit'), '나가기',self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('프로그램을 종료합니다.')
        self.exitAction.triggered.connect(qApp.quit)
        
        self.statusBar()
        
        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.addAction(self.connectAction)
        self.toolbar.addAction(self.saveAction)
        self.toolbar.addAction(self.exitAction)
       
        widget = QWidget()
        widget = EEG_Mouse()
        self.setCentralWidget(widget) # 메인 윈도우에 들어갈 위젯을 설정하고 대입한다
        
        #title, Icon #
        self.setWindowTitle('EEG Mouse')
        self.setWindowIcon(QIcon('icon/brain.png'))
        self.setFixedSize(QSize(600, 550)) #크기변경 불가
        self.center()
        self.show()
        
    def center(self): # 중앙배치 함수
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '메시지', '프로그램을 종료 하시겠습니까?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

        
class eegSignal(QThread):
    timeout = pyqtSignal()    # 사용자 정의 시그널  
    signal = np.zeros(12)
    signal = signal.astype(int)

    def __init__(self):
        super().__init__()

    def run(self):
        while True: # MindWaveSignal에서 작동되는 스레드로부터 신호를 받아온다
            if isconnected == 1:
                self.timeout.emit()     # 방출
                eegSignal.signal[0] = Mw.attention 
                eegSignal.signal[1] = Mw.meditation
                eegSignal.signal[2] = Mw.delta/40000
                eegSignal.signal[3] = Mw.theta/5000
                eegSignal.signal[4] = Mw.lowAlpha/500
                eegSignal.signal[5] = Mw.highAlpha/500
                eegSignal.signal[6] = Mw.lowBeta/500
                eegSignal.signal[7] = Mw.highBeta/500
                eegSignal.signal[8] = Mw.lowGamma/500
                eegSignal.signal[9] = Mw.midGamma/500
                eegSignal.signal[10] = Mw.poorSignal
            self.sleep(1)

class EEG_Mouse(QWidget):
    mode = 0
    mouseMode = 0
    label1_Action = ''
    label2_Action = ''
    label3_Action = ''
    #keyboardMode = 0
    
    def __init__(self):
        super().__init__()
        self.initUI()
            
        self.eeg = eegSignal()
        self.eeg.start()
        self.eeg.timeout.connect(self.signal_input)

    def initUI(self):     
        # Tab Layout #
        tabs = QTabWidget()
        tabs.addTab(self.tab_mouse(), 'Mouse')
        #tabs.addTab(self.tab_keyboard(), 'Keyboard')
        tabs.addTab(self.tab_signal(), 'Signal')
        
        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        self.setLayout(vbox)
        
    def Mw_start(self): # 장비 연결 시작
        global isconnected
        if isconnected == 0:
            Mw.start()
            isconnected = 1
            #self.icon_mouse.setPixmap(QPixmap("icon/mouse_c.jpg"))
        elif isconnected == 1:
            Mw.stop()
            isconnected = 0
            #self.icon_mouse.setPixmap(QPixmap("icon/mouse_dc.jpg"))
        
    def tab_mouse(self):
        # Mouse 이미지 #
        self.icon_mouse = QLabel()
        self.icon_mouse.setPixmap(QPixmap("icon/mouse_dc.jpg"))
        self.icon_mouse.setScaledContents(False)
        
        vbox1 =QVBoxLayout()
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.presetModeGroup())

        hbox = QHBoxLayout()
        hbox.addWidget(self.customModeGroup())
        hbox.addWidget(self.icon_mouse)
        
        ################### Frame #####################
        top = QFrame()
        top.setFrameShape(QFrame.Box)
        top.setLayout(vbox)
        
        bottom = QFrame()
        bottom.setFrameShape(QFrame.WinPanel)
        bottom.setFrameShadow(QFrame.Sunken)
        bottom.setLayout(hbox)
        ###############################################
        
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        
        vbox1.addWidget(splitter)
        
        tab = QWidget()
        tab.setLayout(vbox1)
        return tab
    
    def presetModeGroup(self): # 마우스 모드 관리
        groupbox = QGroupBox('프리셋 사용')
        
        self.rbtn1 = QRadioButton('클릭모드', self)
        self.rbtn1.clicked.connect(self.modeChange)
        self.rbtn2 = QRadioButton('스크롤 모드', self)
        self.rbtn2.clicked.connect(self.modeChange)
        self.rbtn3 = QRadioButton('커스텀 모드', self)
        self.rbtn3.clicked.connect(self.modeChange)
        self.rbtn1.setChecked(True)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.rbtn1)
        hbox.addWidget(self.rbtn2)
        hbox.addWidget(self.rbtn3)
        groupbox.setLayout(hbox)
        
        return groupbox
    
    def modeChange(self):
        if EEG_Mouse.mode == 0:
            if self.rbtn1.isChecked():
                EEG_Mouse.mouseMode = 0
                self.gb.setEnabled(False)
            elif self.rbtn2.isChecked():
                EEG_Mouse.mouseMode = 1
                self.gb.setEnabled(False)
            elif self.rbtn3.isChecked():
                EEG_Mouse.mouseMode = 2
                self.gb.setEnabled(True)
            print('마우스 모드: ', EEG_Mouse.mouseMode)   
        
    def customModeGroup(self):
        self.gb = QGroupBox('커스텀 설정')  
        self.gb.setEnabled(False)
        
        cb1 = QComboBox(self) # 왼쪽 마우스
        cb1.addItem('클릭')
        cb1.addItem('더블클릭')
        cb1.addItem('드래그')
        cb1.addItem('동작 없음')
        cb1.activated[str].connect(self.set_cb1)
        
        cb2 = QComboBox(self) # 휠 동작
        cb2.addItem('클릭')
        cb2.addItem('휠 위로')
        cb2.addItem('휠 아래로')
        cb2.addItem('동작 없음')
        cb2.activated[str].connect(self.set_cb2)
        
        cb3 = QComboBox(self) # 측면 버튼
        cb3.addItem('버튼 업')
        cb3.addItem('버튼 다운')
        cb3.addItem('동작 없음')
        cb3.activated[str].connect(self.set_cb3)
    
        grid = QGridLayout()
        
        grid.addWidget(QLabel('왼쪽 버튼'), 0,0)
        grid.addWidget(QLabel('휠'), 2,0)
        grid.addWidget(QLabel('측면 버튼'), 4,0)
        grid.addWidget(cb1, 1,0)
        grid.addWidget(cb2, 3,0)
        grid.addWidget(cb3, 5,0)
        
        self.gb.setLayout(grid)
        
        return self.gb
    
    def set_cb1(self, text):
        if text == '클릭':
            self.label1_Action = '클릭'
        elif text == '더블클릭':
            self.label1_Action = '더블클릭'
        elif text == '드래그':
            self.label1_Action = '드래그'
        elif text == '동작 없음':
            self.label1_Action = '동작 없음'
    
    def set_cb2(self, text):
        if text == '휠 클릭':
            self.label2_Action = '휠 클릭'
        elif text == '휠 위로':
            self.label2_Action = '휠 위로'
        elif text == '휠 아래로':
            self.label2_Action = '휠 아래로'
        elif text == '동작 없음':
            self.label2_Action = '동작 없음'
    
    def set_cb3(self, text):
        if text == '버튼 업':
            self.label3_Action = '버튼 업'
        elif text == '버튼 다운':
            self.label3_Action = '버튼 다운'
        elif text == '동작 없음':
            self.label3_Action = '동작 없음'
        
    def tab_keyboard(self):
        check1 = QCheckBox('체크버튼1', self)
        check2 = QCheckBox('체크버튼2', self)
        check3 = QCheckBox('체크버튼3', self)
        
        vbox = QVBoxLayout()
        vbox.addWidget(check1)
        vbox.addWidget(check2)
        vbox.addWidget(check3)
        
        tab = QWidget()
        tab.setLayout(vbox)
        return tab
    
    def signal_input(self):
        global timeCount 
        global state
        global label
        global data
        global isconnected
        s = eegSignal.signal
        s_input = np.argmax(softmax(eegNet.predict(np.array([[s[:10]]]))))
        
        print(s_input)
        
        if s[10] <= 10:
            self.icon_mouse.setPixmap(QPixmap("icon/mouse_c.jpg"))
        elif s[10] >= 10:
            self.icon_mouse.setPixmap(QPixmap("icon/mouse_dc.jpg"))
        
        if (self.tcb.isChecked() == True) and (isconnected == 1):
            r_num = random.randint(1, 10)
            if timeCount == 0:
                timeCount = random.randrange(5, 8)
                label = random.choice(state)
                if label == 'RED':
                    self.recallimage.setPixmap(QPixmap("icon/recallimage/red/red" + str(r_num)))
                elif label == 'GREEN':
                    self.recallimage.setPixmap(QPixmap("icon/recallimage/green/green" + str(r_num)))
                elif label == 'NEUTRAL':
                    self.recallimage.setPixmap(QPixmap("icon/recallimage/neutral/neutral" + str(1)))
            if s[10] <= 1:
                s = s[:10]
                if label == 'RED':
                    s = np.append(s, [1, 0 ,0])
                elif label == 'GREEN':
                    s = np.append(s, [0, 1, 0])
                elif label == 'NEUTRAL':
                    s = np.append(s, [0, 0, 1])
                data = np.vstack((data, s))
            timeCount = timeCount-1     
        elif (self.tcb.isChecked() == False) or (isconnected == 0):
            self.recallimage.setPixmap(QPixmap("icon/waiting.png"))
            self.tableWidget.setItem(11,1, QTableWidgetItem(""))
            if timeCount != 0:
                timeCount = 0
                      
        self.tableWidget.setItem(0,1, QTableWidgetItem(str(s[0])))
        self.tableWidget.setItem(1,1, QTableWidgetItem(str(s[1])))
        self.tableWidget.setItem(2,1, QTableWidgetItem(str(s[2])))
        self.tableWidget.setItem(3,1, QTableWidgetItem(str(s[3])))
        self.tableWidget.setItem(4,1, QTableWidgetItem(str(s[4])))
        self.tableWidget.setItem(5,1, QTableWidgetItem(str(s[5])))
        self.tableWidget.setItem(6,1, QTableWidgetItem(str(s[6])))
        self.tableWidget.setItem(7,1, QTableWidgetItem(str(s[7])))
        self.tableWidget.setItem(8,1, QTableWidgetItem(str(s[8])))
        self.tableWidget.setItem(9,1, QTableWidgetItem(str(s[9])))
        self.tableWidget.setItem(10,1, QTableWidgetItem(str(s[10])))
        self.tableWidget.setItem(11,1, QTableWidgetItem(label))
        
        if EEG_Mouse.mouseMode == 0: #클릭 모드
            if s_input == 0:
                pg.doubleClick()
            elif input == 1:
                pg.click(button='right')
        elif EEG_Mouse.mouseMode == 1:#스크롤 모드
            if s_input == 0:
                pg.scroll(100)
            elif s_input == 1:
                pg.scroll(-100)
        elif EEG_Mouse.mouseMode == 2: #커스텀 모드
            #label1_Action
            if s_input == 0:
                if self.label1_Action == '클릭':
                    pg.click(button='left')
                elif self.label1_Action == '더블클릭':
                    pg.doubleClick()
                elif self.label1_Action == '드래그':
                    pg.dragRel(-100, -100, duration = 0.5)
            #label2_Action
            if s_input == 1:
                if self.label2_Action == '휠 클릭':
                    pg.middleClick()
                elif self.label2_Action == '휠 위로':
                    pg.scroll(100)
                elif self.label2_Action == '휠 아래로':
                    pg.scroll(-100)
            #label3_Action
            if s_input == 2:
                if self.label3_Action == '버튼 업':
                    pg.mouseDown()
                elif self.label3_Action == '버튼 다운':
                    pg.mouseUp()
        
        
    def tab_signal(self):
        self.tcb = QCheckBox('학습모드', self)
        self.recallimage = QLabel()
        self.recallimage.setPixmap(QPixmap("icon/waiting.png"))
        self.recallimage.setScaledContents(True)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setFixedSize(230, 410)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(12)
        
        labels = ["항목", "값"]
        self.tableWidget.setHorizontalHeaderLabels(labels)
        #self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents) #가변길이

        self.tableWidget.setItem(0,0, QTableWidgetItem("attention"))
        self.tableWidget.setItem(1,0, QTableWidgetItem("meditation"))
        self.tableWidget.setItem(2,0, QTableWidgetItem("delta"))
        self.tableWidget.setItem(3,0, QTableWidgetItem("theta"))
        self.tableWidget.setItem(4,0, QTableWidgetItem("lowAlpha"))
        self.tableWidget.setItem(5,0, QTableWidgetItem("highAlpha"))
        self.tableWidget.setItem(6,0, QTableWidgetItem("lowBeta"))
        self.tableWidget.setItem(7,0, QTableWidgetItem("highBeta"))
        self.tableWidget.setItem(8,0, QTableWidgetItem("lowGamma"))
        self.tableWidget.setItem(9,0, QTableWidgetItem("midGamma"))
        self.tableWidget.setItem(10,0, QTableWidgetItem("poorSignal"))
        self.tableWidget.setItem(11,0, QTableWidgetItem("label"))
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers) #수정 불가능
        
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox.addWidget(self.tableWidget)
        hbox.addWidget(self.recallimage)
        vbox.addWidget(self.tcb)
        vbox.addLayout(hbox)
        
        tab = QWidget()
        tab.setLayout(vbox)
        
        return tab 
                      
    def keyboardINPUT(input):
        pg.press(input)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
    Mw.stop()


# In[ ]:




