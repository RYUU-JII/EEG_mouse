import serial
import _thread as thread
from datetime import datetime

class MindWave(object):
    attention = 0
    meditation = 0
    delta = 0
    theta = 0
    lowAlpha = 0
    highAlpha = 0
    lowBeta = 0
    highBeta = 0
    lowGamma = 0
    midGamma = 0

    poorSignal = 0
    
    rawValue = 0
    blinkStrength = 0
    
    port = None
    baudRate = None
    srl = None
    threadRun = True

    def __init__(self, port, baudRate=57600):
        self.port, self.baudRate = port, baudRate
        
    
    def start(self):
        self.threadRun = True # 별도의 스레드에서 패킷 분석기 시작
        self.srl = serial.Serial(self.port, self.baudRate)

    def packetParser(self, srl):
        # 지속적으로 신호를 얻기 위해 별도의 스레드에서 실행됨
        
        while self.threadRun:
            p1 = srl.read(1).hex()  # 처음 2개 패킷 읽기
            p2 = srl.read(1).hex()
            while p1 != 'aa' or p2 != 'aa':
                p1 = p2
                p2 = srl.read(1).hex()
            else:
                # 패킷이 사용 가능한 경우
                payload=[]
                checksum=0;
                payloadLength=int(srl.read(1).hex(),16)
                for i in range(payloadLength):
                    tempPacket=srl.read(1).hex()
                    payload.append(tempPacket)
                    checksum+=int(tempPacket,16)
                checksum=~checksum&0x000000ff
                if checksum==int(srl.read(1).hex(),16):
                   i=0
                   while i<payloadLength:
                       code=payload[i]
                       if(code=='02'):#poorSignal
                           i=i+1; self.poorSignal=int(payload[i],16)
                       elif(code=='04'):#attention
                           i=i+1; self.attention=int(payload[i],16)
                       elif(code=='05'):#meditation
                           i=i+1; self.meditation=int(payload[i],16)
                       elif(code=='16'):#blink strength
                           i=i+1; self.blinkStrength=int(payload[i],16)
                       elif(code=='80'):#raw value
                           i=i+1
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; self.rawValue=val0*256+int(payload[i],16)
                           if self.rawValue>32768 :
                               self.rawValue=self.rawValue-65536
                       elif(code=='83'):#ASIC_EEG_POWER
                           i=i+1;
                           #delta:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.delta=val0*65536+val1*256+int(payload[i],16)
                           #theta:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.theta=val0*65536+val1*256+int(payload[i],16)
                           #lowAlpha:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.lowAlpha=val0*65536+val1*256+int(payload[i],16)
                           #highAlpha:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.highAlpha=val0*65536+val1*256+int(payload[i],16)
                           #lowBeta:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.lowBeta=val0*65536+val1*256+int(payload[i],16)
                           #highBeta:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.highBeta=val0*65536+val1*256+int(payload[i],16)
                           #lowGamma:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.lowGamma=val0*65536+val1*256+int(payload[i],16)
                           #midGamma:
                           i=i+1; val0=int(payload[i],16)
                           i=i+1; val1=int(payload[i],16)
                           i=i+1; self.midGamma=val0*65536+val1*256+int(payload[i],16)
                       else:
                           pass
                       i=i+1
   
        self.srl.close() # 스레드 종료
        thread.exit() # 스레드가 닫히지 않을 경우 예외처리

    def stop(self):
        self.threadRun = False # 스레드 중지 후 COM 포트 해제(장비연결 종료)

    ### 값을 return하는 함수들 ###
    
    #attention
    def attention(self):
        return self.attention

    #meditation
    def meditation(self):
        return self.meditation

    #rawValue
    def rawValue(self):
        return self.rawValue

    #delta
    def delta(self):
        return self.delta

    #theta
    def theta(self):
        return self.theta

    #lowAlpha
    def lowAlpha(self):
        return self.lowAlpha

    #highAlpha
    def highAlpha(self):
        return self.highAlpha
    
    #lowBeta
    def lowBeta(self):
        return self.lowBeta

    #highBeta
    def highBeta(self):
        return self.highBeta

    #lowGamma
    def lowGamma(self):
        return self.lowGamma

    #midGamma
    def midGamma(self):
        return self.midGamma
    
    #poorSignal
    def poorSignal(self):
        return self.poorSignal

    #blinkStrength
    def blinkStrength(self):
        return self.blinkStrength
