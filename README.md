프로젝트 제목
"EEG Mouse"​

목표​
뇌전도(EEG) 신호를 머신러닝 모델을 사용해 실시간으로 분류하고, 이를 마우스 입력 신호로 변환하여 사용자 인터페이스를 제어하는 장치 개발.​

주요기능​
EEG 신호 수집 및 전처리​
신호 분류를 위한 머신러닝 모델 설계 및 학습​
분류된 신호를 마우스 입력 신호로 매핑 및 변환​

사용 기술 스택​
하드웨어: NeuroSky MindWave, 마우스​
프로그래밍 언어: Python​
개발 환경: Window, Jupyter Notebook​​
GUI 라이브러리: PyQt

파일 설명
1. Cmodel.py - DNN 모델을 구성하고 학습을 진행
2. EEG_Project.py - 소프트웨어의 GUI와 멀티스레드 구조를 구현
3. MindWaveSignal.py - EEG 장비로부터 데이터화된 신호를 수신하기 위한 모듈
4. data.pickle - 학습 데이터를 저장하고 읽어오기 위한 .pickle 파일
5. model.pickle - 학습 데이터를 바탕으로 학습된 모델의 정보를 담은 .pickle 파일
6. icon - GUI 구성에 사용된 이미지 데이터 파일
7. pyautogui - 마우스 조작을 위한 파이썬 라이브러리

   
