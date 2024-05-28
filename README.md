# hyunseo_lidar_detection
LiDAR 데이터를 사용하여 사람을 인식하는 ROS2 패키지입니다. 
이 패키지는 PointNet++ 모델을 사용하여 사람을 실시간으로 인식합니다.

## 설치 방법

### 1. Ubuntu 22.04, ROS2 Humble 설치

### 2. SemanticKITTI Data 중 00 Data를 data 폴더 안에 넣기

### 3. requirements.txt 설치
pip install -r requirements.txt

### 4. 데이터 전처리 스크립트 실행
python3 scripts/process_data.py

### 5. 모델 학습
python3 models/train.py

### 6. 실행
ros2 launch hyunseo_lidar_detection velodyne_launch.py