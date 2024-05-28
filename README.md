# hyunseo_lidar_detection
LiDAR 데이터를 사용하여 사람을 인식하는 ROS2 패키지입니다. 
PointNet++ 모델을 사용하여 사람을 실시간으로 인식합니다.

## 설치 방법

### 1. Ubuntu 22.04, ROS2 Humble 설치
python 3.10 버전을 권장합니다. (이후 버전의 경우 open3d와 충돌 발생)

### 2. SemanticKITTI Data 중 00 Data를 data 폴더 안에 넣기
```
├── data/
│ ├── sequences/
│ │ ├── 00/
│ │ │ ├── velodyne/ (SemanticKITTI)
│ │ │ ├── labels/ (SemanticKITTI)
│ ├── processed/
│ │ ├── person/
│ │ └── non_person/
```
위와 같은 형태로 폴더를 만들어야 합니다.

### 3. requirements.txt 설치
pointnet2의 경우 아래 링크를 통해 수동으로 빌드하여 설치해야 합니다.
(python 버전 매칭 안됨)
https://github.com/erikwijmans/Pointnet2_PyTorch
```
sudo apt-get install python3-pcl pcl-tools
pip install -r requirements.txt
```

### 4. 데이터 전처리 스크립트 실행
```
python3 scripts/process_data.py
```

### 5. 모델 학습
```
python3 models/train.py
```

### 6. 실행
```
ros2 launch hyunseo_lidar_detection velodyne_launch.py
```