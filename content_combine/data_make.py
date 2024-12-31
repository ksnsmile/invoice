import json  # JSON 파일을 처리하는 라이브러리
import os  # 파일 및 디렉토리 관련 작업을 위한 라이브러리

# 경로 설정
path_to_labels_file = "C:/Users/USER/Desktop/24_김성남/git/invoice/content_combine/annotation_combine"  # 라벨 파일 경로

# 폴더 내 파일 목록 가져오기
file_list = os.listdir(path_to_labels_file)

file_list = file_list[1:]

# JSON 파일만 필터링 (확장자가 .json인 파일만)
json_files = [file for file in file_list if file.endswith('.json')]

# 모든 JSON 데이터를 저장할 리스트
all_labels = []

# 각 JSON 파일 읽어서 처리
for json_file in json_files:
    file_path = os.path.join(path_to_labels_file, json_file)  # 파일의 전체 경로 생성
    
    # JSON 파일 열기 및 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)  # JSON 파일 로드
        all_labels.append(labels)  # 모든 데이터를 리스트에 추가
        
        
# 훈련 데이터와 테스트 데이터를 나누는 비율
train_split = 0.8  # 훈련 데이터 비율
split_idx = int(len(json_files) * train_split)  # 훈련/테스트 데이터 분할 인덱스 계산

# 폴더가 존재하지 않으면 새로 생성
folder_path = "CRAFT_data/"  # 데이터 저장 경로
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # 폴더 생성

# 훈련 폴더 생성
full_folder_path = os.path.join(folder_path, "ch4_training_localization_transcription_gt")  # 훈련 데이터 폴더 경로
if not os.path.exists(full_folder_path):
    os.makedirs(full_folder_path)  # 폴더 생성

# 테스트 폴더 생성
test_folder_path = os.path.join(folder_path, "ch4_test_localization_transcription_gt")  # 테스트 데이터 폴더 경로
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)  # 폴더 생성

# 확장자 제외한 파일 이름 추출
image_names = [name.split('.')[0] for name in json_files]

# labels 항목을 하나씩 확인
for idx, obj in enumerate(labels["objects"]):
    # obj가 dict 형태인지 확인
    if isinstance(obj, dict):  # obj가 dict일 때만 진행
        x1 = obj["x"]  # x 좌표
        y1 = obj["y"]  # y 좌표
        w = obj["width"]  # 너비
        h = obj["height"]  # 높이
        transcription = obj["transcription"]  # 텍스트

        # 레이블 정보 파일 저장
        with open(os.path.join(full_folder_path, f'gt_{image_name}.txt'), 'a', encoding='utf-8') as f:
            x2 = x1 + w  # x2 좌표
            y2 = y1  # y2 좌표
            x3 = x1 + w  # x3 좌표
            y3 = y1 + h  # y3 좌표
            x4 = x1  # x4 좌표
            y4 = y1 + h  # y4 좌표

            # 레이블 문자열 작성
            full_label_string = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcription}"
            f.write(full_label_string + '\n')  # 파일에 저장







def load_dataset(train_split=0.8, folder_path="CRAFT_data/"):
    # 첫 번째 훈련
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print("폴더가 이미 존재하므로 덮어쓰지 않으므로 반환합니다. 다시 만들려면 폴더를 제거하세요.")
        return  # TODO: label.json 파일 

    path_to_labels_file = "C:/Users/USER/Desktop/24_김성남/git/invoice/content_combine/annotation_combine/corrected_CAM 20230916074132_1_warped.json"
    with open(path_to_labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    full_folder_path = os.path.join(folder_path, "ch4_training_localization_transcription_gt")
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)

    split_idx = int(len(labels["objects"]) * train_split)

    for idx, image_info in enumerate(labels["objects"]):
        if idx == split_idx:
            full_folder_path = os.path.join(folder_path, "ch4_test_localization_transcription_gt")
            if not os.path.exists(full_folder_path):
                os.makedirs(full_folder_path)

        image_name = labels["file_name"].split('/')[-1].split('.')[0]

        # 경로 형식에 맞게 파일을 열고 처리
        with open(os.path.join(full_folder_path, f'gt_{image_name}.txt'), 'w', encoding='utf-8') as f:
            for obj in image_info:
                # 객체 정보에서 좌표를 가져옵니다
                x1 = obj["x"]
                y1 = obj["y"]
                w = obj["width"]
                h = obj["height"]
                
                # 바운딩 박스의 다른 3점 좌표 계산
                x2 = x1 + w
                y2 = y1
                x3 = x1 + w
                y3 = y1 + h
                x4 = x1
                y4 = y1 + h

                # 레이블과 좌표 정보를 텍스트로 작성
                full_label_string = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{obj['transcription']}"
                f.write(full_label_string + '\n')

load_dataset()
