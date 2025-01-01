import json  # JSON 파일을 처리하는 라이브러리
import os  # 파일 및 디렉토리 관련 작업을 위한 라이브러리
import re
import cv2  # OpenCV를 사용하여 이미지를 처리
import matplotlib.pyplot as plt  # 이미지 시각화


# 경로 설정
path_to_labels_file = "C:/Users/USER/Desktop/24_김성남/git/invoice/content_combine/annotation_combine"  # 라벨 파일 경로

# 폴더 내 파일 목록 가져오기
file_list = os.listdir(path_to_labels_file)

file_list = file_list[:]

# JSON 파일만 필터링 (확장자가 .json인 파일만)
json_files = [file for file in file_list if file.endswith('.json')]


# 우선순위 정의
priority_order = [
    "corrected_CAM",
    "corrected_KakaoTalk_",
    "coupang_",
    "daehan_",
    "epost_",
    "hanjin_",
    "lotte_",
    "rozen_"
]

# 파일 이름 정렬 함수
def custom_sort_key(file_name):
    # 정규식으로 파일 이름에서 종류와 숫자를 추출 '+' : 문자 반복,  '\s': 공백문자 , \d+: 하나 이상의 숫자가 반복됨 
    match = re.match(r"([a-zA-Z_]+)(\d+)", file_name)  # "corrected_CAM 20230916074132_1_warped.json"에 맞춤
    if match:
        file_type, number = match.groups()
        number = int(number)  # 숫자를 정수로 변환
        # 종류 우선순위와 숫자를 기준으로 정렬
        priority_index = priority_order.index(file_type) if file_type in priority_order else len(priority_order)
        return (priority_index, number)
    return (len(priority_order), 0)  # 매칭 실패 시 맨 뒤로


# 정렬된 파일 리스트
sorted_file_list = sorted(file_list, key=custom_sort_key)

len(sorted_file_list)

a=sorted_file_list[:444]
b=sorted_file_list[444:]

sorted_file_list=b+a


# 모든 JSON 데이터를 저장할 리스트
all_labels = []

# 각 JSON 파일 읽어서 처리
for json_file in sorted_file_list:
    file_path = os.path.join(path_to_labels_file, json_file)  # 파일의 전체 경로 생성
    
    # JSON 파일 열기 및 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)  # JSON 파일 로드
        all_labels.append(labels)  # 모든 데이터를 리스트에 추가
        
        
# 훈련 데이터와 테스트 데이터를 나누는 비율
train_split = 0.8  # 훈련 데이터 비율
split_idx = int(len(sorted_file_list) * train_split)  # 훈련/테스트 데이터 분할 인덱스 계산

# 폴더가 존재하지 않으면 새로 생성
folder_path = "CRAFT_data/"  # 데이터 저장 경로
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # 폴더 생성

# 훈련 폴더 생성
train_folder_path = os.path.join(folder_path, "ch4_training_localization_transcription_gt")  # 훈련 데이터 폴더 경로
if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)  # 폴더 생성

# 테스트 폴더 생성
test_folder_path = os.path.join(folder_path, "ch4_test_localization_transcription_gt")  # 테스트 데이터 폴더 경로
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)  # 폴더 생성

# 확장자 제외한 파일 이름 추출
image_names = [name.split('.')[0] for name in sorted_file_list]

train_files = sorted_file_list[:split_idx]
test_files = sorted_file_list[split_idx:]



# 레이블 저장 함수
def save_labels_to_file(file_name, labels, output_folder):
    image_name = file_name.split('.')[0]
    output_file = os.path.join(output_folder, f'gt_{image_name}.txt')
    
    for obj in labels.get("objects", []):
        if isinstance(obj, dict):
            x1 = round(obj["x"])
            y1 = round(obj["y"])
            w = round(obj["width"])
            h = round(obj["height"])
            transcription = obj["transcription"]

            # 좌표 계산
            x2, y2 = x1 + w, y1
            x3, y3 = x1 + w, y1 + h
            x4, y4 = x1, y1 + h

            # 레이블 문자열 작성
            full_label_string = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcription}"

            # 파일에 저장
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(full_label_string + '\n')


# JSON 파일 처리 및 저장
for json_file in train_files:
    file_path = os.path.join(path_to_labels_file, json_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    save_labels_to_file(json_file, labels, train_folder_path)

for json_file in test_files:
    file_path = os.path.join(path_to_labels_file, json_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    save_labels_to_file(json_file, labels, test_folder_path)

print("파일 저장이 완료되었습니다.")


from PIL import Image, ImageDraw

# 이미지 열기
image = Image.open("./CRAFT_data/ch4_test_images/daehan_172.jpg")

# 텍스트 파일에서 좌표와 텍스트를 추출 (예시 텍스트)
text_data = """
78,53,328,53,328,130,78,130,운송장번호
354,34,1437,34,1437,160,354,160,6810-4962-3794
1615,55,2093,55,2093,140,1615,140,2023.10.12
2380,69,2563,69,2563,169,2380,169,1/2
2842,51,3235,51,3235,172,2842,172,재출력: 2
2881,231,3025,231,3025,349,2881,349,코스
3062,282,3151,282,3151,361,3062,361,CJ
3272,284,3492,284,3492,361,3272,361,대한통운
3151,405,3486,405,3486,475,3151,475,택배이용문의
3149,478,3483,478,3483,549,3149,549,전국어디서나
3149,558,3484,558,3484,615,3149,615,1588-1255
964,253,1164,253,1164,580,964,580,1
1236,197,2201,197,2201,662,1236,662,M81
2233,266,2701,266,2701,606,2233,606,- 1f
3387,649,3458,649,3458,698,3387,698,BY
54,799,159,799,159,1165,54,1165,받는분
42,1282,167,1282,167,1359,42,1359,보내
45,1367,170,1367,170,1451,45,1451,는분
206,702,449,702,449,811,206,811,김민*
634,701,1415,701,1415,822,634,822,010-2023-****/
1428,709,2134,709,2134,804,1428,804,010-2023-****
192,821,464,821,464,922,192,922,경기도
491,821,760,821,760,926,491,926,의왕시
791,822,1353,822,1353,923,791,923,철도박물관로
1377,835,1551,835,1551,922,1377,922,176
1568,826,2463,826,2463,947,1568,947,(한국철도기술연구원)
2483,832,2629,832,2629,934,2483,934,1동
2664,832,2914,832,2914,930,2664,930,104호
2941,830,3308,830,3308,937,2941,937,블루스톤
191,933,422,933,422,1020,191,1020,MRO
443,927,721,927,721,1025,443,1025,사무실
760,928,1077,928,1077,1036,760,1036,[월암동
1106,931,1414,931,1414,1037,1106,1037,360-1]
194,1033,629,1033,629,1268,194,1268,월암
700,1052,1343,1052,1343,1261,700,1261,360-1
1401,1042,2282,1042,2282,1281,1401,1281,기술연구
195,1270,563,1270,563,1348,195,1348,플로발코리아
1299,1275,1763,1275,1763,1339,1299,1339,02-6124-4090
1812,1295,1923,1295,1923,1377,1812,1377,수량
1949,1275,2329,1275,2329,1368,1949,1368,극소 D 1
2507,1290,2625,1290,2625,1388,2507,1388,운임
2932,1244,3010,1244,3010,1326,2932,1326,0
3071,1293,3189,1293,3189,1381,3071,1381,정산
3269,1244,3453,1244,3453,1343,3269,1343,신용
3387,1476,3433,1476,3433,1548,3387,1548,1
197,1365,557,1365,557,1449,197,1449,서울특별시
568,1369,787,1369,787,1448,568,1448,구로구
805,1370,1024,1370,1024,1447,805,1447,구로동
1037,1366,1397,1366,1397,1446,1037,1446,615-3번지
1415,1362,1766,1362,1766,1445,1415,1445,에스티엑스
1781,1368,1994,1368,1994,1449,1781,1449,더블유
2013,1368,2152,1368,2152,1449,2013,1449,타워
2169,1368,2385,1368,2385,1442,2169,1442,814호
65,1446,1409,1446,1409,1552,65,1552,이현민/송승호/김기현/고상필/문경호
1376,2219,2194,2219,2194,2302,1376,2302,개인정보 유출우려가 있으니
1377,2296,2061,2296,2061,2376,1377,2376,운송장은 폐기바랍니다.
130,2556,1003,2556,1003,2759,130,2759,뉴삼동-F14
1221,2629,1326,2629,1326,2714,1221,2714,운임
1762,2624,1890,2624,1890,2709,1762,2709,정산
2609,2670,3090,2670,3090,2741,2609,2741,681049623794
1869,2388,2160,2388,2160,2476,1869,2476,총수량:1
153,1377,3221,1377,3221,2509,153,2509,1.김민중
"""  # 여기서 텍스트 데이터를 사용자가 주어진 것처럼 넣어주세요.

# 그리기 객체 생성
draw = ImageDraw.Draw(image)

# 텍스트 데이터를 처리
for line in text_data.strip().split("\n"):
    parts = line.split(",")  # ',' 기준으로 분리
    coordinates = [(int(parts[i]), int(parts[i + 1])) for i in range(0, 8, 2)]  # (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    text = parts[8]  # 텍스트는 마지막 부분

    # 바운딩박스를 그리기
    draw.polygon(coordinates, outline="red", width=3)



# 이미지 보기
image.show()







