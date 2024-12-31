import numpy as np
import random
import easyocr

import json
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
import time
from os import listdir
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import RoIAlign
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers.models.layoutlm import LayoutLMModel, LayoutLMConfig
from transformers.modeling_outputs import TokenClassifierOutput

from transformers import AdamW
from tqdm.notebook import tqdm

import matplotlib.font_manager as fm
import os   
from torchvision.models import ResNet101_Weights
import easyocr
from sklearn.preprocessing import MultiLabelBinarizer
from shapely.geometry import box

# 시드 설정
seed = 20
np.random.seed(seed) #Numpy 난수 생성기 동일
random.seed(seed) #pyhton 난수 생성기 동일
torch.manual_seed(seed) #pyTorch 난수 생성기 동일 
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) #현재 사용중인 GPU 시드 설정
    torch.cuda.manual_seed_all(seed) #다중 GPU 환경에서 모든 GPU동일한 시드 설정
torch.backends.cudnn.deterministic = True #PyTorch가 CUDNN 사용할 때 연산결과 항상 동일 
torch.backends.cudnn.benchmark = False #CUDNN이 연산 그래프에 따라 가장 빠른 알고리즘 선택하지 않도록 설정


# ## 0. raw data => annotation 후
# ## 1. 이미지 로드 및 전처리



class krri_invoice(Dataset):
    """LayoutLM dataset with visual features."""

    def __init__(self, image_file_names, tokenizer, max_length, target_size, train=True):
        # 이미지 파일만을 선택하기 위해 확장자 필터링 추가
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # 허용할 이미지 확장자
        self.image_file_names = [
            name for name in image_file_names 
            if name.lower().endswith(valid_extensions) and ".ipynb_checkpoints" not in name
        ]
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.target_size = target_size
        self.pad_token_box = [0, 0, 0, 0]
        self.train = train

    def __len__(self):
        return len(self.image_file_names) #데이터셋의 총 샘플 수 

    def __getitem__(self, idx):

        # first, take an image
        item = self.image_file_names[idx] # 해당하는 이미지 파일 가져오기
        if self.train: #train이 false가 되는경우는?
            base_path = "content_combine/data_combine"
        else:
            base_path = "content_combine/data_test2"

        ########체크포인트 처리 필요x
        # target = base_path + '/.ipynb_checkpoints'
        # if base_path + '/' + item == target:
        #   print('체크포인트 걸림!')
        #   return

        #original_image = Image.open(base_path + '/' + item).convert("L")
        original_image = Image.open(base_path + '/' + item).convert("RGB")

        # resize to target size (to be provided to the pre-trained backbone)
        resized_image = original_image.resize((self.target_size, self.target_size))
        
        # first, read in annotations at word-level (words, bounding boxes, labels)
        # with open(base_path + '/annotations/' + item[:-4] + '.json') as f:
        
        # ex) item="image1.jpg" 이면 content_combine/annotation_combine/image1.json
        with open('content_combine/annotation_combine/' + item.split('.')[0] + '.json', encoding='utf-8') as f:
         origin = json.load(f)
        #data = origin[item.split('.')[0] ]
         data = origin

        words = []
        unnormalized_word_boxes = []
        word_labels = []

        #nrm_to_abs_width = original_image.size[0] / 100
        #nrm_to_abs_height = original_image.size[1] / 100

        for i in data['objects']:
            words.append(i['transcription'])
            word_labels.append(i['label'])
            unnormalized_word_boxes.append([int(i['x']), int(i['y']), int((i['x'] + i['width'])), int((i['y'] + i['height']))])
            
       # for annotation in data['form']:
         # get label
        #  label = annotation['label']
          # get words
       #   for annotated_word in annotation['words']:
        #      if annotated_word['text'] == '':
        #        continue
         #     words.append(annotated_word['text'])
        #      unnormalized_word_boxes.append(annotated_word['box'])
         #     word_labels.append(label)

        width, height = original_image.size
        # 0에서 1사이로 픽셀좌표값 정규화
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        #조건이 False일때 전처리가 잘 못 된 것 
        assert len(words) == len(normalized_word_boxes)
        #print(len(words))
        #print(len(normalized_word_boxes))
        #print(len(word_labels))

        # next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
        token_boxes = []
        unnormalized_token_boxes = []
        token_labels = []
        for word, unnormalized_box, box, label in zip(words, unnormalized_word_boxes, normalized_word_boxes, word_labels):
            word_tokens = self.tokenizer.tokenize(word)
            # 하나의 단어가 여러 토큰으로 나뉘더라도 모든 토큰이 동일한 바운딩 박스를 공유하도록 
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
            # label first token as B-label (beginning), label all remaining tokens as I-label (inside)
            for i in range(len(word_tokens)):
              if i == 0:
                token_labels.extend(['B-' + label])
              else:
                token_labels.extend(['I-' + label])

        # Truncation of token_boxes + token_labels
        special_tokens_count = 2
        if len(token_boxes) > self.max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
            token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]

        # add bounding boxes and labels of cls(클래스시작) + sep(구분) tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
        token_labels = [-100] + token_labels + [-100]

        encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True) #truncation은 길이를 초과할 경우 자른다.
        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = self.max_seq_length - len(input_ids)
        token_boxes += [self.pad_token_box] * padding_length
        unnormalized_token_boxes += [self.pad_token_box] * padding_length
        token_labels += [-100] * padding_length
        encoding['bbox'] = token_boxes 
        encoding['labels'] = token_labels

        assert len(encoding['input_ids']) == self.max_seq_length
        assert len(encoding['attention_mask']) == self.max_seq_length
        assert len(encoding['token_type_ids']) == self.max_seq_length
        assert len(encoding['bbox']) == self.max_seq_length
        assert len(encoding['labels']) == self.max_seq_length

        encoding['resized_image'] = ToTensor()(resized_image) #이미지를 pyTorch 텐서 형태로 변환
        # rescale and align the bounding boxes to match the resized image size (typically 224x224)
        encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, self.target_size)
                                                          for bbox in unnormalized_token_boxes]

        encoding['unnormalized_token_boxes'] = unnormalized_token_boxes

        # finally, convert everything to PyTorch tensors
        for k,v in encoding.items():
            if k == 'labels':
              label_indices = []
              # convert labels from string to indices
              for label in encoding[k]:
                if label != -100:
                  label_indices.append(label2idx[label])
                else:
                  label_indices.append(label)
              encoding[k] = label_indices
            encoding[k] = torch.as_tensor(encoding[k])

        return encoding
    
#LayoutLM과 ResNet-101을 결합하여 텍스트의 문맥, 문서의 레이아웃을 고려하여 정보처리 하는 모델
class LayoutLMForTokenClassification(nn.Module):
    def __init__(self, output_size=(3,3), #1,9216 ROlAlign의 출력 크기 
                 spatial_scale=14/1024,
                 sampling_ratio=2,
                 dropout_prob=0.1
        ):
        super().__init__()

        # LayoutLM base model + token classifier
        self.num_labels = len(label2idx)
        self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=self.num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        #self.dropout = nn.Dropout(self.layoutlm.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.layoutlm.config.hidden_size, self.num_labels)

        # backbone + roi-align + projection layer
        model = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-3])) 
        #self.backbone = nn.Sequential(*(list(model.children())[:0]))

        self.roi_align = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
        self.projection = nn.Linear(in_features=1024*3*3, out_features=self.layoutlm.config.hidden_size)


#입력 인자
    def forward(
        self,
        input_ids,
        bbox,
        attention_mask,
        token_type_ids,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        resized_images=None, # shape (N, C, H, W), with H = W = 224
        resized_and_aligned_bounding_boxes=None, # single torch tensor that also contains the batch index for every bbox at image size 224
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:torch.LongTensor of shape :obj:(batch_size, sequence_length), optional):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels -
            1]`.

        """
        return_dict = return_dict if return_dict is not None else self.layoutlm.config.use_return_dict

        # first, forward pass on LayoutLM
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # next, send resized images of shape (batch_size, 3, 224, 224) through backbone to get feature maps of images
        # shape (batch_size, 1024, 14, 14)
        feature_maps = self.backbone(resized_images)

        # next, use roi align to get feature maps of individual (resized and aligned) bounding boxes
        # shape (batch_size*seq_len, 1024, 3, 3)
        device = input_ids.device
        resized_bounding_boxes_list = []
        for i in resized_and_aligned_bounding_boxes:
          resized_bounding_boxes_list.append(i.float().to(device))

        feat_maps_bboxes = self.roi_align(input=feature_maps,
                                        # we pass in a list of tensors
                                        # We have also added -0.5 for the first two coordinates and +0.5 for the last two coordinates,
                                        # see https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
                                        rois=resized_bounding_boxes_list
                           )

        # next, reshape  + project to same dimension as LayoutLM.
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        feat_maps_bboxes = feat_maps_bboxes.view(batch_size, seq_len, -1) # Shape (batch_size, seq_len, 1024*3*3)
        projected_feat_maps_bboxes = self.projection(feat_maps_bboxes) # Shape (batch_size, seq_len, hidden_size)

        # add those to the sequence_output - shape (batch_size, seq_len, hidden_size)
        sequence_output += projected_feat_maps_bboxes

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
#loss 계산하는 ㅂ누분
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
def resize_bounding_box(bbox, original_image, target_size):
    # 원본 이미지의 가로, 세로 크기
    original_width, original_height = original_image.size

    # 이미지의 가로/세로 비율을 계산
    aspect_ratio = original_width / original_height

    # 타겟 크기의 비율에 맞게 조정
    if aspect_ratio > 1:
        # 가로가 더 길면, 타겟 크기의 가로를 기준으로 세로 크기 조정
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # 세로가 더 길면, 타겟 크기의 세로를 기준으로 가로 크기 조정
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # 스케일 팩터 계산
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    
    # 바운딩 박스 좌표에 비율을 적용하여 크기 조정
    orig_left, orig_top, orig_right, orig_bottom = bbox

    # 좌표를 새롭게 조정
    resized_left = int(np.round(orig_left * x_scale))
    resized_top = int(np.round(orig_top * y_scale))
    resized_right = int(np.round(orig_right * x_scale))
    resized_bottom = int(np.round(orig_bottom * y_scale))

    # 새로운 좌표 반환
    return [resized_left, resized_top, resized_right, resized_bottom]


    
def normalize_box(box, width, height):
     return [
         int((box[0] / width)),
         int((box[1] / height)),
         int((box[2] / width)),
         int((box[3] / height)),
     ]


def resize_and_align_bounding_box(bbox, original_image, target_size):
  x_, y_ = original_image.size

  x_scale = target_size / x_
  y_scale = target_size / y_

  origLeft, origTop, origRight, origBottom = tuple(bbox)

  x = int(np.round(origLeft * x_scale))
  y = int(np.round(origTop * y_scale))
  xmax = int(np.round(origRight * x_scale))
  ymax = int(np.round(origBottom * y_scale))

  return [x-0.5, y-0.5, xmax+0.5, ymax+0.5]


def visualize_predictions_with_labels(image, predictions, font_size=12):
    draw = ImageDraw.Draw(image)
    font_path = r'C:\Users\USER\AppData\Local\Microsoft\Windows\Fonts\NotoSansKR-Medium.ttf'
    
    # 폰트 설정
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("폰트를 불러오지 못했습니다. 기본 폰트를 사용합니다.")

    for bbox_coords, label, ocr_text, gt_text, prob in predictions:
        # 라벨에서 B- 또는 I- 접두사를 제거하여 표시
        display_label = label[2:] if label.startswith(('B-', 'I-')) else label
        
        # 바운딩 박스 색상 설정: 항상 초록색으로 설정
        box_color = "green"

        # 바운딩 박스 그리기 (좌상단과 우하단 좌표 형식)
        left, top, right, bottom = bbox_coords[0][0], bbox_coords[0][1], bbox_coords[2][0], bbox_coords[2][1]
        draw.rectangle([left, top, right, bottom], outline=box_color, width=1)

        # OCR로 인식한 텍스트와 LayoutLM으로 예측한 클래스 텍스트 삽입
        text_position = (left, max(0, top - font_size))
        display_text = f"{ocr_text} ({display_label})" if ocr_text else f"({display_label})"  # OCR 결과가 없는 경우 빈칸
        draw.text(text_position, display_text, fill=box_color, font=font)

    return image

def align_bounding_boxes(bounding_boxes):
    aligned_bounding_boxes = []
    for bbox in bounding_boxes:
        # 약간의 오프셋을 주어 좌표를 조정합니다.
        aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
        aligned_bounding_boxes.append(aligned_bbox)
    return aligned_bounding_boxes





#image
image = Image.open('content_combine/data_test2/epost_64.jpg') 
image = image.convert("RGB")

#data #annotaion2 폴더에서 가져옴
with open('content_combine/annotation_combine/epost_64.json', encoding='UTF-8') as f:
  data = json.load(f)

words = []
bounding_boxes = []
labels = []

# words, word_labels, unnormalized_boxes 변수에 어노테이션 결과를 저장
for i in data['objects']:
    words.append(i['transcription'])
    labels.append(i['label'])
    bounding_boxes.append([int(i['x']), int(i['y']), int((i['x'] + i['width'])), int((i['y'] + i['height']))])

#print("Words:", words)
#print("Bounding boxes:", bounding_boxes)
#print("Labels:", labels)

assert len(words) == len(bounding_boxes) == len(labels)
#print(len(words))

image_with_bboxes = image.copy()
draw = ImageDraw.Draw(image_with_bboxes, "RGBA") #이미지에 그림 그릴 수 있도록 설정하는 객체 생성

for bbox in bounding_boxes:
    draw.rectangle(bbox, outline='red', width=1)

image_with_bboxes

# resize image #입력하면 전부 들어가게 하는 것 맞나 확인 #일단 여기는 하나의 image(사진 형태)
target_size = 1024
resized_image = image.copy().resize((target_size, target_size))
resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bounding_boxes]
draw = ImageDraw.Draw(resized_image, "RGBA")
for bbox in resized_bounding_boxes:
    draw.rectangle(bbox, outline='red', width=1)

resized_image

#딥러닝 모델에 이미지를 입력할 수 있도록 준비 
image = ToTensor()(resized_image).unsqueeze(0) # batch size of 1 #아직까지 하나
image.shape

# pretrained=True를 제거하고, weights=ResNet101_Weights.DEFAULT로 수정
model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-3])) #resnet101의 특정 층만 선택


with torch.no_grad():
    feature_map = model(image)

feature_map = F.adaptive_avg_pool2d(feature_map, output_size=(14, 14))

#이미지의 고수준 특징을 얻어냄
print(feature_map.size())  # torch.Size([1, 1024, 14, 14])

output_size = (3,3)
spatial_scale = feature_map.shape[2]/target_size # 14/224
sampling_ratio = 2

roi_align = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)



feature_maps_bboxes = roi_align(input=feature_map,
                                # we pass in a single tensor, with each bounding box also containing the batch index (0)
                                # We also add -0.5 for the first two coordinates and +0.5 for the last two coordinates,
                                # see https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
                                rois=torch.tensor([[0] + bbox for bbox in align_bounding_boxes(resized_bounding_boxes)]).float()
                      )
print(feature_maps_bboxes.shape)

visual_embeddings = torch.flatten(feature_maps_bboxes, 1)
visual_embeddings.shape

#BERT모델에서 사용하기 위해서 768특징을 맞춤 
projection = nn.Linear(in_features=visual_embeddings.shape[-1], out_features=768)
output = projection(visual_embeddings)
print(output.shape)


# list all training image file names #data2는 jpg
image_files_train = [f for f in listdir('content_combine/data_combine') 
                     if f.endswith(('.jpg', '.jpeg', '.png')) and ".ipynb_checkpoints" not in f]

# list all test image file names (jpg, jpeg, png 파일만 필터링)
image_files_test = [f for f in os.listdir('content_combine/data_test2')
                    if f.endswith(('.jpg', '.jpeg', '.png')) and ".ipynb_checkpoints" not in f]

labels = ['B-sign', 'I-sign',
'B-recipient_key', 'I-recipient_key',
'B-recipient_name', 'I-recipient_name',
'B-recipient_phone_number_key', 'I-recipient_phone_number_key',
'B-recipient_phone_number', 'I-recipient_phone_number',
'B-recipient_address_do', 'I-recipient_address_do',
'B-recipient_address_si', 'I-recipient_address_si',
'B-recipient_address_gun', 'I-recipient_address_gun',
'B-recipient_address_gu', 'I-recipient_address_gu',
'B-recipient_address_eup', 'I-recipient_address_eup',
'B-recipient_address_myeon', 'I-recipient_address_myeon',
'B-recipient_address_ri', 'I-recipient_address_ri',
'B-recipient_address_dong', 'I-recipient_address_dong',
'B-recipient_address_jibeon', 'I-recipient_address_jibeon',
'B-recipient_address_ro_name', 'I-recipient_address_ro_name',
'B-recipient_address_gil_name', 'I-recipient_address_gil_name',
'B-recipient_address_ro_number', 'I-recipient_address_ro_number',
'B-recipient_address_building_number', 'I-recipient_address_building_number',
'B-recipient_address_room_number', 'I-recipient_address_room_number',
'B-recipient_address_detail', 'I-recipient_address_detail',
'B-sender_key', 'I-sender_key',
'B-sender_name', 'I-sender_name',
'B-sender_phone_number_key', 'I-sender_phone_number_key',
'B-sender_phone_number', 'I-sender_phone_number',
'B-sender_address_do', 'I-sender_address_do',
'B-sender_address_si', 'I-sender_address_si',
'B-sender_address_gun', 'I-sender_address_gun',
'B-sender_address_gu', 'I-sender_address_gu',
'B-sender_address_eup', 'I-sender_address_eup',
'B-sender_address_myeon', 'I-sender_address_myeon',
'B-sender_address_ri', 'I-sender_address_ri',
'B-sender_address_dong', 'I-sender_address_dong',
'B-sender_address_jibeon', 'I-sender_address_jibeon',
'B-sender_address_ro_name', 'I-sender_address_ro_name',
'B-sender_address_gil_name', 'I-sender_address_gil_name',
'B-sender_address_ro_number', 'I-sender_address_ro_number',
'B-sender_address_building_number', 'I-sender_address_building_number',
'B-sender_address_room_number', 'I-sender_address_room_number',
'B-sender_address_detail', 'I-sender_address_detail',
'B-volume_key', 'I-volume_key',
'B-volume', 'I-volume',
'B-delivery_message_key', 'I-delivery_message_key',
'B-delivery_message', 'I-delivery_message',
'B-product_name_key', 'I-product_name_key',
'B-product_name', 'I-product_name',
'B-tracking_number_key', 'I-tracking_number_key',
'B-tracking_number', 'I-tracking_number',
'B-weight_key', 'I-weight_key',
'B-weight', 'I-weight',
'B-terminal_number', 'I-terminal_number',
'B-company_name', 'I-company_name',
'B-handwriting', 'I-handwriting',
'B-others', 'I-others'
]
labels

idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}
label2idx

#print(words)
#print(bounding_boxes)

#Now let's define the PyTorch dataset:
#BERT는 양방향으로 텍스트의 문맥을 학습하여, 특정 단어의 의미를 주변 단어를 통해 파악합니다.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = krri_invoice(image_file_names=image_files_train, tokenizer=tokenizer, max_length=512, target_size=1024)

#len(image_files_train) #423
#len(train_dataset) #422

for i in range(len(train_dataset)):
  print(train_dataset[i])
encoding = train_dataset[0]
encoding.keys()

tokenizer.decode(encoding.input_ids)
encoding['resized_image'].shape

test_image = ToPILImage()(encoding['resized_image']).convert("RGB")
#test_image

draw = ImageDraw.Draw(test_image, "RGBA")
for bbox in encoding['resized_and_aligned_bounding_boxes'].tolist():
    draw.rectangle(bbox, outline='red', width=1)

test_image #바운딩 + 리사이즈된 컬러 이미지 나와야함(47 => 워드, 박스 개수, 라벨개수)


train_dataloader = DataLoader(train_dataset, batch_size=4) # 기존 4
batch = next(iter(train_dataloader))

## 모델 정의 토큰 분류를 하는 모델 설정 
model = LayoutLMForTokenClassification()
batch.keys()
image.shape
###### 여기부터 12/26 모델 입력 구성
input_ids=batch['input_ids']
bbox=batch['bbox']
attention_mask=batch['attention_mask']
token_type_ids=batch['token_type_ids']
labels=batch['labels']
resized_images = batch['resized_image'] # shape (N, C, H, W), with H = W = 224
resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'] # single torch tensor that also contains the batch index for every bbox at image size 224


#예측 수행
outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                labels=labels, resized_images=resized_images, resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes)

outputs.loss
#예측 결과의 텐서크기 (배치크기, 시퀀스 길이, 클래스 개수)
outputs.logits.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_dataset = krri_invoice(image_file_names=image_files_test, tokenizer=tokenizer, max_length=512, target_size=1024, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# 테스트 데이터를 사용하여 학습된 모델을 평가

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ocr 부분 

reader = easyocr.Reader(['ko', 'en'], model_storage_directory='C:\\Users\\USER\\EasyOCR\\workspace\\user_network_dir', 
                        user_network_directory='C:\\Users\\USER\\EasyOCR\\workspace\\user_network_dir', 
                        recog_network='custom') #gpu=True

layoutlm_preds, easyocr_preds = None, []
layoutlm_out_label_ids, easyocr_out_label_ids = None, []
eval_loss = 0.0
nb_eval_steps = 0

def clean_labels(label_list):
    clean_list = []
    for label_seq in label_list:
        clean_seq = []
        for label in label_seq:
            # Check if label follows BIO format, otherwise assign "O" (non-entity)
            if label and (label.startswith("B-") or label.startswith("I-") or label == "O"):
                clean_seq.append(label)
            else:
                clean_seq.append("O")  # Non-entity for invalid labels
        clean_list.append(clean_seq)
    return clean_list

# 모델을 평가 모드로 설정합
## 모델 출력 준비 및 추론
## 모델 정의
model = LayoutLMForTokenClassification()
model.load_state_dict(torch.load("trained_model.pth"))
model.to(device)

model.eval()

def log_debug(message, **kwargs):
    print(f"[DEBUG] {message}")
    for key, value in kwargs.items():
        print(f" - {key}: {value}")

def evaluate_and_calculate_f1(model, test_dataloader, device):
    model.eval()
    layoutlm_true_labels, layoutlm_pred_labels = [], []
    easyocr_true_labels, easyocr_pred_labels = [], []

    start_time = time.time()  # 전체 평가 시작 시간
    total_layoutlm_time = 0  # LayoutLM의 총 소요 시간
    total_easyocr_time = 0   # EasyOCR의 총 소요 시간
    total_batches = len(test_dataloader)

    # 이미지별, 모듈별 시간 측정을 위한 리스트 및 딕셔너리
    image_module_times = []

    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            # LayoutLM 예측
            layout_start = time.time()
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            resized_images = batch['resized_image'].to(device)
            resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device)

            outputs = model(
                input_ids=input_ids, bbox=bbox, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, #labels=labels, 
                resized_images=resized_images, resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes
            )

            logits = outputs.logits # 예측결과
            preds = torch.argmax(logits, dim=-1) #예측결과를 넣어서 최종 클래스변환
            layoutlm_true_labels.extend(labels.view(-1).cpu().numpy())
            layoutlm_pred_labels.extend(preds.view(-1).cpu().numpy())

            layout_elapsed = time.time() - layout_start
            total_layoutlm_time += layout_elapsed

        # EasyOCR 예측 상세 시간 측정
        test_image_path = f'content_combine/data_test2/{image_files_test[idx]}'
        
        # 이미지별 모듈 시간 측정을 위한 딕셔너리
        current_image_times = {
            'image_filename': image_files_test[idx],
            'input_load': 0,
            'preprocessing': 0,
            'recognition': 0,
            'postprocessing': 0,
            'total_easyocr': 0,
            'bounding_box_count': 0  # 바운딩 박스 갯수
        }

        # 이미지 입력 시간 측정
        input_start = time.time()
        test_image = Image.open(test_image_path).convert("RGB")
        current_image_times['input_load'] = time.time() - input_start

        # 이미지 전처리 시간 측정
        preprocess_start = time.time()
        json_path = f'content_combine/annotation_combine/{image_files_test[idx].split(".")[0]}.json'
        with open(json_path, encoding='utf-8') as f:
            json_data = json.load(f)
        current_image_times['preprocessing'] = time.time() - preprocess_start

        # EasyOCR 전체 시간 측정
        easyocr_start = time.time()

        # 바운딩 박스 처리
        current_image_times['bounding_box_count'] = len(json_data['objects'])
        for obj in json_data['objects']:
            transcription = obj['transcription']
            easyocr_true_labels.append(transcription)
            left, top, right, bottom = int(obj['x']), int(obj['y']), int(obj['x'] + obj['width']), int(obj['y'] + obj['height'])
            
            # 개별 영역 크롭
            crop_start = time.time()
            cropped_image = test_image.crop((left, top, right, bottom))

            # Recognition 시간 측정
            recognition_start = time.time()
            easyocr_result = reader.readtext(np.array(cropped_image), detail=0)
            recognition_time = time.time() - recognition_start
            current_image_times['recognition'] += recognition_time

            # 후처리 시간 측정
            postprocess_start = time.time()
            predicted_text = easyocr_result[0] if easyocr_result else ""
            easyocr_pred_labels.append(predicted_text)
            current_image_times['postprocessing'] += time.time() - postprocess_start

        # EasyOCR 총 소요 시간
        current_image_times['total_easyocr'] = time.time() - easyocr_start
        
        # 이미지별 모듈 시간 저장
        image_module_times.append(current_image_times)

        total_easyocr_time += current_image_times['total_easyocr']

    # LayoutLM 평가 메트릭 계산
    layoutlm_precision = precision_score(layoutlm_true_labels, layoutlm_pred_labels, average='macro', zero_division=1)
    layoutlm_recall = recall_score(layoutlm_true_labels, layoutlm_pred_labels, average='macro', zero_division=1)
    layoutlm_f1 = f1_score(layoutlm_true_labels, layoutlm_pred_labels, average='macro', zero_division=1)
    layoutlm_accuracy = accuracy_score(layoutlm_true_labels, layoutlm_pred_labels)

    # EasyOCR 평가 메트릭 계산
    easyocr_precision = precision_score(easyocr_true_labels, easyocr_pred_labels, average='macro', zero_division=0)
    easyocr_recall = recall_score(easyocr_true_labels, easyocr_pred_labels, average='macro', zero_division=0)
    easyocr_f1 = f1_score(easyocr_true_labels, easyocr_pred_labels, average='macro', zero_division=0)
    easyocr_accuracy = accuracy_score(easyocr_true_labels, easyocr_pred_labels)

    end_time = time.time()  # 전체 평가 종료 시간
    total_elapsed = end_time - start_time
    avg_layoutlm_time = total_layoutlm_time / total_batches
    avg_easyocr_time = total_easyocr_time / total_batches

    # 결과 출력
    print("LayoutLM Evaluation Results:")
    print(f" - Accuracy: {layoutlm_accuracy}")
    print(f" - Precision: {layoutlm_precision}")
    print(f" - Recall: {layoutlm_recall}")
    print(f" - F1 Score: {layoutlm_f1}")

    print("\nEasyOCR 이미지별 모듈 평균 실행 시간:")
    # 각 모듈별 평균 시간 계산
    module_names = ['input_load', 'preprocessing', 'recognition', 'postprocessing', 'total_easyocr']
    avg_module_times = {}
    for module in module_names:
        avg_module_times[module] = sum(img_time[module] for img_time in image_module_times) / len(image_module_times)
        print(f" - {module}: {avg_module_times[module]:.8f} seconds")

    # 개별 이미지 상세 시간 출력 (선택사항)
    print("\n상세 이미지별 모듈 실행 시간 및 바운딩 박스 갯수:")
    for img_time in image_module_times:
        print(f"\n이미지: {img_time['image_filename']}")
        print(f" - Bounding Box Count: {img_time['bounding_box_count']}")
        for module in module_names:
            print(f" - {module}: {img_time[module]:.8f} seconds")
    
    print("\nEasyOCR Evaluation Results:")
    print(f" - Accuracy: {easyocr_accuracy}")
    print(f" - Precision: {easyocr_precision}")
    print(f" - Recall: {easyocr_recall}")
    print(f" - F1 Score: {easyocr_f1}")
    
    print("\nTotal Evaluation Time:", total_elapsed, "seconds")
    print("Average LayoutLM Time per Batch:", avg_layoutlm_time, "seconds")
    print("Average EasyOCR Time per Batch:", avg_easyocr_time, "seconds")

# 모델 평가 및 F1-score 계산 실행
evaluate_and_calculate_f1(model, test_dataloader, device)

def evaluate_and_visualize_with_labels(model, test_dataloader, device):
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            # LayoutLM 예측 수행
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            resized_images = batch['resized_image'].to(device)
            resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device)

            outputs = model(
                input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, 
                token_type_ids=token_type_ids, labels=labels, 
                resized_images=resized_images, resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        predictions = []
        for i, bbox in enumerate(batch['resized_and_aligned_bounding_boxes'][0].cpu().numpy()):
            label = idx2label[preds[0][i].item()]
            prob = 1.0  # 기본적으로 1.0으로 가정
            left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cropped_image = ToPILImage()(batch['resized_image'][0]).crop((left, top, right, bottom))
            
            # OCR 결과 획득
            if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:
                easyocr_result = reader.readtext(np.array(cropped_image), detail=0)
                ocr_text = easyocr_result[0] if easyocr_result else ""
            else:
                ocr_text = ""

            # JSON에서 ground truth 텍스트 가져오기
            gt_text = ""  # 예시용, JSON으로부터 실제 ground truth 텍스트를 가져오도록 수정 필요

            # 바운딩 박스와 OCR 텍스트, 레이블 추가
            bbox_coords = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            predictions.append((bbox_coords, label, ocr_text, gt_text, prob))

        # 시각화
        test_image = ToPILImage()(batch['resized_image'][0].cpu()).convert("RGB")
        test_image_with_predictions = visualize_predictions_with_labels(test_image, predictions, font_size=15)
        
        # 결과 이미지 출력
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(test_image_with_predictions)
        plt.axis('off')
        plt.show()

# 모델 평가 및 시각화 실행
evaluate_and_visualize_with_labels(model, test_dataloader, device)

