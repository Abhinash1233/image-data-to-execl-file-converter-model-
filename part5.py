import paddleocr
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

ocr = paddleocr.PaddleOCR()
img = '/Users/Sachin/OneDrive/Desktop/pdf to img/result data/page0.jpg'
data = ocr.ocr(img)
image =cv2.imread(img)
image_height = image.shape[0]
image_width  = image.shape[1]

data1 = []
for blocks in data:
    for i in range(len(blocks)):
        block  = blocks[i]
        data1.append(block)
       
 
boxes = []
texts = []
probabilities = []
for out in data1:
    k =out[0]
    y =out[1][0]
    q =out[1][1]
    boxes.append(k)
    texts.append(y)
    probabilities.append(q)


   
for box in boxes:
    x = int(box[0][0])
    y = int(box[0][1])
    w = int(box[2][0])
    z = int(box[2][1])
    #cv2.rectangle(image,(x,y),(w,z),(0,0,255))
    
horiz_box = []
vert_box =[]
for box in boxes:
    x_h,x_v = 0, int(box[0][0])
    y_h,y_v = int(box[0][1]), 0

    width_h,width_v = image_width, int(box[2][0]-box[0][0])
    height_h,height_v = int(box[2][1]-box[0][1]),image_height
    horiz_box.append([x_h,y_h,x_h+width_h,y_h+height_h])
    vert_box.append([x_v,y_v,x_v+width_v,y_v+height_v])
   # cv2.rectangle(image,(x_h,y_h),(x_h+width_h,y_h+height_h),(0,255,0),1)
   # cv2.rectangle(image,(x_v,y_v),(x_v+width_v,y_v+height_v),(255,255,0),1)

horiz_out = tf.image.non_max_suppression(
    horiz_box,
    probabilities,
    max_output_size = 1000,
    iou_threshold = 0.1,
    score_threshold = float('-inf'),
    name =None
)
horiz_lines = np.sort(np.array(horiz_out))


for val in horiz_lines:
   cv2.rectangle(image,(int(horiz_box[val][0]),int(horiz_box[val][1])),(int(horiz_box[val][2]),int(horiz_box[val][3])),(255,0,0),2)


vert_out = tf.image.non_max_suppression(
    vert_box,
    probabilities,
    max_output_size = 5000,
    iou_threshold = 0.01,
    score_threshold = float(0.1),
    name =None
)


vert_lines = np.sort(np.array(vert_out))



for val in vert_lines:
    cv2.rectangle(image,(int(vert_box[val][0]),int(vert_box[val][1])),(int(vert_box[val][2]),int(vert_box[val][3])),(0,0,255),2)
    

cv2.imwrite('verthori.jpg',image)


out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

unordered_boxes = []
for i in vert_lines:
    unordered_boxes.append(vert_box[i][0])

ordered_boxes = np.argsort(unordered_boxes)



def intersection(box_1,box_2):
    return [box_2[0],box_1[1],box_2[2],box_1[3]]

def iou(box_1,box_2):
    x_1 = max(box_1[0],box_2[0])
    y_1 = max(box_1[1],box_2[1])
    x_2 = min(box_1[2],box_2[2])
    y_2 = min(box_1[3],box_2[3])
    
    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1,0)))
    if inter ==0:
        return 0
    box_1_area = abs((box_1[2] -box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] -box_2[0]) * (box_2[3] - box_2[1]))
    return inter / float(box_1_area + box_2_area - inter)


for i in range(len(horiz_lines)):
    for j in range(len(vert_lines)):
        resultant = intersection(horiz_box[horiz_lines[i]], vert_box[vert_lines[ordered_boxes[j]]])
        
       
        for b in range(len(boxes)):
            the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
            if(iou(resultant,the_box)>0.01):
                out_array[i][j] = texts[b]
               


pd.DataFrame(out_array).to_csv('sample.csv')


df = pd.read_csv('sample.csv')
df.to_excel('output.xlsx', index=False)










