"""
AI-HUB의 레이블 파일들을 잘 모아서 PaddleOCR 형태에 맞게 수정
"""

import json

with open("/home/dataset/AI-Hub/간판_가로형간판_000001.json") as f:
    data = json.load(f)
    
    print(data[0])
# json.load("")

# icdar_c4_train_imgs/img_61.jpg	[{"transcription": "###", "points": [[427, 293], [469, 293], [468, 315], [425, 314]]}, 
#                                 {"transcription": "###", "points": [[480, 291], [651, 289], [650, 311], [479, 313]]}, 
#                                 {"transcription": "Ave", "points": [[655, 287], [698, 287], [696, 309], [652, 309]]}, 
#                                 {"transcription": "West", "points": [[701, 285], [759, 285], [759, 308], [701, 308]]}, 
#                                 {"transcription": "YOU", "points": [[1044, 531], [1074, 536], [1076, 585], [1046, 579]]}, 
#                                 {"transcription": "CAN", "points": [[1077, 535], [1114, 539], [1117, 595], [1079, 585]]}, 
#                                 {"transcription": "PAY", "points": [[1119, 539], [1160, 543], [1158, 601], [1120, 593]]}, 
#                                 {"transcription": "LESS?", "points": [[1164, 542], [1252, 545], [1253, 624], [1166, 602]]}, 
#                                 {"transcription": "Singapore's", "points": [[1032, 177], [1185, 73], [1191, 143], [1038, 223]]}, 
#                                 {"transcription": "no.1", "points": [[1190, 73], [1270, 19], [1278, 91], [1194, 133]]}]