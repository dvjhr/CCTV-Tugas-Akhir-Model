from _utils import face_to_box, inside_area
from _generate_filename import generate_filename
from _replace_file import replace_file
import cv2
import time
from datetime import datetime, timedelta
import numpy as np
import traceback

print("Starting program...")

from ultralytics import YOLO
import torch

torch.cuda.set_device(0)

yolo_model = YOLO('yolov8n.pt')

from insightface.app import FaceAnalysis
model = FaceAnalysis(name='buffalo_mm', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_thresh=0.5)

import psycopg2
from psycopg2 import sql

connection_params = {
    'host': 'localhost',
    'database': 'tugas_akhir',
    'user': 'postgres',
    'password': 'postgres',
}

conn = psycopg2.connect(**connection_params)

# Create a cursor
cursor = conn.cursor()
insert_query = sql.SQL("INSERT INTO face_data (path, embedding) VALUES (%s, %s)")

print("""
1. Lift Gerbang Barat - 101 (1)
2. Selasar Gerbang Barat - 201 (2)
3. Selasar Lab KCKS - 301 (3)
4. Lab KCKS Belakang - 1101 (4)
""")
while True:
    input_def = input("Select Camera: ")
    if str(input_def) == "1" or str(input_def) == "2" or str(input_def) == "3" or str(input_def) == "4":
        input_name = f'rtsp://KCKS:majuteru5@10.15.40.48/Streaming/Channels/{input_def}01'
        break
    else:
        print("Please try again with correct number")


img_input = cv2.VideoCapture(input_name)

fps_list = []
infer_list = {}

frame_count = 0
fps_start_time = time.time()

temp_database = {}

print("Running main program...")
print("Press Q to interrupt")

while (True):
    ret, img = img_input.read()
    img_copy = img.copy()
    # print(img.shape)

    if frame_count % 5 == 0:
        pass
    else:    
        detection_result = yolo_model.track(img, conf=0.4, iou=0.9, device='0', classes=0, show=False, stream=True, verbose=False, persist=True, tracker="custom_bytetrack.yaml")
        try:
            faces = model.get(img)
            for det in detection_result:
                dict_box = {}
                # print(det.boxes.xyxy.cpu().numpy().astype(int))
                boxes = det.boxes.xyxy.cpu().numpy().astype(int)
                ids = det.boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    dict_box[id] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
                    cv2.putText(img, f"{id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                print(f"{len(faces)} Found")
                for i, face in enumerate(faces):
                    img_copy2 = img_copy.copy()
                    face_box = list(map(int, face["bbox"]))
                    print(f'{i}: {face["embedding"].shape}')
                    cv2.rectangle(img, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (0, 0, 255), 2)
                    for area_box in dict_box:
                        if inside_area(face_box, dict_box[area_box]):
                            print("IF 1")
                            # print(f"{generate_filename(input_name)}")
                            print(i, area_box, inside_area(face_box, dict_box[area_box]))
                            try: 
                                print("TRY")
                                temp_database[id]["embedding"] = np.mean((temp_database[id]["embedding"], face["embedding"]), axis=0)
                                img_copy2 = cv2.rectangle(img_copy2, (int(dict_box[area_box][0]), int(dict_box[area_box][1])), (int(dict_box[area_box][2]), int(dict_box[area_box][3])), (255, 255, 0), 2)
                                img_copy2 = cv2.putText(img_copy2, f"{id}", (dict_box[area_box][0], dict_box[area_box][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                img_copy2 = cv2.rectangle(img_copy2, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (0, 0, 255), 2)
                                replace_name = replace_file(temp_database[id]["filename"])
                                cv2.imwrite(f"static/img/record/{replace_name}", img_copy2)
                                print("TRY 2")
                            except:
                                print("EXCEPT")
                                print(traceback.print_exc())
                                filename_result = generate_filename(input_name).replace('XX', str(id))
                                print(filename_result)
                                temp_database[id] = {"embedding": face["embedding"], "filename": filename_result}
                                img_copy2 = cv2.rectangle(img_copy2, (int(dict_box[area_box][0]), int(dict_box[area_box][1])), (int(dict_box[area_box][2]), int(dict_box[area_box][3])), (255, 255, 0), 2)
                                img_copy2 = cv2.putText(img_copy2, f"{id}", (dict_box[area_box][0], dict_box[area_box][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                img_copy2 = cv2.rectangle(img_copy2, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (0, 0, 255), 2)
                                
                                cv2.imwrite(f"static/img/record/{filename_result}", img_copy2)
                                cursor.execute(insert_query, (f"{generate_filename(input_name)}", face["embedding"]))
                                insert_query = sql.SQL("INSERT INTO face_data (path, embedding) VALUES (%s, %s)")
                                cursor.execute(insert_query, (temp_database[id]['filename'], temp_database[id]['embedding'].tolist()))
                                conn.commit()
                                print("COMMITTED")

                            # finally:
                            #     print("FINALLY 1")
                            #     print(generate_filename(input_name), dict_box[id])
                            #     img_copy2 = cv2.rectangle(img_copy2, (int(dict_box[area_box][0]), int(dict_box[area_box][1])), (int(dict_box[area_box][2]), int(dict_box[area_box][3])), (255, 255, 0), 2)
                            #     img_copy2 = cv2.putText(img_copy2, f"{id}", (dict_box[area_box][0], dict_box[area_box][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            #     img_copy2 = cv2.rectangle(img_copy2, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (0, 0, 255), 2)
                                
                            #     cv2.imwrite(f"record/{generate_filename(input_name).replace('XX', str(id))}.jpg", img_copy2)
                            #     print("FINALLY 2")
                        else:
                            print("ELSE")
                            print(i, area_box, inside_area(face_box, dict_box[area_box]))

        except:
            pass

    frame_count += 1
    fps_elapsed_time = time.time() - fps_start_time
    fps_result = round((frame_count / fps_elapsed_time), 3)
    fps_list.append(fps_result)

    cv2.putText(img, str(fps_result), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    img = cv2.resize(img, (1280, 720))
    cv2.imshow(f"Running System", img)

 # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # release the video capture object
img_input.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

print("Program stopped")
exit()