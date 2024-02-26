from flask import Flask, request, jsonify ,Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
# from google.colab.patches import cv2_imshow
import torch
import numpy as np
from flask_socketio import SocketIO, emit


app = Flask(__name__)
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# socketio = SocketIO(app,cors_allowed_origins="*")

socketio = SocketIO(app)
CORS(app)
socketio.init_app(app, cors_allowed_origins="*", logger=True, engineio_logger=True)
app.config['CORS_HEADERS'] = 'Content-Type'



personModel = YOLO('yolov8n.pt')  # Assuming model for detecting persons
shopliftingModel = YOLO('shoplifting-best.pt')  # Load the model for the other class
robberyModel = YOLO('robbery-best.pt')  # Load the model for the other class

flag = True

@socketio.on('connect')
def handle_connect():
    print('Client connected!')
    emit('status', data={'message': 'Welcome to the server!'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected!')

@socketio.on('message')
def handle_message(data):
    print('Received message from client:', data)
    emit('message', data, broadcast=True) 

@socketio.on('terminate')
def terminate_func():
    global flag
    flag = False

@socketio.on('startvideo')
def start_video():
    emit('videostart', data={'message': 'Video started!'})

@app.route("/api/video",methods=["POST","GET"])
def process():
    return jsonify({"msg":"Video received", "status":200})


# @app.route("/api/livevideo",methods=["POST","GET"])
# def process_live():
#     print("Hi")
#     vid = cv2.VideoCapture(0)  # Access webcam
#     while True:
#         ret, frame = vid.read()
#         confidence_threshold = 0.5
#         shoplifting_threshold = 0.5
#         # input_data = preprocess_frame(frame)
#         # output_data = session.run(None, input_data)
#         # processed_frame = visualize_predictions(frame, output_data)
#         # _, encoded_frame = cv2.imencode('.jpg', processed_frame)
#         # yield (b'--frame\r\n'
#         #         b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_frame) + b'\r\n')



def process_live_shoplifting(videolink,threshold=0.4):
#     data = request.get_json()
#     print(data['videolink'])
    # chunk_size = 20
    print("func started")
    global flag
    flag = True
    video = cv2.VideoCapture(videolink)
#     print(video)
    video_frames = []
    print("Video started")
    confidence_threshold = threshold
    shoplifting_threshold = threshold
    socketio.emit('videostart', data={'message': 'Video started!'})
    while flag:
        # emit('videostart', data={'message': 'Video started!'})
        
        ret, frame = video.read()
        if not ret:
            break
        # Track and detect actions
        person_results = personModel.predict(frame,classes=[0])
        for result in person_results:
            class_id = result.boxes.cpu().numpy().cls
            zero_indices = np.nonzero(class_id == 0)[0]
            bbox = result.boxes.cpu().numpy()
            confidence = result.boxes.cpu().numpy().conf
            for i in zero_indices:
                if confidence[i] >= confidence_threshold:
                    xyxys = bbox[i].xyxy[0].astype(int)
                    cropped_frame = frame[xyxys[1]:xyxys[3],xyxys[0]:xyxys[2]]
                    shoplifting_results = shopliftingModel(source = cropped_frame)
                    for shoplift in shoplifting_results:
                        print("classes: ",shoplift.boxes.cpu().numpy().cls)
                        shoplift_class = shoplift.boxes.cpu().numpy().cls
                        shoplift_conf = shoplift.boxes.cpu().numpy().conf
                        class_conf_dict = dict(zip(shoplift_class, shoplift_conf[:len(shoplift_class)]))
                        print(class_conf_dict)
                        if(len(class_conf_dict) > 1 and len(class_conf_dict) != 0):
                            class_detected = "Shoplifting" if max(class_conf_dict, key=class_conf_dict.get) == 1 else "Normal"
                            print(class_detected)
                            if class_detected == "Shoplifting" and class_conf_dict[1] >= shoplifting_threshold:
                                _,encoded_frame = cv2.imencode('.jpg', frame)
                                cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                            (xyxys[0] + 5, xyxys[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.rectangle(frame, (xyxys[0], xyxys[1]), (xyxys[2], xyxys[3]), (0, 0, 255), 2)
                                socketio.emit("Anamoly", {"Class": "Shoplifting","Coordinates" : str(xyxys)})
                            elif class_detected == "Normal" and class_conf_dict[0] >= shoplifting_threshold:
                                cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                                            (xyxys[0] + 5, xyxys[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.rectangle(frame, (xyxys[0], xyxys[1]), (xyxys[2], xyxys[3]), (0, 255, 0), 2)
                        elif(len(class_conf_dict) == 1):
                            class_detected = "Shoplifting" if list(class_conf_dict.keys())[0] == 1 else "Normal"
                            print(class_detected)
                            if class_detected == "Shoplifting" and class_conf_dict[1] >= shoplifting_threshold:
                                cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                                        (xyxys[0] + 5, xyxys[1]- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.rectangle(frame, (xyxys[0], xyxys[1]), (xyxys[2], xyxys[3]), (0, 0, 255), 2)
                                _,encoded_frame = cv2.imencode('.jpg', frame)
                                socketio.emit("Anamoly", {"Class": "Shoplifting","Coordinates" : str(xyxys)})
                            elif class_detected == "Normal" and class_conf_dict[0] >= shoplifting_threshold:
                                cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                                        (xyxys[0] + 5, xyxys[1]- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.rectangle(frame, (xyxys[0], xyxys[1]), (xyxys[2], xyxys[3]), (0, 255, 0), 2)
        _, encoded_frame = cv2.imencode('.jpg', frame)
        video_frames.append(encoded_frame.tobytes())
        # socketio.emit("frames",{"frame":encoded_frame.tobytes()})
        # cv2.imshow('frame',frame)
        # return encoded_frame.tobytes()
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
        # video_frames.append(frame)                                
    socketio.emit("videocompleted")    

#     return jsonify({"msg":"Video received", "status":200})



def process_live_robbery(videolink,threshold=0.4):
    #     data = request.get_json()
#     print(data['videolink'])
    # chunk_size = 20
    print("func started")
    global flag
    flag = True
    video = cv2.VideoCapture(videolink)
#     print(video)
    video_frames = []
    print("Video started")
    # confidence_threshold = 0.5
    robbery_threshold = threshold
    socketio.emit('videostart', data={'message': 'Video started!'})
    while flag:
        # emit('videostart', data={'message': 'Video started!'})
        
        ret, frame = video.read()
        if not ret:
            break
        # Track and detect actions
        
        robbery_results = robberyModel(source = frame)
        for robbery in robbery_results:
            print("classes: ",robbery.boxes.cpu().numpy().cls)
            robbery_class = robbery.boxes.cpu().numpy().cls
            robbery_conf = robbery.boxes.cpu().numpy().conf
            bbox = robbery.boxes.cpu().numpy()
            print("bbox",bbox)
            class_conf_dict = dict(zip(robbery_class, robbery_conf[:len(robbery_class)]))
            print(class_conf_dict)
            if(len(class_conf_dict) > 1 and len(class_conf_dict) != 0):
                class_detected = "Robbery" if max(class_conf_dict, key=class_conf_dict.get) == 1 else "Normal"
                print(class_detected)
                if class_detected == "Robbery" and class_conf_dict[1] >= robbery_threshold:
                    xyxys = bbox.xyxy.astype(int)
                    for i in range(len(xyxys)):
                        cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                            (xyxys[i][0] + 5, xyxys[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.rectangle(frame, (xyxys[i][0], xyxys[i][1]), (xyxys[i][2], xyxys[i][3]), (0, 0, 255), 2)
                    socketio.emit("Anamoly", {"Class": "Robbery","Coordinates" : str(123)})
                elif class_detected == "Normal" and class_conf_dict[0] >= robbery_threshold:
                    pass
                    # cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                                # (xyxys[0] + 5, xyxys[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # cv2.rectangle(frame, (xyxys[0], xyxys[1]), (xyxys[2], xyxys[3]), (0, 255, 0), 2)
            elif(len(class_conf_dict) == 1):
                class_detected = "Robbery" if list(class_conf_dict.keys())[0] == 1 else "Normal"
                print(class_detected)
                if class_detected == "Robbery" and class_conf_dict[1] >= robbery_threshold:
                    xyxys = bbox.xyxy.astype(int)
                    for i in range(len(xyxys)):
                        cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                            (xyxys[i][0] + 5, xyxys[i][1]- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.rectangle(frame, (xyxys[i][0], xyxys[i][1]), (xyxys[i][2], xyxys[i][3]), (0, 0, 255), 2)
                    socketio.emit("Anamoly", {"Class": "Robbery","Coordinates" : str(123)})
                elif class_detected == "Normal" and class_conf_dict[0] >= robbery_threshold:
                    pass
                    # cv2.putText(frame, f"{class_detected}: {class_conf_dict[max(class_conf_dict, key=class_conf_dict.get)]:.2f}%",
                            # (xyxys[0] + 5, xyxys[1]- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # cv2.rectangle(frame, (xyxys[0], xyxys[1]), (xyxys[2], xyxys[3]), (0, 255, 0), 2)
        _, encoded_frame = cv2.imencode('.jpg', frame)
        video_frames.append(encoded_frame.tobytes())
        # cv2.imshow('frame',frame)
        # return encoded_frame.tobytes()
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
        # video_frames.append(frame)                                
    socketio.emit("videocompleted")    

#     return jsonify({"msg":"Video received", "status":200})



    
@app.route("/api/livevideo/<model>/<threshold>")
def video_feed(model,threshold):
    print("Hi")
    print(model,threshold)
    if(model == "shoplifting"):
        return Response(process_live_shoplifting(0,int(threshold)/100),mimetype='multipart/x-mixed-replace; boundary=frame')
    if(model == "robbery"):
        return Response(process_live_robbery(0,int(threshold)/100),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/analyzevideo/<model>/<threshold>/<videoParam1>/<videoParam2>")
def analyze_video(model,threshold,videoParam1,videoParam2):
    # videoParam1 = request.args.get("videoParam1");
    # videoParam2 = request.args.get("videoParam2");
    print(model,threshold,videoParam1,videoParam2)
    video = f"https://res.cloudinary.com/dxkxrii3x/video/upload/{videoParam1}/{videoParam2}"
    print(video,model,threshold)
    if(model == "shoplifting"):
        return Response(process_live_shoplifting(video,int(threshold)/100),mimetype='multipart/x-mixed-replace; boundary=frame')
    if(model == "robbery"):
        return Response(process_live_robbery(video,int(threshold)/100),mimetype='multipart/x-mixed-replace; boundary=frame')





# @app.route("/api/livevideo")
# def video_feed():
#     print("Hi")
#     return Response(process_live(),mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    print("Hello world")
    socketio.run(app,host='0.0.0.0',port=8000,debug=True)
    


# app = Flask(__name__)
# socketio = SocketIO(app)
# CORS(app)
# socketio.init_app(app, cors_allowed_origins="*", logger=True, engineio_logger=True)



# @socketio.on('connect')
# def handle_connect():
#     print('Client connected!')
#     # emit('connect', data={'message': 'Welcome to the server!'})

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected!')

# @socketio.on('message')
# def handle_message(data):
#     print('Received message from client:', data)
#     emit('message', data, broadcast=True) 

# if __name__ == "__main__":
#     print("Hello world")
#     socketio.run(app,host='0.0.0.0',port=8000,debug=True)