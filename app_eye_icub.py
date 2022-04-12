from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

import cv2,time,math

global accuracy
accuracy = 1200

Conf_threshold = 0.5     # proba mini accepted
NMS_threshold = 0.4      # reduice this value de decrese the nb of box found
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


from tobiiglassesctrl import TobiiGlassesController

Builder.load_string('''
<WelcomeScreen>:   
    canvas.before: 
        Color: 
            rgba: (1,1, 1,1)
        Rectangle: 
            size: root.width, root.height 
            pos: self.pos
    
    FloatLayout:
        Label:
            text :"Tobii.App V0.2"
            color: 0, 0, 0, 1
            pos_hint: {"center_x":0.5, "center_y":0.02}
            size_hint: 0.8, 0.4
        Image:
            id: img1
            source: "tobii.jpg"
            pos_hint: {"center_x":0.5, "center_y":0.5}
            size_hint: 0.8, 1
        Image:
            id: img2
            source: "logo_iit.png"
            pos_hint: {"center_x":0.25, "center_y":0.9}
            size_hint: 0.2, 0.2
        Image:
            id: img3
            source: "logo_tobii.jpg"
            pos_hint: {"center_x":0.75, "center_y":0.9}
            size_hint: 0.2, 0.2
        Button:
            id: btn1
            text: "Connection" if btn1.state == "normal" else "Connection ... "
            font_size: 30
            pos_hint: {"center_x":0.5, "center_y":0.1}
            size_hint: 0.55, 0.12 
            on_release:root.connect()

<CalibrationScreen>:   
    canvas.before: 
        Color: 
            rgba: (1, 1, 1, 1)
        Rectangle: 
            size: root.width, root.height 
            pos: self.pos
    
    FloatLayout:
        Label:
            text :"Watch the target during calibration."
            color: 0, 0, 0, 1
            pos_hint: {"center_x":0.5, "center_y":0.93}
            size_hint: 0.8, 0.4
        Image:
            id: img1
            source: "target.png"
            pos_hint: {"center_x":0.5, "center_y":0.55}
            size_hint: 0.5, 0.5
        Button:
            id: btn1
            text: "Start Calibration"
            font_size: 30
            pos_hint: {"center_x":0.5, "center_y":0.1}
            size_hint: 0.55, 0.12 
            on_release:root.calibration()

<MenuScreen>:   
    canvas.before: 
        Color: 
            rgba: (1, 1, 1, 1)
        Rectangle: 
            size: root.width, root.height 
            pos: self.pos
     
    FloatLayout:
        Label:
            text :"MENU"
            color: 1, 1, 1, 1
            pos_hint: {"center_x":0.5, "center_y":0.93}
            size_hint: 0.8, 0.4
        Image:
            id: img1
            source: "tobii.jpg"
            pos_hint: {"center_x":0.5, "center_y":0.5}
            size_hint: 0.5, 0.5
        Image:
            id: img2
            source: "logo_iit.png"
            pos_hint: {"center_x":0.25, "center_y":0.9}
            size_hint: 0.2, 0.2
        Image:
            id: img3
            source: "logo_tobii.jpg"
            pos_hint: {"center_x":0.75, "center_y":0.9}
            size_hint: 0.2, 0.2
        Button:
            id: btn1
            text: "Icub Eye Tracker"
            font_size: 30
            pos_hint: {"center_x":0.25, "center_y":0.06}
            size_hint: 0.5, 0.12 
            on_release:root.button_eye_tracker()
        Button:
            id: btn2
            text: "Accelerometers"
            font_size: 30
            pos_hint: {"center_x":0.75, "center_y":0.18}
            size_hint: 0.5, 0.12 
            on_release:root.button_acc()
        Button:
            id: btn3
            text: "Object Detection"
            font_size: 30
            pos_hint: {"center_x":0.75, "center_y":0.06}
            size_hint: 0.5, 0.12 
            on_release:root.button_object_detection()
        Button:
            id: btn4
            text: " . . . "
            font_size: 30
            pos_hint: {"center_x":0.25, "center_y":0.18}
            size_hint: 0.5, 0.12 
            on_release:root.button_other()
        

<AccScreen>:   
    canvas.before: 
        Color: 
            rgba: (1, 1, 1, 1)
        Rectangle: 
            size: root.width, root.height 
            pos: self.pos
    
    FloatLayout:
        Image:
            id: img1
            source: "accelerometer.png"
            pos: 50,100
            size_hint: 0.8, 0.8
        Button:
            id: btn1
            text: "Start" if btn1.state == "normal" else "Runing... "
            font_size: 30
            pos_hint: {"center_x":0.25, "center_y":0.06}
            size_hint: 0.5, 0.12 
            on_release:root.start()
        Button:
            id: btn2
            text: "MENU"
            font_size: 30
            pos_hint: {"center_x":0.75, "center_y":0.06}
            size_hint: 0.5, 0.12 
            on_release:root.menu()

<IcubScreen>:   
    canvas.before: 
        Color: 
            rgba: (1, 1, 1, 1)
        Rectangle: 
            size: root.width, root.height 
            pos: self.pos
       
    FloatLayout:
        Image:
            id: imag1
            source: "icub_logo.png"
            pos: 50,100
            size_hint: 0.8, 0.8
        Button:
            id: btn1
            text: "Start" if btn1.state == "normal" else "Runing... "
            font_size: 30
            pos_hint: {"center_x":0.25, "center_y":0.06}
            size_hint: 0.5, 0.12
            on_release:root.start()
        Button:
            id: btn2
            text: "MENU"
            font_size: 30
            pos_hint: {"center_x":0.75, "center_y":0.06}
            size_hint: 0.5, 0.12
            on_release:root.menu()
            
<ObjectDetectionScreen>:   
    canvas.before: 
        Color: 
            rgba: (1, 1, 1, 1)
        Rectangle: 
            size: root.width, root.height 
            pos: self.pos
       
    FloatLayout:
        Image:
            id: img1
            source: "object_detection.png"
            pos: 50,100
            size_hint: 0.8, 0.8
        Button:
            id: btn1
            text: "Start" if btn1.state == "normal" else "Runing... "
            font_size: 30
            pos_hint: {"center_x":0.25, "center_y":0.06}
            size_hint: 0.5, 0.12
            on_release:root.start()
        Button:
            id: btn2
            text: "MENU"
            font_size: 30
            pos_hint: {"center_x":0.75, "center_y":0.06}
            size_hint: 0.5, 0.12
            on_release:root.menu()
''')

display_dim = Window.size
Window.size = (800,800)

class TobiiApp(App):
    def build(self):
        print('Starting  Tobbi app V1')
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name='welcomeScreen'))
        sm.add_widget(CalibrationScreen(name='calibration1Screen'))
        sm.add_widget(MenuScreen(name='menuScreen'))
        sm.add_widget(AccScreen(name='accScreen'))
        sm.add_widget(IcubScreen(name='icubScreen'))
        sm.add_widget(ObjectDetectionScreen(name='objectDetectionScreen'))
        return sm
    
class WelcomeScreen(Screen):
    def connect(self):
        print('Trying connection with Tobii glasses')     
        global ipv4_address, tobiiglasses, project_id, participant_id
        ipv4_address = "192.168.71.50"
        tobiiglasses = TobiiGlassesController(ipv4_address, video_scene=True)
        project_id = tobiiglasses.create_project("Test live_scene_and_gaze.py")
        participant_id = tobiiglasses.create_participant(project_id, "participant_test")
        print('Connection Done')
        self.manager.current = 'calibration1Screen'
        
class CalibrationScreen(Screen):
    def calibration(self):
        print('Start calibration')
        global ipv4_address, tobiiglasses, project_id, participant_id, calibration_id, res
        calibration_id = tobiiglasses.create_calibration(project_id, participant_id)
        tobiiglasses.start_calibration(calibration_id)
        res = tobiiglasses.wait_until_calibration_is_done(calibration_id)
        if res is False:
            print("Calibration failed!")
            self.manager.current = 'welcomeScreen'
        else:
            print('Calibration done')
            self.manager.current = 'menuScreen'
                 
        tobiiglasses.start_streaming()
        nb_mesure = 0
        while(nb_mesure<10):
            info_tobii_glasses = tobiiglasses.get_data()
            dico_r_eye  = info_tobii_glasses['right_eye']['pc']
            dico_l_eye  = info_tobii_glasses['left_eye']['pc']
            if dico_r_eye['ts'] > 0 and dico_l_eye['ts'] > 0:
                    data = dico_r_eye['pc']
                    z_right = abs(data[2])
                    #print('z_right',z_right)
                    data = dico_l_eye['pc']
                    z_left = abs(data[2])
                    #print('z_left',z_left)
                    time.sleep(0.1)
                    nb_mesure+=1
                    
            z_max = 31.47
            z_min = 23.39
            global min_max
            min_max = (37, 30, 29, 23, z_max, z_min)
        
class MenuScreen(Screen):
    def button_eye_tracker(self):
        self.manager.current = 'icubScreen'
    def button_acc(self):
        self.manager.current = 'accScreen'
    def button_object_detection(self):
        self.manager.current = 'objectDetectionScreen'
    def button_other(self):
        print('other...')
        pass
    
class AccScreen(Screen):      
    def start(self):
        tobiiglasses.start_streaming()
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        temps = []
        figure, axis = plt.subplots(2, 2)
        plt.axis([0, 10, 0, 100])
        i = 0
        while(1):
            info_tobii_glasses = tobiiglasses.get_data()
            dico_acc = info_tobii_glasses['mems']['ac']
            if dico_acc['ts'] > 0:
                data = dico_acc['ac']
                x_acc = data[0]
                y_acc = data[1]
                z_acc = data[2]
                sum_acc =  abs(data[0])+abs(data[1])+abs(data[2])
                
                data1.append(x_acc)
                data2.append(y_acc)
                data3.append(z_acc)
                data4.append(sum_acc)
                temps.append(i)
                
                axis[0, 0].plot(temps, data1)
                axis[0, 0].set_title("X")
                  
                axis[0, 1].plot(temps, data2)
                axis[0, 1].set_title("Y")
                
                axis[1, 0].plot(temps, data3)
                axis[1, 0].set_title("Z")
                  
                axis[1, 1].plot(temps, data4)
                axis[1, 1].set_title("Sum")
                    
                if i > 20:
                   plt.axis([i-18, i+2, -20, 20]) 
                
                plt.pause(0.001)
                i+=1
                
                '''
                if 
                 break'''
            
            
    def menu(self):
        self.manager.current = 'menuScreen'


class IcubScreen(Screen):
    def start(self):
        global min_max
        min_max = [37, 30, 29, 22, 31, 24]
        tobiiglasses.start_streaming()
        
        '''icub_img = cv2.imread('init_icub.png',1)
        icub_img = cv2.resize(icub_img, (540, 600))
        height, width = icub_img.shape[:2]
        rotate_matrix = cv2.getRotationMatrix2D(center=(width/2, height/2), angle=-1, scale=1)
        icub_img = cv2.warpAffine(src=icub_img, M=rotate_matrix, dsize=(width, height))
        icub_img = cv.imwrite('icub.png',icub_img)'''
        
        r_eye_x = 230
        r_eye_y = 290
        l_eye_x = 345
        l_eye_y = 290
        alpha = 0
        
        while(1):
            
            info_tobii_glasses = tobiiglasses.get_data()
            dico_r_eye  = info_tobii_glasses['right_eye']['pc']
            if dico_r_eye['ts'] > 0:
                data = dico_r_eye['pc']
                x = abs(data[0])
                y = abs(data[1])
                deltaX = abs(min_max[0]-min_max[1])
                deltaY = abs(min_max[2]-min_max[3])
                r_eye_x = int(230+40*((x-min_max[0])/deltaX))
                r_eye_y = int(290+30*((y-min_max[2])/deltaY))
                # PUT SOME LIMIT VALUE
            
            dico_l_eye  = info_tobii_glasses['left_eye']['pc']
            if dico_l_eye['ts'] > 0:
                data = dico_l_eye['pc']
                x = abs(data[0])
                y = abs(data[1])
                deltaX = abs(min_max[0]-min_max[1])
                deltaY = abs(min_max[2]-min_max[3])
                l_eye_x = int(345-40*((x-min_max[0])/deltaX))
                l_eye_y = int(290+30*((y-min_max[2])/deltaY))
            
            dico_gyroscope = info_tobii_glasses['mems']['gy']
            if dico_gyroscope['ts'] > 0:
                data = dico_gyroscope['gy']
                alpha = data[0]
                   
            dico_acc = info_tobii_glasses['mems']['ac']
            if dico_acc['ts'] > 0:
                data = dico_acc['ac']
                #print(data)
                x_acc = data[0]
                y_acc = data[1]
                z_acc = data[2]
                #print(x_acc+y_acc+z_acc)
            
            
            gyro_angle = 0  
            icub_img = cv2.imread('icub.png',1)
            height, width = icub_img.shape[:2]
            rotate_matrix = cv2.getRotationMatrix2D(center=(width/2, height/2), angle=gyro_angle, scale=1)
            icub_img = cv2.warpAffine(src=icub_img, M=rotate_matrix, dsize=(width, height))
            cv2.circle(icub_img,(r_eye_x, r_eye_y), 15, (0,0,0), -1)
            cv2.circle(icub_img,(l_eye_x, l_eye_y), 15, (0,0,0), -1)
            cv2.imshow('icub gaze',icub_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
    def menu(self):
        self.manager.current = 'menuScreen'


class ObjectDetectionScreen(Screen):
    
    def stream_video(ipv4_address, tobiiglasses, project_id, participant_id, calibration_id, res, accuracy):
        print('Start Streaming')
        cap = cv2.VideoCapture("rtsp://%s:8554/live/scene" % ipv4_address)

        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        #save the video
        '''fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))'''
        
        
        # Filtering of gaze position to know when the gaze is mouving and when is fixed
        previous_x_pos = 0
        previous_y_pos = 0

        # Read until video is completed
        tobiiglasses.start_streaming()
        i=0
        delta = 400
        first_detection = False
        x_prev_pos = 0
        y_prev_pos = 0
            
        while(cap.isOpened()):
            
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            
            if ret == True:
                height, width = frame.shape[:2]
                data_gp  = tobiiglasses.get_data()['gp']
                
            if data_gp['ts'] > 0:
                starting_time = time.time()
                x_pos = int(data_gp['gp'][0]*width)
                y_pos = int(data_gp['gp'][1]*height)
                cv2.circle(frame,(x_pos,y_pos), 50, (0,0,255), 5) # Show where the user is watching
                                    
            # idea check velocity of the gaze acces with tobii api provide by gaze angle
            if data_gp['ts'] > 0:
                distance = math.sqrt((x_pos-x_prev_pos)**2+(x_pos-x_prev_pos)**2)
                x_prev_pos = x_pos
                y_prev_pos = y_pos
                dt = time.time() - starting_time
                velocity = int(distance/dt)
                if velocity < 1000:
                    # put inside the object detection
                    message =('Fixation gaze')
                    cv2.putText(frame, message, (700, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
                    
                    detection_img = frame
                    classes, scores, boxes = model.detect(detection_img, Conf_threshold, NMS_threshold)
                    
                    for (classid, score, box) in zip(classes, scores, boxes):
                        label = "%s : %f" % (class_name[classid[0]], score)
                        
                        if (box[0]<x_pos<box[2]+box[0])and (box[1]<y_pos<box[3]+box[1]):
                            object_fixed = str(label)
                            first_detection = True
                            
                            cv2.rectangle(detection_img, box, (0, 255, 255), 1)
                            cv2.putText(detection_img, label, (box[0], box[1]+15), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)
                            cv2.circle(detection_img,(x_pos,y_pos), 50, (0,0,255), 5)
                                                   
                            #saving the image
                            '''name_file = str(i)
                            cv2.imwrite('save_frame/' + 'image'+ str(i) + '.png', detection_img)'''
                            i += 1
                        
            # show last object fixed
            if first_detection == True:
                message =('last object fixed: '+ object_fixed)
                cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
            else:
                message =('No object fixed yet')
                cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
             
            #save frame per frame
            '''hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            out.write(hsv)'''
            
            # Display the resulting frame
            cv2.imshow('Tobii Pro Glasses 2',frame)
            
            # Press q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # When everything done, release the video capture object
                cap.release()
                out.release() 
                # Closes all the framesq
                cv2.destroyAllWindows()
                # Disconnect tobbiglasses
                tobiiglasses.stop_streaming()
                tobiiglasses.close()
                break

        print('still alive')
        time.sleep(3)
        '''# When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        # Disconnect tobbiglasses
        tobiiglasses.stop_streaming()
        tobiiglasses.close()'''
        
        
    def start(self):
        global ipv4_address, tobiiglasses, project_id, participant_id, calibration_id, res, accuracy
        ObjectDetectionScreen.stream_video(ipv4_address, tobiiglasses, project_id, participant_id, calibration_id, res, accuracy)
        
    def menu(self):
        self.manager.current = 'menuScreen'

if __name__ == '__main__':
    TobiiApp().run()
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                