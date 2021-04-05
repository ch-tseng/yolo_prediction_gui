# -*- coding: utf-8 -*-
from libMain import UI
from libMain import LABELING
from libMain import VIDEOPLAY
import PySimpleGUI as sg
import io
import os, sys
import glob
from yoloOpencv import opencvYOLO
import cv2
import numpy as np
import win32gui
import shutil

win_preview_size = (1024,768)
finished_path = r'D:\wait\road_finished'

#AI Model
'''
model_cfg = r'AI\crowd_human_v0\yolov3.cfg'
model_names = r'AI\crowd_human_v0\obj.names'
model_weights = r'AI\crowd_human_v0\yolov3_29000.weights'
model_size = (608, 608)
class_list = { 'head': '頭', 'body': '身體' }
'''
'''
model_cfg = r'AI\Medicial_mask\v1\yolov4.cfg'
model_names = r'AI\Medicial_mask\v1\obj.names'
model_weights = r'AI\Medicial_mask\v1\yolov4_80000.weights'
model_size = (608, 608)
'''
'''
model_cfg = r'AI\Medicial_mask\yolov3_v1\yolov3.cfg'
model_names = r'AI\Medicial_mask\yolov3_v1\obj.names'
model_weights = r'AI\Medicial_mask\yolov3_v1\yolov3_40000.weights'
model_size = (608, 608)
'''

model_cfg = r'AI\crowd_human_swim\yolov3.cfg'
model_names = r'AI\crowd_human_swim\obj.names'
model_weights = r'AI\crowd_human_swim\yolov3_16000.weights'
model_size = (608, 608)

class_list = { 'head': '頭', 'body': '身體' }
'''
model_cfg = r'AI\Medicial_mask\v1\yolov3.cfg'
model_names = r'AI\Medicial_mask\v1\obj.names'
model_weights = r'AI\Medicial_mask\v1\yolov3_40000.weights'
model_size = (608, 608)
class_list = { 'balaclava_ski_mask': '滑雪面罩', 'eyeglasses':'眼鏡', 'face_no_mask':'未戴口罩','face_other_covering':'其它物品遮蓋', \
'face_shield':'面罩', 'face_with_mask':'有戴口罩', 'face_with_mask_incorrect':'口罩沒戴好', 'gas_mask':'防毒面罩', 'goggles':'遮風面罩', \
'hair_net':'髮罩', 'hat':'帽子', 'helmet':'頭盔', 'hijab_niqab': '蓋頭頭巾', 'hood': '兜帽', 'mask_colorful':'花式口罩', 'mask_surgical':'醫用口罩', \
'other': '其它物品', 'scarf_bandana': '圍巾', 'sunglasses':'太陽眼鏡', 'turban':'頭巾' }
'''

#----------------------------------------------------------------------------------

finished_path = finished_path.replace('\\', '/')

def checkenv():
    if not os.path.exists(finished_path):
        os.makedirs(finished_path)

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def get_mode_classes():
    classes = []
    with open(model_names, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            classes.append(line)

    return classes

def rgb2hex(rgb):
    (r, g, b) = rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def load_model(model_path):
    model_cfg, model_names, model_weights, model_size = None, None, None, None
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if file_extension == '.cfg':
                model_cfg = file_path.replace('\\', '/')
            if file_extension == '.names':    
                model_names = file_path.replace('\\', '/')
            if file_extension == '.weights':
                model_weights = file_path.replace('\\', '/')

        model_size = int(values["-MODEL_SIZE-"][0].split('x')[0])

    return model_cfg, model_names, model_weights, model_size

def selected_file(values):
    filename_preview = values["-FILE_LIST-"][0]
    filename, file_extension = os.path.splitext(filename_preview)
    file_extension = file_extension.lower()
    path_preview_img = os.path.join(values["-FOLDER_PATH-"], values["-FILE_LIST-"][0])

    if file_extension in [".mp4",".mov", "mkv", ".mpg"]:
        type = 'video'
    else:
        type = 'image'

    return type, filename_preview, path_preview_img

#------------------------------------------------------------------------------------
checkenv()

winUI = UI(preview_size=win_preview_size)
window = winUI.create_window(win_title="AI Object detection", resizable=True)



# for image prediction
path_preview_img, filename_preview = '', ''  #for selected file


VIDEOFRAME, aiLABEL = None, None
last_model_cfg, last_model_names, last_model_weights, last_model_size ='', '', '', 0
while True:

    event, values = window.read() 


    if event == sg.WIN_CLOSED:  # if the X button clicked, just exit
        break

    elif event == '-MODEL_FOLDER-':
        winUI.refresh_listmodels(values["-MODEL_FOLDER-"])

    elif event == "-FOLDER_IMAGE-":
        path_img_dataset = values["-FOLDER_PATH-"]

    elif event == "-FOLDER_PATH-":  # A file was chosen from the listbox
        winUI.refresh_listfiles(values["-FOLDER_PATH-"])
    
    elif event == '-SCORE_INPUT-':
        try:
            print(float(values['-SCORE_INPUT-']))
        except:
            window['-SCORE_INPUT-'].update('0.25')

    elif event == '-NMS_INPUT-':
        try:
            print(float(values['-NMS_INPUT-']))
        except:
            window['-NMS_INPUT-'].update('0.55')

    elif event == "-FILE_LIST-":  # A file was chosen from the listbox
        if len(values["-FILE_LIST-"])>0:
            filetype, filename, file_fullpath = selected_file(values)

            if filetype == 'video':
                VIDEOFRAME = VIDEOPLAY(video_path=file_fullpath)
                if VIDEOFRAME.preview_img is None:
                    sg.Popup("{} 影片無法讀取.".format(filename))
                    continue

                else:
                    winUI.update_preview_img(img_path=VIDEOFRAME.preview_img, first=False)

            else:
                path_preview_img = file_fullpath
                winUI.update_preview_img(img_path=file_fullpath, first=True)
            #aiLABEL = LABELING(classColors, option_classes)

    elif event == "-AUTO_LABEL-":
        if not len(values["-FILE_LIST-"])>0:
            continue
            
        filetype, filename, file_fullpath = selected_file(values)

        if filetype == 'video':
            if VIDEOFRAME.preview_img is not None:
                grabbed = VIDEOFRAME.get_frame()
                while grabbed:
                    winUI.update_preview_img(img_path=VIDEOFRAME.preview_img, first=False)
                    grabbed = VIDEOFRAME.get_frame()

        else:

            if os.path.exists(path_preview_img):
                img_predict = cv2.imread(path_preview_img)
                try:
                    test = img_predict.shape
                except:
                    sg.Popup('該圖片檔名有非英文字元, 因此無法讀取 (圖片檔名請使用英文)')
                    continue

                if not len(values["-MODEL_LIST-"]) > 0:
                    sg.popup('請選擇一個模型.')

                else:
                    print('img_predict', img_predict.shape, 'score/nms', float(values['-SCORE_INPUT-']), float(values['-NMS_INPUT-']))
                    (model_cfg, model_names, model_weights, model_size) = load_model( os.path.join(values["-MODEL_FOLDER-"], values["-MODEL_LIST-"][0]) )
                    

                    if (last_model_cfg!=model_cfg or last_model_names!=model_names  or last_model_weights!=model_weights \
                            or last_model_size!=model_size) or last_model_cfg=='' :

                        yolo = opencvYOLO(imgsize=(model_size,model_size), \
                        objnames=model_names, \
                        weights=model_weights,\
                        cfg=model_cfg, score=float(values['-SCORE_INPUT-']), nms=float(values['-NMS_INPUT-']))

                        last_model_cfg, last_model_names, last_model_weights, last_model_size = model_cfg, model_names, model_weights, model_size

                    objects = get_mode_classes()
                    classColors = []
                    for i in range(0, len(objects)):
                        classColors.append(np.random.choice(range(256), size=3).tolist())

                    option_classes = []
                    for  i, cname in enumerate(objects):
                            option_classes.append('{}_{}'.format(i,cname,cname))


                    yolo.getObject(img_predict, score=float(values['-SCORE_INPUT-']), nms=float(values['-NMS_INPUT-']), \
                            labelWant=objects, drawBox=True, bold=2, textsize=1.2, bcolor=(255,255,255), tcolor=classColors)
                    print(yolo.labelNames)

                    cv2.imwrite('predicted.png', img_predict)
                    winUI.update_preview_img(img_path='predicted.png', first=True)
                    aiLABEL = LABELING(classColors, option_classes)
            '''


            


            objects = get_mode_classes()
            classColors = []
            for i in range(0, len(class_list)):
                classColors.append(np.random.choice(range(256), size=3).tolist())

            option_classes = []
            for  i, cname in enumerate(class_list):
                    option_classes.append('{}_{}_{}'.format(i,cname,cname+'/'+class_list[cname]))


            yolo.getObject(img_predict, score=float(values['-SCORE_INPUT-']), nms=float(values['-NMS_INPUT-']), \
                    labelWant=objects, drawBox=True, bold=2, textsize=1.2, bcolor=(255,255,255), tcolor=classColors)
            print(yolo.labelNames)
            #cv2.imwrite('predicted.png', aiLABEL.rectangle(img=img_predict, bboxes=yolo.bbox, color=(255,0,0)))
            cv2.imwrite('predicted.png', img_predict)
            winUI.update_preview_img(img_path='predicted.png', first=True)

            #for id, box in enumerate(yolo.bbox):
            #    print(' yolo box',  box)
            #    aiLABEL.add_rect(box, window["-img_preview-"], yolo.classIds[id], winUI.img_size, winUI.img_orgsize)
            '''
    elif event == '-EXPORT_FILE-':
        if os.path.exists(path_preview_img):
            winUI.save_graph_as_file('graph.png')
            #os.rename(path_preview_img , os.path.join(finished_path, filename_preview))
            base_name, ext_name = filename_preview.split('.')[0], filename_preview.split('.')[-1]
            #os.rename('graph.png' , os.path.join(finished_path, base_name+'_ans.'+ext_name))
            shutil.copyfile('graph.png' , os.path.join(finished_path, base_name+'_predicted.'+ext_name))
            winUI.refresh_listfiles(values["-FOLDER_PATH-"])

    elif event == '-img_preview-':
        # https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Graph_Drag_Rectangle.py
        x, y = values["-img_preview-"]
        print('x,y', x,y)
        if aiLABEL is not None: aiLABEL.start_drag(values, window["-img_preview-"])
        '''
        x, y = values["-img_preview-"]
        graph = window["-img_preview-"]
        if not dragging:
            start_point = (x, y)
            dragging = True
        else:
            end_point = (x, y)

        if prior_rect:
            graph.delete_figure(prior_rect)
        

        if None not in (start_point, end_point):
            prior_rect = graph.draw_rectangle(start_point, end_point, line_color='red')
        '''

    elif event.endswith('+UP'):  # The drawing has ended because mouse up
        if aiLABEL is not None: aiLABEL.end_drag(window["-img_preview-"], winUI.img_size)
        '''
        print(start_point, end_point)
        if start_point == end_point:
            continue

        
        
        class_event, class_values = sg.Window('Choose an class', [[sg.Text('Class name ->'), sg.Listbox(option_classes, size=(20, 6), \
            key='class_choose')],  [sg.Button('Ok'), sg.Button('Cancel')]]).read(close=True)

        #刪除rectangle
        if prior_rect:
            graph.delete_figure(prior_rect)

        #popup menu for class: https://stackoverflow.com/questions/62559454/how-do-i-make-a-pop-up-window-with-choices-in-python
        if class_event == 'Ok':
            selection = class_values["class_choose"][0].split('_')
            color_c = classColors[int(selection[0])]
            print(start_point, end_point)
            class_rect = graph.draw_rectangle(start_point, end_point, line_color=rgb2hex(color_c))
            graph.DrawText(selection[2], start_point, font=("Courier New", 12), color=rgb2hex(color_c), \
                text_location=sg.TEXT_LOCATION_TOP_LEFT)
            #sg.popup(f'You chose {class_values["class_choose"][0]}')
            img_bboxes.append([start_point[0], winUI.img_size[1]-start_point[1], abs(end_point[0]-start_point[0]), abs(end_point[1]-start_point[1])])
            img_rects.append(prior_rect)
            print('img size', winUI.img_size)
            print('img_bboxes', img_bboxes)
            print('img_rects', img_rects)

        
            
        start_point, end_point = None, None  # enable grabbing a new rect
        dragging = False
        '''
    else:
        print("unhandled event", event, values)            
