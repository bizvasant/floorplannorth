import cv2
import numpy as np
import pandas as pd
from quadrants import quadrants

class quadrants_area(quadrants):
    def __init__(self):
        pass
    
    def quad_area_info(image, filename):
        
        name_dict = {
            '1bhk_sample1':[1100,1100],
            '1bhk_sample2':[850,950],
            '2bhk_sample1':[1200,1200],
            '2bhk_sample2':[1300,1100],
            '3bhk_sample1':[1400,1300], # with lines
            '3bhk_sample2':[1600,1100]
        }

        print(name_dict)

        # org_image = cv2.imread(image)
        key = str(filename)

        if any(key in key for key in name_dict.keys()):
            print(key)
            image = cv2.resize(image, (name_dict[key][0], name_dict[key][1]))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(thresh, thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        box = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(box)
        mask = np.zeros(thresh.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts=[np.asarray(box)], color=(1))
        M = cv2.moments(mask, binaryImage=True)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])


        if key == '3bhk_sample2' or key == '1bhk_sample2':
            print('find_rooms_with_lines')
            colored_house, room_mapping = quadrants.find_rooms_wo_lines(image)
        else :
            print('find_rooms_wo_lines')
            colored_house, room_mapping = quadrants.find_rooms_with_lines(image)

        colored_house, room_mapping = quadrants.find_rooms_wo_lines(image)
        
        im_rgb = cv2.cvtColor(colored_house, cv2.COLOR_BGR2RGB)
        room_color = room_mapping['color']

        q2 = im_rgb[y:cy, x:cx]
        q3 = im_rgb[cy:y+(h), x:cx]
        q4 = im_rgb[cy:y+(h), cx:x+(w)]
        q1 = im_rgb[y:cy, cx:x+(w)]

        images = {"Quadrant2": q2,
                "Quadrant1": q1,
                "Quadrant3": q3,
                "Quadrant4": q4}

        keys = list(images.keys())
        values = list(images.values())

        j=0
        q_list = ['Q2','Q1','Q3','Q4']
        data_quad = pd.DataFrame()
        for q in values:
            val = q_list[j]
            j+=1
            for i in range(len(room_color)):
                abc = room_color[i].tolist()
                color = abc[::-1]
                result = 100*((np.count_nonzero(np.all(q==color,axis=2)))/(q.shape[0]*q.shape[1]))
                if result > 1:
                    df_temp = pd.DataFrame({'Quadrant': [val] ,'Color' : [i],'color_percent' : [result] })
                    data_quad = pd.concat([data_quad,df_temp],axis=0)
                    data_quad.reset_index(inplace = True, drop = True)
        # abc = data_quad.copy()

        print('data_quad_',data_quad.shape)

        from ocr import ocr
        df2 = ocr.ocr_info_quard_distro(image)

        # df2 = pd.read_csv("data_img_2.csv")

        # df2=df2[['Features','Area in sq. ft.']].copy()

        # df2.Features=df2.Features.str.lower().replace(['living room-1','living-1','dinning-1','dinning room-1'],'Living/Dinning room-1')\
        #                             .replace(['kitchen-1','dry bal-1','dry-1'],'Kitchen/Dry Bal-1')
        # df2.Features=df2.Features.str.title()
        # df2=df2.groupby('Features').sum('Area in sq. ft.').reset_index()

        print('df2_2_',df2.shape)
        df2 = df2.sort_values(['Area in sq. ft.'],ascending= False)
        df2.reset_index(inplace = True, drop = True)      
        print('df2_',df2.shape)

        rooms = pd.DataFrame()
        room_area =[]
        for i in range(len(room_mapping['room'])):
            if i < len(room_mapping['room']):
                t = room_mapping['room'][i]
                area = room_mapping['area'][i]
                df_temp = pd.DataFrame({'Color' : [i],'Area' : [area] })
                rooms = pd.concat([rooms,df_temp],axis=0)
        rooms = rooms.sort_values(['Area'],ascending= False)
        rooms.reset_index(inplace = True, drop = True)      
        #rooms = rooms[0:len(df2)]

        print('rooms_',rooms.shape)

        features_dict = dict(zip(rooms['Color'], df2['Features']))
        area_dict_1 = dict(zip(rooms['Color'], df2['Area in sq. ft.']))

        data_quad['Features'] = data_quad['Color'].map(features_dict)
        data_quad['Actual_area'] = data_quad['Color'].map(area_dict_1)

        abc = data_quad.groupby('Color')['color_percent'].sum().reset_index().rename(columns = {'color_percent':'perc_total'})
        final = pd.merge(data_quad,abc,on=['Color'])
        final['Feature_Quadrant_area'] = round((final['color_percent']/ final['perc_total']) * final['Actual_area'],2)
        final = final[final['Features'].notna()]
        final.drop(['Color','color_percent','perc_total'], axis=1, inplace=True)
        final = final.sort_values(['Quadrant'],ascending= True)
        final.head(20)
        return final 
        