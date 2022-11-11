import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import easyocr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import string
import difflib 


class ocr:
    def __init__(self):
        pass

    def ocr_info(img):
        reader = easyocr.Reader(['en'])
        # if img == "1bhk_sample1.png":
        #     result = reader.readtext(img,paragraph="False")
        # else:
            # result = reader.readtext(img,paragraph="False", add_margin=0.2,slope_ths=0.5)
        result = reader.readtext(img,paragraph="False")
        ALPHA = string.ascii_letters
        master_features = ["utility","toilet","passage","dining","kitchen", "bathroom", "bedroom", "hallway", "hall", "living room", 
                        "balcony","master bedroom","m bedroom","living","stair","puja","pooja","sit out","terrace","clearance",
                        "dressing room","dressing","store","lobby","open area","wash area","parking space","corridor","foyer",
                        "portico","dry bal","wc"]
        final_list=[]
        for i in result:
            if i[-1].startswith(tuple(ALPHA)):
                temp = i[-1].replace(" m'", "").replace(' m"', '').replace(" m", "").replace(" / ", "-").replace(",", ".")
                lst = temp.split()
                if len(lst) > 1 :
                    if str(lst[1]).isalpha():
                        lst[0] = lst[0] + " " + lst[1]
                        lst.pop(1)
                    if len(lst) >= 2: final_list.append(lst)

        # res = [i for n, i in enumerate(final_list) if i not in final_list[:n]]
        res=final_list
        print(res)
        temp=[]
        for i in res:
            if len(i)==2:temp.append(i)
            else:
                t=[]
                t.append(i[0])
                t1="".join(i[1:])
                t.append(t1)
                temp.append(t)  
        res=temp
        features,area=[],[]
        for i in res:
            if difflib.get_close_matches(i[0].lower(), master_features):
                features.append(i[0])
                area.append(i[1])
        df=pd.DataFrame()
        df['features']=features
        df['area']=area
        df['area']=df['area'].replace('xX',"""'X""",regex=True).replace('[oO]',"0",regex=True) \
                            .replace('S',"5",regex=True).replace('[iI]',"1",regex=True).replace('[: ]'," ",regex=True)
    #     d=df.copy()
        #this code to handle if measurment is given in two scale and one measure is inside ()
        #code started from here
        for i in range(len(df)):
            if df.loc[i,'area'].find('(')>0:
                l=df.loc[i,'area'].split('(')
                df.loc[i,'area']=l[1].replace(')',' ')
            if df.loc[i,'area'].find('(')<0 and df.loc[i,'area'].find(')')>0:
                pos=str(df.loc[i,'area']).lower().rfind('x')
                spos=pos-4
        #         print(spos,df.iloc[i]['area'][spos:])
                df.loc[i,'area']=df.loc[i,'area'][spos:].replace(')',' ')
    #     df0=df.copy()        
        for i in range(len(df)):
            if df.loc[i,'area'].lower().find('m')>0:
                l=df.loc[i,'area'].lower().split('m')
                df.loc[i,'area']=l[1].replace(')',' ')
            if df.loc[i,'area'].rstrip(' ').lower()[-1]=="""'""": 
                df.loc[i,'area']=df.loc[i,'area'].rstrip(' ')+'''0"'''
            if df.loc[i,'area'].rstrip(' ').lower()[-1]=='''"''': pass
            elif df.loc[i,'area'].rstrip(' ').lower()[-1]!='''"''' or df.loc[i,'area'].rstrip(' ').lower()[-1]!="""'""":
                df.loc[i,'area']=df.loc[i,'area'].rstrip(' ')+"""'0"""+'''"'''
            else:pass
    #     df_1=df.copy()
        for i in range(len(df)):
            if df.loc[i,'area'].lower().find("""'x""")>0:
                df.loc[i,'area']=df.loc[i,'area'].lower().replace("""'x""","""'0"""+'''"x''')
            if df.loc[i,'area'].lower().find("""'"""+'''"x''')>0: 
                df.loc[i,'area']=df.loc[i,'area'].lower().replace("""'"""+'''"x''',"""'0"""+'''"x''')
            xpos=df.loc[i,'area'].lower().find("x")
            ft_pos=df.loc[i,'area'].lower().find("""'""")
            r_in_pos=df.loc[i,'area'].lower().rfind('''"''')
            in_pos=df.loc[i,'area'].lower().find('''"''')
        #     print(ft_pos,in_pos)
            if in_pos-ft_pos<0:
        #         print('inside it',df.loc[i,'area'],df.loc[i,'area'][:in_pos-1]+"""'"""+df.loc[i,'area'][in_pos-1:])
                df.loc[i,'area']=df.loc[i,'area'][:in_pos-1]+"""'"""+df.loc[i,'area'][in_pos-1:]
            elif xpos-ft_pos<0 or xpos-in_pos<0:
                df.loc[i,'area']=df.loc[i,'area'].lower().replace("x","""'0"""+'''"x''')
            elif r_in_pos-xpos<=2:df.loc[i,'area']=df.loc[i,'area'].rstrip(' ')[:-1]+"""'0"""+'''"'''
            else:pass
        # df_updated=df.copy()           
        for t in range(len(df)):
            dim=[]
            t1=df.iloc[t].area.lower().split('x')
        #     print(df.iloc[t].area.lower())
            ft_cnt=df.iloc[t].area.lower().count("""'""")
            in_cnt=df.iloc[t].area.lower().count('''"''')
            if ft_cnt==2 and in_cnt==2:pass
            else:
                for i in t1:
                    temp=i[:-1].replace("""'""",'').replace('''"''','').strip('')
            #         print(temp)
                    if int(temp)>=12:
                        ft,inch=temp[:1],temp[1:]
                        if str(inch)[0]==0:
                            ft=ft+"0"
                            inch=inch[1:]
            #             print(ft,inch)
                        if int(inch)==0 or int(inch)>12:
            #                 print('inside inch test')
                            if len(inch)>=2:ft,inch=temp[:2],temp[2:]
                            else:ft,inch=temp[:1],temp[1:]
            #             print(ft+"""'"""+inch+'''"''')
                        dim.append(ft+"""'"""+inch+'''"''')
                df.iloc[t].area='x'.join(dim)
        final_df=df.copy()
        final_df['col1'] = final_df.groupby('features').cumcount()+1
        final_df['Col2'] = final_df['features'] + '-' + final_df['col1'].astype('str')
        final_df.drop(['features','col1'], axis=1, inplace=True)
        final_df.rename(columns = {'Col2':'Features','area':'Dimension'}, inplace=True)
        #code end here
        # df = pd.DataFrame(mapp, columns=mapp.keys())
        area_lst = {}
        for i in df["area"]:
            if len(i) > 3:
        #         print(i)
        #         text = i.lower().replace("m", " ").replace("o", "0").replace('O','0').replace("*", """'""").split("x")
                text=i.lower().split("x")
                length, breath = text[0], text[1]
                length = length.replace('''"''', '').replace("""''""", """'""").split("""'""")
                breath = breath.replace('''"''', '').replace("""''""", """'""").split("""'""")
                print(length,breath)
                if len(length)>1: total_length = float(length[0]) + (float(length[1])/12)
        #         else:total_length = round(float(length[-1][:-1]) + (float(length[-1][-1:])/12), 2)
                if len(breath)>1: total_breath = float(breath[0]) + (float(breath[1])/12)
        #         else:total_breath = round(float(breath[-1][:-1]) + (float(breath[-1][-1:])/12), 2)
        #         total_breath = round(float(breath[0]) + (float(breath[1])/12), 2)
                area = round((total_length * total_breath), 2)
                area_lst[i] = area
        df = df.replace({"area": area_lst})
        df=df.sort_values(by=['features','area'],ascending=[True,False])
        df['col1'] = df.groupby('features').cumcount()+1
        df['Col2'] = df['features'] + '-' + df['col1'].astype('str')
        df.drop(['features','col1'], axis=1, inplace=True)
        df.rename(columns = {'Col2':'Features','area':'Area in sq. ft.'}, inplace=True)
        df1=df.merge(final_df,on='Features',how='left')
        # df1.sort_values(by='Features')

        # print(df1[['']])
        # df1 = df["features"].value_counts().reset_index()
        # df1.columns = ["Features", "Count"]
        # X = {}
        # for i in df["features"]:
        #     x = df[df["features"] == i]["area"].to_list()
        #     X[i] = x
        # df1["Area"] = 0
        # df1.loc[:,'Area']=df1.Features.map(X)
        return df1[['Features','Dimension','Area in sq. ft.']]

    def ocr_info_quard_distro(img):
        reader = easyocr.Reader(['en'])
        # if img == "1bhk_sample1.png":
        #     result = reader.readtext(img,paragraph="False")
        # else:
        #     result = reader.readtext(img,paragraph="False", add_margin=0.2,slope_ths=0.5)
        result = reader.readtext(img,paragraph="False")
        ALPHA = string.ascii_letters
        master_features = ["utility","toilet","passage","dining","kitchen", "bathroom", "bedroom", "hallway", "hall", "living room", 
                        "balcony","master bedroom","m bedroom","living","stair","puja","pooja","sit out","terrace","clearance",
                        "dressing room","dressing","store","lobby","open area","wash area","parking space","corridor","foyer",
                        "portico","dry bal","wc"]
        final_list=[]
        for i in result:
            if i[-1].startswith(tuple(ALPHA)):
                temp = i[-1].replace(" m'", "").replace(' m"', '').replace(" m", "").replace(" / ", "-").replace(",", ".")
                lst = temp.split()
                if len(lst) > 1 :
                    if str(lst[1]).isalpha():
                        lst[0] = lst[0] + " " + lst[1]
                        lst.pop(1)
                    if len(lst) >= 2: final_list.append(lst)

        # res = [i for n, i in enumerate(final_list) if i not in final_list[:n]]
        res=final_list
        temp=[]
        for i in res:
            if len(i)==2:temp.append(i)
            else:
                t=[]
                t.append(i[0])
                t1="".join(i[1:])
                t.append(t1)
                temp.append(t)  
        res=temp
        features,area=[],[]
        for i in res:
            if difflib.get_close_matches(i[0].lower(), master_features):
                features.append(i[0])
                area.append(i[1])
        df=pd.DataFrame()
        df['features']=features
        df['area']=area
        df['area']=df['area'].replace('xX',"""'X""",regex=True).replace('[oO]',"0",regex=True) \
                            .replace('S',"5",regex=True).replace('[iI]',"1",regex=True).replace('[: ]'," ",regex=True)
        #     d=df.copy()
        #this code to handle if measurment is given in two scale and one measure is inside ()
        #code started from here
        for i in range(len(df)):
            if df.loc[i,'area'].find('(')>0:
                l=df.loc[i,'area'].split('(')
                df.loc[i,'area']=l[1].replace(')',' ')
            if df.loc[i,'area'].find('(')<0 and df.loc[i,'area'].find(')')>0:
                pos=str(df.loc[i,'area']).lower().rfind('x')
                spos=pos-4
        #         print(spos,df.iloc[i]['area'][spos:])
                df.loc[i,'area']=df.loc[i,'area'][spos:].replace(')',' ')
        #     df0=df.copy()        
        for i in range(len(df)):
            if df.loc[i,'area'].lower().find('m')>0:
                l=df.loc[i,'area'].lower().split('m')
                df.loc[i,'area']=l[1].replace(')',' ')
            if df.loc[i,'area'].rstrip(' ').lower()[-1]=="""'""": 
                df.loc[i,'area']=df.loc[i,'area'].rstrip(' ')+'''0"'''
            if df.loc[i,'area'].rstrip(' ').lower()[-1]=='''"''': pass
            elif df.loc[i,'area'].rstrip(' ').lower()[-1]!='''"''' or df.loc[i,'area'].rstrip(' ').lower()[-1]!="""'""":
                df.loc[i,'area']=df.loc[i,'area'].rstrip(' ')+"""'0"""+'''"'''
            else:pass
        #     df_1=df.copy()
        for i in range(len(df)):
            if df.loc[i,'area'].lower().find("""'x""")>0:
                df.loc[i,'area']=df.loc[i,'area'].lower().replace("""'x""","""'0"""+'''"x''')
            if df.loc[i,'area'].lower().find("""'"""+'''"x''')>0: 
                df.loc[i,'area']=df.loc[i,'area'].lower().replace("""'"""+'''"x''',"""'0"""+'''"x''')
            xpos=df.loc[i,'area'].lower().find("x")
            ft_pos=df.loc[i,'area'].lower().find("""'""")
            r_in_pos=df.loc[i,'area'].lower().rfind('''"''')
            in_pos=df.loc[i,'area'].lower().find('''"''')
        #     print(ft_pos,in_pos)
            if in_pos-ft_pos<0:
        #         print('inside it',df.loc[i,'area'],df.loc[i,'area'][:in_pos-1]+"""'"""+df.loc[i,'area'][in_pos-1:])
                df.loc[i,'area']=df.loc[i,'area'][:in_pos-1]+"""'"""+df.loc[i,'area'][in_pos-1:]
            elif xpos-ft_pos<0 or xpos-in_pos<0:
                df.loc[i,'area']=df.loc[i,'area'].lower().replace("x","""'0"""+'''"x''')
            elif r_in_pos-xpos<=2:df.loc[i,'area']=df.loc[i,'area'].rstrip(' ')[:-1]+"""'0"""+'''"'''
            else:pass
        #     df_updated=df.copy()           
        for t in range(len(df)):
            dim=[]
            t1=df.iloc[t].area.lower().split('x')
        #     print(df.iloc[t].area.lower())
            ft_cnt=df.iloc[t].area.lower().count("""'""")
            in_cnt=df.iloc[t].area.lower().count('''"''')
            if ft_cnt==2 and in_cnt==2:pass
            else:
                for i in t1:
                    temp=i[:-1].replace("""'""",'').replace('''"''','').strip('')
            #         print(temp)
                    if int(temp)>=12:
                        ft,inch=temp[:1],temp[1:]
                        if str(inch)[0]==0:
                            ft=ft+"0"
                            inch=inch[1:]
            #             print(ft,inch)
                        if int(inch)==0 or int(inch)>12:
            #                 print('inside inch test')
                            if len(inch)>=2:ft,inch=temp[:2],temp[2:]
                            else:ft,inch=temp[:1],temp[1:]
            #             print(ft+"""'"""+inch+'''"''')
                        dim.append(ft+"""'"""+inch+'''"''')
                df.iloc[t].area='x'.join(dim)
        final_df=df.copy()
        final_df['col1'] = final_df.groupby('features').cumcount()+1
        final_df['Col2'] = final_df['features'] + '-' + final_df['col1'].astype('str')
        final_df.drop(['features','col1'], axis=1, inplace=True)
        final_df.rename(columns = {'Col2':'Features','area':'Dimension'}, inplace=True)
        #code end here
        # df = pd.DataFrame(mapp, columns=mapp.keys())
        area_lst = {}
        for i in df["area"]:
            if len(i) > 3:
        #         print(i)
        #         text = i.lower().replace("m", " ").replace("o", "0").replace('O','0').replace("*", """'""").split("x")
                text=i.lower().split("x")
                length, breath = text[0], text[1]
                length = length.replace('''"''', '').replace("""''""", """'""").split("""'""")
                breath = breath.replace('''"''', '').replace("""''""", """'""").split("""'""")
        #         print(length,breath)
                if len(length)>1: total_length = float(length[0]) + (float(length[1])/12)
        #         else:total_length = round(float(length[-1][:-1]) + (float(length[-1][-1:])/12), 2)
                if len(breath)>1: total_breath = float(breath[0]) + (float(breath[1])/12)
        #         else:total_breath = round(float(breath[-1][:-1]) + (float(breath[-1][-1:])/12), 2)
        #         total_breath = round(float(breath[0]) + (float(breath[1])/12), 2)
                area = round((total_length * total_breath), 2)
                area_lst[i] = area
        df = df.replace({"area": area_lst})
        df['col1'] = df.groupby('features').cumcount()+1
        df['Col2'] = df['features'] + '-' + df['col1'].astype('str')
        df.drop(['features','col1'], axis=1, inplace=True)
        df.rename(columns = {'Col2':'Features','area':'Area in sq. ft.'}, inplace=True)
        df1=df.merge(final_df,on='Features',how='left')
        df1.sort_values(by='Features')

        temp=df1.copy()
        temp=temp[['Features','Area in sq. ft.']].copy()

        temp.Features=temp.Features.str.lower().replace(['living room-1','living-1','dinning-1','dinning room-1'],'Living/Dinning room-1')\
                                    .replace(['kitchen-1','dry bal-1','dry-1'],'Kitchen/Dry Bal-1')
        temp.Features=temp.Features.str.title()
        temp=temp.groupby('Features').sum('Area in sq. ft.').reset_index()


        return temp