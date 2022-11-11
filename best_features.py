import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class best_features:
    def __init__(self):
        pass
    
    def pre_process(df):
        df_total = df[['Features','Area in sq. ft.']]
        df_total.rename(columns= {'Area in sq. ft.':'Area','Features':'Room'},inplace=True)
        df_total['Features'] = np.where(df_total['Count'] == 2 ,
                                        "[" + df_total['Room'] + "," + df_total['Room'] + "]",
                    (np.where(df_total['Count'] == 3 ,"[" + df_total['Room'] + "," + df_total['Room'] + "," + df_total['Room'] + "]"                                
                                        ,df_total['Room'])))


        df_total.drop('Room',inplace= True,axis=1)
        df_1 = df_total.copy()
        df_1 = df_1.replace('[\([{})\]]','',regex=True)
        df_1['Features']=df_1['Features'].str.split(',')
        df_1['Area'] = df_1['Area'].astype(str)
        df_1['Area']=df_1['Area'].str.split(',')
        df_2 = df_1.set_index(['Count']).apply(pd.Series.explode).reset_index()
        df_2['Features']=df_2['Features'].str.split('/')
        df_3 = df_2.set_index(['Count','Area']).apply(pd.Series.explode).reset_index()
        #     df_3['Features'] = df_3['Features'].str.replace('Living','Living Room') 
        df_3['Features'] = np.where(df_3['Features'] == 'Living','Living Room',df_3['Features'])
        df_3['Col2'] = df_3.groupby('Features').cumcount()+1
        df_3['Col2'] = df_3['Features'] + '-' + df_3['Col2'].astype('str')
        df_3.drop(['Count','Features'], axis=1, inplace=True)
        df_3.rename(columns = {'Col2':'Features','Area':'Area in sq. ft.'}, inplace=True)
        return df_3

    def best_feature(df1,df2):

        df1 = df1[['Features','Area in sq. ft.']]
        df2 = df2[['Features','Area in sq. ft.']]
        df1['filename'] = 'floorplan_1'
        df2['filename'] = 'floorplan_2'
        df_total = pd.concat([df1,df2] ,axis=0)
        df_total.rename(columns= {'Area in sq. ft.':'Area'},inplace=True)
        df_2 = df_total.copy()


        # df_2.drop('Count', axis=1, inplace=True)
        df_3 = df_2[['Features','Area']]
        # df_3['Area in sq. ft.']=df_1['Area in sq. ft.'].astype(float)
        df_4 = df_3.groupby('Features')['Area'].max().reset_index()


        df_5 = pd.merge(df_4,df_2,how='left',on=['Features','Area'])
        # df_5.head()

        file_names =  df_5['filename'].unique()

        df_6 = df_5.groupby(['Features','Area'])['filename'].apply(list).reset_index()
        df_6.rename(columns= {'filename':'concat'},inplace=True)

        df_7 = df_5.groupby(['Features','Area'])['filename'].apply(lambda x: ','.join(x)).reset_index()
        df_7.rename(columns= {'filename':'concat_1'},inplace=True)

        df_8 = pd.merge(df_6,df_7,on=['Features','Area'])
        # df_8['Summary'] = np.where(len(df_8['concat']) == 1,
        #                            (df_8['Features'] + ' is better in ' + df_8['concat_1']),
        #                    (df_8['Features'] + ' is same in ' +  df_8['concat'].str[0] + ' and ' + df_8['concat'].str[1])
        #                    )

        df_8['Summary'] = np.where(df_8['concat_1'].str.len() > 11,
                           (df_8['Features'] + ' is same in ' +  df_8['concat'].str[0] + ' and ' + df_8['concat'].str[1]),
                                   (df_8['Features'] + ' is better in ' + df_8['concat_1'])
                           )

        new_df = pd.DataFrame(index = df_8.index,columns=file_names)
        final_output = pd.concat([df_8,new_df],axis=1)

        for names in file_names:
            final_output[names] = np.where(final_output['concat_1'].str.contains(names),1,0)

        final_output.rename(columns= {'Area':'Area in sq. ft.'},inplace=True)
        final_output.drop(['concat','concat_1'], axis=1, inplace=True)
        # final_output.to_csv("summary_output.csv", index=False)
    #         final_output.head(20)
        return final_output
        
    def comp(df):
        df_comp = df[['Features', 'floorplan_1','floorplan_2','Summary']]
        df_comp['floorplan_1'] = df_comp['floorplan_1'].apply(lambda x:'❌' if x == 0 else ('✔️' if x == 1 else ''))
        df_comp['floorplan_2'] = df_comp['floorplan_2'].apply(lambda x:'❌' if x == 0 else ('✔️' if x == 1 else ''))
        summary = ''
        for i in df['Summary']:
            if summary == '':
                summary = i.strip()
            else:
                summary = summary + ". " + i.strip()
        return summary, df_comp