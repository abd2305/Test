import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.markdown("<span style=“background-color:#111111”>",unsafe_allow_html=True)

header  = st.container()
dataset = st.container()
dataset_summary = st.container()

@st.cache(allow_output_mutation=True)

def Calcul_NewData(df_ini):
    df=df_ini
    df.reset_index(inplace=True,drop = True)
    df['SPEED_'] = df['SPEED']
    df['LATITUDE_'] = df['LATITUDE']
    df['LONGITUDE_'] = df['LONGITUDE']
    df['LINIAR_X_ACCEL'] = df['1_LINIAR_X_ACCEL']
    df['LINIAR_Y_ACCEL'] = df['1_LINIAR_Y_ACCEL']
    df['LINIAR_Z_ACCEL'] = df['1_LINIAR_Z_ACCEL']
    
    if '1_GRAVITY_X_VECT' in (list(df.columns)):
        df['X_GRAV'] = df['1_GRAVITY_X_VECT']
        df['Y_GRAV'] = df['1_GRAVITY_Y_VECT']
        df['Z_GRAV'] = df['1_GRAVITY_Z_VECT']
    
    df['X_GIRO'] = df['1_X_GIRO']
    df['Y_GIRO'] = df['1_Y_GIRO']
    df['Z_GIRO'] = df['1_Z_GIRO']

    df['X_EULER'] = df['1_X_EULER']
    df['Y_EULER'] = df['1_Y_EULER']
    df['Z_EULER'] = df['1_Z_EULER']

    df['VERT_VELOCITY'] = df['3_VERT_VELOCITY']
    if 'VBAT' in (list(df.columns)):
        df['VBAT'] = df['VBAT']
    df['DATE_'] = df['DATE']+' '+df['TIME']
    df["DATE_"]=pd.to_datetime(df["DATE_"])
    df = df.sort_values(by="DATE_", key=pd.to_datetime)
    
    time_ms = []
    for i in range(0,len(df)):
        time_ms.append((df["DATE_"].iloc[i]-df["DATE_"].iloc[0]).total_seconds())

    df_time = pd.DataFrame(data = time_ms,columns=['Time_s'])
    df_final = pd.concat([df,df_time], axis=1)
    
    df_final['Var_X_Accel'] = 0
    df_final['Var_Y_Accel'] = 0
    df_final['Var_Z_Accel'] = 0
    for i in range(1,len(df_final)):
       df_final['Var_X_Accel'].iloc[i] = ((df_final['LINIAR_X_ACCEL'].iloc[i] - df_final['LINIAR_X_ACCEL'].iloc[i-1]))
       df_final['Var_Y_Accel'].iloc[i] = ((df_final['LINIAR_Y_ACCEL'].iloc[i] - df_final['LINIAR_Y_ACCEL'].iloc[i-1]))
       df_final['Var_Z_Accel'].iloc[i] = ((df_final['LINIAR_Z_ACCEL'].iloc[i] - df_final['LINIAR_Z_ACCEL'].iloc[i-1]))
    
    df_final['Var_X_Giro'] = 0
    df_final['Var_Y_Giro'] = 0
    df_final['Var_Z_Giro'] = 0
    for i in range(1,len(df_final)):
       df_final['Var_X_Giro'].iloc[i] = (df_final['X_GIRO'].iloc[i] - df_final['X_GIRO'].iloc[i-1])
       df_final['Var_Y_Giro'].iloc[i] = (df_final['Y_GIRO'].iloc[i] - df_final['Y_GIRO'].iloc[i-1])
       df_final['Var_Z_Giro'].iloc[i] = (df_final['Z_GIRO'].iloc[i] - df_final['Z_GIRO'].iloc[i-1])
    
    
    ######################################################################################################################################################
    ##########################################################         Crash Detection        ############################################################
    ######################################################################################################################################################  
    #Recuperation des données calculées:

    #df_final['X_GIRO_Mean']= df_final['X_GIRO_MOAV']
    #df_final['Y_GIRO_Mean']= df_final['Y_GIRO_MOAV']
    #df_final['Z_GIRO_Mean']= df_final['Z_GIRO_MOAV']
    #df_final['Var_X_Giro']= df_final['X_GIRO_VAR']
    #df_final['Var_Y_Giro']= df_final['Y_GIRO_VAR']
    #df_final['Var_Z_Giro']= df_final['Z_GIRO_VAR']
    #df_final['LINIAR_X_ACCEL_Mean']= df_final['X_LIN_ACC_MOAV']
    #df_final['LINIAR_Y_ACCEL_Mean']= df_final['Y_LIN_ACC_MOAV']
    #df_final['LINIAR_Z_ACCEL_Mean']= df_final['Z_LIN_ACC_MOAV']
    #df_final['Var_X_Accel']= df_final['X_LIN_ACC_VAR']
    #df_final['Var_Y_Accel']= df_final['Y_LIN_ACC_VAR']
    #df_final['Var_Z_Accel']= df_final['Z_LIN_ACC_VAR']
    #
    #df_final['compteur']=df_final['VINTERRUPTS']*100
    df_final['DETECT'] = 100
    df_final['compteur'] = 0
    df_final['X_GIRO_Mean']= abs(df_final['X_GIRO']).rolling(3).mean()
    df_final['Y_GIRO_Mean']= abs(df_final['Y_GIRO']).rolling(3).mean()
    df_final['Z_GIRO_Mean']= abs(df_final['Z_GIRO']).rolling(3).mean()
    df_final['LINIAR_X_ACCEL_Mean']= abs(df_final['LINIAR_X_ACCEL']).rolling(3).mean()
    df_final['LINIAR_Y_ACCEL_Mean']= abs(df_final['LINIAR_Y_ACCEL']).rolling(3).mean()
    df_final['LINIAR_Z_ACCEL_Mean']= abs(df_final['LINIAR_Z_ACCEL']).rolling(3).mean()
    df_final['Accel_Norm']=np.sqrt((df_final['LINIAR_X_ACCEL']**2)+(df_final['LINIAR_Y_ACCEL']**2)+(df_final['LINIAR_Z_ACCEL']**2))
    df_final['Giro_Norm']=np.sqrt((df_final['X_GIRO']**2)+(df_final['Y_GIRO']**2)+(df_final['Z_GIRO']**2))
    df_final['Impact_G'] = df_final['Accel_Norm']

    for i in range (0,len(df)):
        compteur = 0
        gyro = 0
        gyrov = 0
        accelv = 0
        if (df_final['SPEED_'].iloc[i] >= 20 or np.isnan(df_final['SPEED_'].iloc[i])):

            if ((abs(df_final['X_GIRO_Mean'].iloc[i])>300) or (abs(df_final['Y_GIRO_Mean'].iloc[i])>400) or (abs(df_final['Z_GIRO_Mean'].iloc[i])>400)):
                compteur = compteur+1
                gyro = gyro+1

            if ((abs(df_final['Var_X_Giro'].iloc[i])>300) or (abs(df_final['Var_Y_Giro'].iloc[i])>500) or (abs(df_final['Var_Z_Giro'].iloc[i])>600)):
                compteur = compteur+1
                gyrov = gyrov+1

            if ((abs(df_final['LINIAR_X_ACCEL_Mean'].iloc[i])>40) or (abs(df_final['LINIAR_Y_ACCEL_Mean'].iloc[i])>40) or (abs(df_final['LINIAR_Z_ACCEL_Mean'].iloc[i])>40)):
                compteur = compteur+1

            if ((abs(df_final['Var_X_Accel'].iloc[i])>60) or (abs(df_final['Var_Y_Accel'].iloc[i])>60) or (abs(df_final['Var_Z_Accel'].iloc[i])>60)):
                compteur = compteur+1
                accelv = accelv+1

            if ((compteur >= 3) or ((gyro+accelv)>=2) or (gyro+gyrov >= 2)):
                df_final['compteur']=compteur*100

                break
            else:
                df_final['DETECT'].iloc[i]=0
        else:
             df_final['DETECT'].iloc[i]=0
            

    ######################################################################################################################################################
    ######################################################              Detection  End              ######################################################
    ######################################################################################################################################################

    return (df_final)
@st.cache(allow_output_mutation=True)
def uploaded_file(url_file):
    data = pd.read_table(url_file, sep=';' ,decimal = '.')
    return(data)



def add_map_(data,var,nb_map,map_params):
    points = []
    range_inf=data[data['Time_s']>=nb_map[0]]
    range_sup = range_inf[range_inf['Time_s']<=nb_map[1]]
    m = folium.Map(location = [data['LATITUDE_'].iloc[0],data['LONGITUDE_'].iloc[0]], zoom_start=15,tiles="OpenStreetMap")

    for i in range(0,len(range_sup)):
        points.append([range_sup['LATITUDE_'].iloc[i], range_sup['LONGITUDE_'].iloc[i]])
        tooltip = (' / '.join(map(str,(np.round(list(range_sup[map_params].iloc[i]),2)))))
        folium.Circle(
            [range_sup['LATITUDE_'].iloc[i], range_sup['LONGITUDE_'].iloc[i]], tooltip=tooltip, radius = 2
            ).add_to(m)
    folium.vector_layers.PolyLine(points,weight=2.5,opacity=1).add_to(m)
    folium.TileLayer('Stamen Toner').add_to(m)
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.TileLayer('cartodbdark_matter').add_to(m)
    folium.LayerControl().add_to(m)
    return(m)

def detect_crash(x):
    dta = Calcul_NewData(uploaded_file(x))
    detection = 'No'
    if (100 in dta['DETECT'].unique()):
        detection = 'Yes'
    return(detection)

def sort_by_name(file):
    return file.name

def main():

    st.sidebar.markdown("# Parameters")
    analyse_type = st.sidebar.radio('Choose your analysis type !',['Files analysis','Summary of crash detection '])
    if (analyse_type == 'Files analysis'):
        with header:
    
            st.sidebar.markdown("# Parameters")
            st.sidebar.info("Correlation Analysis")
            hmap = st.sidebar.checkbox('Heat Map')
            st.sidebar.info("Descriptive statistics")
            
            n_estimators = st.sidebar.selectbox('VARIABLE CHOICE',['SPEED_','LINIAR_X_ACCEL','LINIAR_Y_ACCEL','LINIAR_Z_ACCEL','LINIAR_X_ACCEL_Mean',
                'LINIAR_Y_ACCEL_Mean','LINIAR_Z_ACCEL_Mean','X_GIRO','Y_GIRO','Z_GIRO','X_GIRO_Mean','Y_GIRO_Mean','Z_GIRO_Mean','X_EULER','Y_EULER',
                'Z_EULER','X_GRAV','Y_GRAV','Z_GRAV','Impact_G','VBAT','VERT_VELOCITY','Accel_Norm','Giro_Norm','Var_X_Accel','Var_Z_Giro','Var_Z_Accel','Var_Y_Accel','Var_X_Giro','Var_Y_Giro'],index=0)
            
            with st.sidebar.expander("Graph line parameters"):
                maxx = st.checkbox("Maximum value")
                minn = st.checkbox("Minimum value")
                meann = st.checkbox("Mean value")
                stdd = st.checkbox("STD value")
    
            st.sidebar.info("More analysis")
            add_map = st.sidebar.checkbox('Add a Map')
    
            uploaded_files  = st.file_uploader("Choose a CSV file",type ='CSV',accept_multiple_files=True)
            list_uploaded_file=[]
            list_uploaded_file_detect=[]
            uploaded_files = sorted(uploaded_files, key=sort_by_name)
            if (uploaded_files != []):
                for i in range (0,len(uploaded_files)):
                    list_uploaded_file.append(uploaded_files[i].name)
                
                data_uploaded = st.sidebar.selectbox('CHOOSE YOUR FILE',list_uploaded_file,index=0)
                data = Calcul_NewData(uploaded_file(uploaded_files[list_uploaded_file.index(data_uploaded)]))
                
    
        with dataset:
    
            if (uploaded_files != []):
                df = data
                df.reset_index(inplace=True, drop=True)
                data = df
                st.subheader('Variable analysis')

                if (100 in data['DETECT'].unique()):
                    st.error('Crash detected',icon="🚨")
                else:
                    st.success("No crash detected",icon="✅")

                st.write(data.head(20))
                nb_data = st.slider('Data sample', value=[2000,5000], min_value=0, max_value=int(data[str(n_estimators)].count()),step =50)
                var = list(data.columns)
                #########################################################################
                #######################          HEAT MAP           #####################
                #########################################################################
                if hmap:
                    st.write('Correlation : ')
                    hmap_params = st.multiselect("Select parameters to include on heatmap", 
                        options=var, 
                        default=[p for p in var if "ang" in p])
                    
                    cor=data[hmap_params][nb_data[0]:nb_data[1]]
                    
                    hmap_fig = px.imshow(cor.corr(),
                        aspect ='auto',color_continuous_scale='RdBu_r',width=1200, height=600)
                    st.write(hmap_fig)
                #########################################################################
                #######################        END HEAT MAP         #####################
                #########################################################################
        
            
                st.subheader("Data Summary of : " + n_estimators)
                    
                st.write('Measures of central tendency and variability')
                
                col = st.columns(4)
                col[0].metric(label="Maximum value", value=np.round(data[str(n_estimators)].max(),2))
                col[1].metric(label="Mean value", value=np.round(data[str(n_estimators)].mean(),2))
                col[2].metric(label="STD value", value=np.round(data[str(n_estimators)].std(),2))
                col[3].metric(label="Minimum value", value=np.round(data[str(n_estimators)].min(),2))
                st.write('Number of line data:',nb_data)
                if (nb_data[1]!=nb_data[0]) : 
                    
                    fig = px.line(data[nb_data[0]:nb_data[1]], x="Time_s", y=str(n_estimators),width=1200, height=600)
                    st.write(fig)
                    
                    fig = plt.figure(figsize = (12,5))
                    if maxx : 
                        plt.axhline(y=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].max(),2),color='black',linestyle='--')
                    if minn :
                        plt.axhline(y=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].min(),2),color='black',linestyle='--')
                    if meann :
                        plt.axhline(y=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].mean(),2),color='red',linestyle='--')
                    if stdd :
                        plt.axhline(y=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].std(),2),color='green',linestyle='--')
                
                    plt.plot(data['Time_s'][nb_data[0]:nb_data[1]],data[n_estimators][nb_data[0]:nb_data[1]])
                    
                    st.pyplot(fig)
                    col=st.columns(5)
                    col[0].metric(label="Maximum graph value ", value=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].max(),2),
                        delta=(np.round(data[str(n_estimators)].max() - data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].max(),2)),
                        delta_color="inverse")
                    col[1].metric(label="Mean graph value", value=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].mean(),2),
                        delta=(np.round(data[str(n_estimators)].mean() - data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].mean(),2)),
                        delta_color="inverse")
                    col[2].metric(label="Median graph value", value=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].median(),2),
                        delta=(np.round(data[str(n_estimators)].median() - data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].median(),2)),
                        delta_color="inverse")
                    col[3].metric(label="STD graph value", value=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].std(),2),
                        delta=(np.round(data[str(n_estimators)].std() - data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].std(),2)),
                        delta_color="inverse")
                    col[4].metric(label="Minimum graph value", value=np.round(data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].min(),2),
                        delta=(np.round(data[str(n_estimators)].min() - data[str(n_estimators)].loc[nb_data[0]:nb_data[1]].min(),2)),
                        delta_color="inverse")
                    st.subheader("Multi-Analysis")
                    nb_data_ml = st.slider('Data sample for Multi-Analysis', value=[0,int(len(data))], min_value=0, max_value=int(len(data)),step =50)
                    Multi_Analysis = st.radio(label = 'Multi-Analysis', options = ['Liniar acceleration','Variation Accel','Angular rate','Variation Giro','Gravity'])
                    if Multi_Analysis=='Liniar acceleration':
                        ml_params = st.multiselect("Select parameters to include on map", options=var, default=['LINIAR_X_ACCEL','LINIAR_Y_ACCEL','LINIAR_Z_ACCEL','LINIAR_X_ACCEL_Mean','LINIAR_Y_ACCEL_Mean','LINIAR_Z_ACCEL_Mean'])
                        fig = plt.figure(figsize = (12.5,6))
                        for x in ml_params :
                            plt.plot(data['Time_s'][nb_data_ml[0]:nb_data_ml[1]],data[x][nb_data_ml[0]:nb_data_ml[1]],label = x)
                        plt.legend()
                        st.plotly_chart(fig)
                    if Multi_Analysis=='Angular rate':
                        ml_params = st.multiselect("Select parameters to include on map", options=var, default=['X_GIRO','Y_GIRO','Z_GIRO','X_GIRO_Mean','Y_GIRO_Mean','Z_GIRO_Mean','SPEED_','DETECT','compteur'])
                        fig = plt.figure(figsize = (12.5,6))
                        for x in ml_params :
                            plt.plot(data['Time_s'][nb_data_ml[0]:nb_data_ml[1]],data[x][nb_data_ml[0]:nb_data_ml[1]],label = x)
                        plt.legend()
                        st.plotly_chart(fig)
                    if Multi_Analysis=='Variation Accel':
                        ml_params = st.multiselect("Select parameters to include on map", options=var, default=['Var_X_Accel','Var_Z_Accel','Var_Y_Accel'])
                        fig = plt.figure(figsize = (12.5,6))
                        for x in ml_params :
                            plt.plot(data['Time_s'][nb_data_ml[0]:nb_data_ml[1]],data[x][nb_data_ml[0]:nb_data_ml[1]],label = x)
                        plt.legend()
                        st.plotly_chart(fig)
                    if Multi_Analysis=='Variation Giro':
                        ml_params = st.multiselect("Select parameters to include on map", options=var, default=['Var_X_Giro','Var_Y_Giro','Var_Z_Giro'])
                        fig = plt.figure(figsize = (12.5,6))
                        for x in ml_params :
                            plt.plot(data['Time_s'][nb_data_ml[0]:nb_data_ml[1]],data[x][nb_data_ml[0]:nb_data_ml[1]],label = x)
                        plt.legend()
                        st.plotly_chart(fig)
                    if Multi_Analysis=='Gravity':
                        ml_params = st.multiselect("Select parameters to include on map", options=var, default=['X_GRAV','Y_GRAV','Z_GRAV'])
                        fig = plt.figure(figsize = (12.5,6))
                        for x in ml_params :
                            plt.plot(data['Time_s'][nb_data_ml[0]:nb_data_ml[1]],data[x][nb_data_ml[0]:nb_data_ml[1]],label = x)
                        plt.legend()
                        st.plotly_chart(fig)
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                
                else :
                    st.warning("please select a data sample !")
                
                st.success("Done !")
            else:
                st.error("Please uploaded data ?!")
    
    else:
        with dataset_summary:

            uploaded_files_  = st.file_uploader("Choose a CSV file",type ='CSV',accept_multiple_files=True)
            list_uploaded_file_=[]
            list_uploaded_file_detect_=[]
            uploaded_files_ = sorted(uploaded_files_,key=sort_by_name)
            if (uploaded_files_ != []):
                for i in range (0,len(uploaded_files_)):
                    list_uploaded_file_.append(uploaded_files_[i].name)
                    list_uploaded_file_detect_.append(detect_crash(uploaded_files_[i]))
                    st.write('File: ',i+1)
                
                list_uploaded_ = {'Files':list_uploaded_file_,'Crash detected':list_uploaded_file_detect_}
                df_uploaded_ = pd.DataFrame(list_uploaded_)
                
                # Display an interactive table
                st.write(df_uploaded_.style.applymap(lambda x: 'color: Red;' if x == 'Yes' else None))
            else:
                st.error("Please uploaded data ?!")

if __name__ == "__main__":
    main()