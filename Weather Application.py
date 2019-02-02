import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, time, date
from dateutil import parser, rrule
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import Frame
from tkinter import BooleanVar
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import Menu
from tkinter import Label
from tkinter import Entry

#Temperature Extraction
def getTemparatureData(day, month, year):
    url = "https://www.wunderground.com/history/airport/VOMM/{year}/{month}/{day}/DailyHistory.html?reqdb.zip=&reqdb.magic=&reqdb.wmo="
    new_url = url.format(day=day, month=month, year=year)
    response = requests.get(new_url)
    res=response.content
    soup=BeautifulSoup(res,"html.parser")
    data={}
    
    data["Date"] = "{day}/{month}/{year}".format(day=day,month=month,year=year)
    
    try:
       df = pd.read_html(str(soup),flavor="bs4")
    except Exception as e:
       pass

    try:
       data["Mean Temperature"] = df[0].Actual[1].replace("\xa0°C", "")
    except Exception as e:
       pass

    try: 
       data["Max Temperature"] = df[0].Actual[2].replace("\xa0°C", "")
    except Exception as e:
       pass

    try:
       data["Min Temperature"] = df[0].Actual[3].replace("\xa0°C", "")
    except Exception as e:
       pass

    try:
       data["Dew point (celcius)"] = df[0].Actual[7].replace("\xa0°C", "")
    except Exception as e:
       pass

    try:
       data["Average Humidity"] = df[0].Actual[8]
    except Exception as e:
       pass

    try:
       data["Maximum Humidity"] = df[0].Actual[9]
    except Exception as e:
       pass

    try: 
       data["Minimum Humidity"] = df[0].Actual[10]
    except Exception as e:
       pass

    try:
       data["Sea Pressure (hPa)"] = df[0].Actual[14].replace("\xa0hPa", "")
    except Exception as e:
       pass
    
    try:
       data["Wind Speed (Km/h)"] = df[0].Actual[16].replace("\xa0km/h  ()", "")   
    except Exception as e:
       pass

    try:
       data["Max Wind Speed (Km/h)"] = df[0].Actual[17].replace("\xa0km/h", "") 
    except Exception as e:
       pass

    try:
       data["Visibility (Kms)"] = df[0].Actual[19].replace("\xa0kilometers", "") 
    except Exception as e:
       pass
    
    return data

#Temperature Extraction into CSV file
def gettemp():
   # start_date = "2000-01-01"
   # end_date = "2018-02-28"
  
   start = parser.parse(extract_start_date)
   end = parser.parse(extract_end_date)
   dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))

   weather_data = []
   for i,d in enumerate(dates):
      Weather = getTemparatureData(d.day,d.month,d.year)
      weather_data.append(Weather)

      if(i%10 == 0):
          print("Extracting Weather Data ...{}".format(d))
          df = pd.DataFrame(weather_data).set_index("Date")
          df.to_csv("Weather_Parameters_Extract.csv")

   df = pd.DataFrame(weather_data).set_index("Date")
   df.to_csv("Weather_Parameters_Extract.csv")


def getPlanetsPosition(year,month,day):
    
    url = "https://www.drikpanchang.com/tables/planetary-positions-sidereal.html?date={day}/{month}/{year}&time=12:10:00"
    data = {}
    fresh_url = url.format(day=day, month=month, year=year)
    #print(fresh_url)
    req = requests.get(fresh_url)
    soup=BeautifulSoup(req.content,"html.parser")
    df = pd.read_html(str(soup),flavor="bs4")
    for i in range(1,15):
        feature01 = df[2].values[i,0] + "_NL"
        feature02 = df[2].values[i,0] + "_Degree"
        data[feature01] = df[2].values[i,4]
        data[feature02] = df[2].values[i,5]
    data["Date"] = "{day}/{month}/{year}".format(day=day,month=month,year=year)
    
    return data


def getpp():
   
   url = "https://www.drikpanchang.com/tables/planetary-positions-sidereal.html?date={day}/{month}/{year}&time=12:10:00"
   
   start = parser.parse(extract_start_date_pp)
   end = parser.parse(extract_end_date_pp)
   dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))

   data = []
   for i,d in enumerate(dates):
      extract = getPlanetsPosition(d.year,d.month,d.day)
      data.append(extract)

      if(i%10 == 0):
        print("Extracting data into file... {}".format(d))
        df = pd.DataFrame(data).set_index("Date")
        df.to_csv("Planet_Position_data.csv")

   df = pd.DataFrame(data).set_index("Date")
   df.to_csv("Planet_Position_data.csv")


def extract_start_date():
   global extract_start_date
   extract_start_date = simpledialog.askstring("Extraction Start Date", "Enter the Extraction Start date in YYYY-MM-DD format")
   esd1=Label(frame1,text=extract_start_date, relief='flat',width=15).grid(row=2,column=2)
   
def extract_end_date():
   global extract_end_date
   extract_end_date = simpledialog.askstring("Extraction End Date", "Enter the Extraction End date in YYYY-MM-DD format")
   eed1=Label(frame1,text=extract_end_date, relief='flat',width=15).grid(row=3,column=2)


def extract_start_date_pp():
   global extract_start_date_pp
   extract_start_date_pp = simpledialog.askstring("Extraction Start Date", "Enter the Extraction Start date in YYYY-MM-DD format")
   esd2=Label(frame2,text=extract_start_date_pp, relief='flat',width=15).grid(row=2,column=2)
   
def extract_end_date_pp():
   global extract_end_date_pp
   extract_end_date_pp = simpledialog.askstring("Extraction End Date", "Enter the Extraction End date in YYYY-MM-DD format")
   eed2=Label(frame2,text=extract_end_date_pp, relief='flat',width=15).grid(row=3,column=2)

def choose_loc_temp():
   global loca_t
   temp_t1 = simpledialog.askinteger("Location", "Choose the Location from below Options \n\n 1 - Chennai \n 2 - Bangalore \n 3 - Mumbai \n 4 - Kolkata \n 5 - Delhi \n ")
   loca_t = switch_funct(temp_t1)
   print(loca_t)
   loca_t_n = switch_func_n(temp_t1)
   Label(frame1,text=loca_t_n, relief='flat',width=15).grid(row=4,column=2)

def choose_loc_pp():
   global loca_p
   temp_t2 = simpledialog.askinteger("Location", "Choose the Location from below Options \n\n 1 - Chennai \n 2 - Bangalore \n 3 - Mumbai \n 4 - Kolkata \n 5 - Delhi \n ")
   loca_p = switch_funcp(temp_t2)
   print(loca_p)
   loca_p_n = switch_func_n(temp_t2)
   Label(frame2,text=loca_p_n, relief='flat',width=15).grid(row=4,column=2)

def switch_funct(temp_t1):
   return {
   1:"VOMM",
   2:"VOBL",
   3:"VABB",
   4:"VIDP",
   5:"VECC"
   }.get(temp_t1,"VOMM")

def switch_func_n(temp_n):
   return {
   1:"Chennai",
   2:"Bangalore",
   3:"Mumbai",
   4:"Kolkata",
   5:"Delhi"
   }.get(temp_n,"Chennai")

def switch_funcp(temp_t2):
   return {
   1:9575,
   2:10645,
   3:1055,
   4:10304,
   5:10436
   }.get(temp_t2,9575)

def getDegrees_Y(dates_string,temperatures,planet_data,y_index):
   #print(temperatures.values[:,0])
   
   try:
     arr_idx =  [np.where(temperatures.values[:,0] == d)[0][0] for d in dates_string]
     Y=[]
     for i in arr_idx:
         try:
           if not np.isnan(float(temperatures.iloc[i,y_index])):
             Y.append([i, temperatures.iloc[i,0], float(temperatures.iloc[i,y_index])])
         except ValueError:
           pass

     Y = np.array(Y)
     Y_keys = ["id","Date",temperatures.keys()[y_index]]
     #print(type(Y))
     #print(type(Y_keys))
   except:
     Y=[]
     Y_keys = []

   try:
     arr_idx2 =  [np.where(planet_data.values[:,0] == d)[0][0] for d in Y[:,1]]
     X_keys = planet_data.iloc[:,1::2].keys()
     X = planet_data.values[arr_idx2][:,1::2]
     assert(len(X)==len(Y))
     return {"X_keys" : X_keys, "X" : X, "Y_keys" : Y_keys, "Y":Y}

   except:
     Y=[]
     Y_keys = []

     arr_idx2 =  [np.where(planet_data.values[:,0] == d)[0][0] for d in dates_string]
     X_keys = planet_data.iloc[:,1::2].keys()
     X = planet_data.values[arr_idx2][:,1::2]
     return {"X_keys" : X_keys, "X" : X, "Y_keys" : Y_keys, "Y": Y}

def getDateString(start_date,end_date):
   start = parser.parse(start_date)
   end = parser.parse(end_date)
   dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))
 
   dates_string = ["{day}/{month}/{year}".format(day=d.day,month=d.month,year=d.year) for d in dates]

   return dates_string

def train(temperatures,planet_data,start_date,end_date,x_index,y_index,fs):

    dates_string = getDateString(start_date,end_date)
    ret = getDegrees_Y(dates_string,temperatures,planet_data,y_index)
    print("Training data :", len(ret["X"]),len(ret["Y"]))

    #Keep mars,mercury,moon,sun,venus
    X_d = ret["X"][:,x_index]

    ##Feature Scaling
    from sklearn.preprocessing import StandardScaler

    if fs:
       sc_X = StandardScaler()
       X_tr = sc_X.fit_transform(X_d)
       sc_Y = StandardScaler()
       y_tr = sc_Y.fit_transform(ret["Y"][:,-1].reshape(-1,1))
    else:
        X_tr = X_d
        y_tr = ret["Y"][:,-1].reshape(-1,1)
        sc_X = None
        sc_Y = None
    
    #train
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_tr,y_tr)
    
    #Print coeff
    print("alpha = ", regressor.intercept_)
    print(pd.DataFrame(np.column_stack((np.array(regressor.coef_)[0],np.array(ret["X_keys"][x_index])))))
    
    return {"reg":regressor,"X_tr":sc_X,"y_tr":sc_Y}


def train_test(indep_var_idx,dep_var_idx,feature_scaling,tr_start,tr_end,test_start,test_end):

   temperatures = pd.read_csv("Chennai_temperature_data.csv")
   planet_data = pd.read_csv("Chennai_PP_data.csv")
    
   ##Training
   reg = train(temperatures,planet_data,tr_start,tr_end,indep_var_idx,dep_var_idx,feature_scaling)

   ##Testing
   dates_string = getDateString(test_start,test_end)
   ret1 = getDegrees_Y(dates_string,temperatures,planet_data,dep_var_idx)
   print("Test data ", len(ret1["X"]),len(ret1["Y"]))

   X_td = ret1["X"][:,indep_var_idx]

   if feature_scaling:
      X_test = reg["X_tr"].transform(X_td)
   else:
      X_test = X_td

   y_pred = reg["reg"].predict(X_test)

   import matplotlib.pyplot as plt

   if feature_scaling:
      plt.subplot(2, 1, 1)
      plt.plot(reg["y_tr"].inverse_transform(y_pred),label="pred")
      plt.ylabel('Predicted')
      plt.legend()
   else:
      plt.subplot(2, 1, 1)
      plt.plot(y_pred,label="pred",color='b')
      plt.ylabel('Predicted')
      plt.legend()

   #print(ret1["Y"])
   #print(len(ret1["Y"]))
   if len(ret1["Y"]) == 0:
      plt.xlabel('time (days)')
      pass
   else:
      plt.subplot(2,1,2)
      plt.plot(ret1["Y"][:,-1],label="actual",color='g')
      plt.xlabel('time (days)')
      plt.ylabel('Actual')
      plt.legend()

   from sklearn.metrics import mean_squared_error, r2_score
   try:
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(X_td, y_pred))
   except:
    pass 
   
   try:  
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(X_td, y_pred))
   except:
    pass
   
   try:  
    # Explained variance score: 1 is perfect prediction
    print('Explained Variance score: %.2f' % explained_variance_score(X_td, y_pred))
   except:
    pass
    
   return plt


def getTrainSD():
   global tr_start
   tr_start = simpledialog.askstring("Training Start Date", "Enter the Training Set Start date in YYYY-MM-DD format")
   Label(frame3,text=tr_start, relief='flat',width=15).grid(row=2,column=2)
   
def getTrainED():
   global tr_end
   tr_end = simpledialog.askstring("Training End Date", "Enter the Training Set End date in YYYY-MM-DD format")
   Label(frame3,text=tr_end, relief='flat',width=15).grid(row=3,column=2)

def getTestSD():
   global test_start
   test_start = simpledialog.askstring("Test Start Date", "Enter the Test Set Start date in YYYY-MM-DD format")
   Label(frame3,text=test_start, relief='flat',width=15).grid(row=4,column=2)

def getTestED():
   global test_end
   test_end = simpledialog.askstring("Test End Date", "Enter the Test Set End date in YYYY-MM-DD format")
   Label(frame3,text=test_end, relief='flat',width=15).grid(row=5,column=2)

def getPredictor():
   global dep_var_idx
   dep_var_idx = 6
   dep_var_idx = simpledialog.askinteger("Prediction", "Choose What to Predict from below Options \n\n 1 - Average Humidity \n 2 - Dew Point \n 3 - Maximum Humidity \n 4 - Maximum Temperature \n 5 - Maximum Wind Speed \n 6 - Mean Temperature \n 7 - Minimum Humidity \n 8 - Minimum Temperature \n 9 - Average Wind Speed")
   dep_var_idx_n = switch_funcd(dep_var_idx)
   Label(frame3,text=dep_var_idx_n, relief='flat',width=15).grid(row=6,column=2)

def switch_funcd(temp_d):
   return {
   1:"Average Humidity",
   2:"Dew point",
   3:"Maximum Humidity",
   4:"Maximum Temperature",
   5:"Maximum Wind Speed",
   6:"Mean Temperature",
   7:"Minimum Humidity",
   8:"Minimum Temperature",
   9:"Average Wind Speed"
   }.get(temp_d,"Mean Temperature")

def getIPredictor():
   global indep_var_idx
   temp_i = simpledialog.askinteger("Planets", "Choose the planets from below Options \n\n 1 - High frequency Planets(Sun, Moon, Mercury, Mars, Venus) \n 2 - Sun \n 3 - Mercury \n 4 - Venus \n 5 - Moon \n 6 - Mars \n 7 - Jupiter \n 8 - Saturn \n 9 - Uranus \n 10 - Neptune \n 11 - Ketu \n 12 - Rahu  \n 13 - True Ketu \n 14 - True Rahu \n ")
   indep_var_idx = switch_func(temp_i)
   #print(indep_var_idx)
   indep_var_idx_n = switch_func1(temp_i)
   Label(frame3,text=indep_var_idx_n, relief='flat',width=15).grid(row=7,column=2)

def switch_func(temp_i):
   return {
   1:[3,4,5,9,13],
   2:[9],
   3:[4],
   4:[13],
   5:[3],
   6:[5],
   7:[1],
   8:[8],
   9:[12],
   10:[6],
   11:[2],
   12:[7],
   13:[10],
   14:[11]
   }.get(temp_i,[3,4,5,9,13])

def switch_func1(temp_i):
   return {
   1:"Sun, Moon, Mercury, \n Mars, Venus",
   2:"Sun",
   3:"Mercury",
   4:"Venus",
   5:"Moon",
   6:"Mars",
   7:"Jupiter",
   8:"Saturn",
   9:"Uranus",
   10:"Neptune",
   11:"Ketu",
   12:"Rahu",
   13:"True Ketu",
   14:"True Rahu"
   }.get(temp_i,"Sun, Moon, Mercury, \n Mars, Venus")

def MLR():
   #try:
   plt = train_test(indep_var_idx,dep_var_idx,feature_scaling,tr_start,tr_end,test_start,test_end)
   plt.show()
   #except:
   # messagebox.showinfo("Error","Fill in the correct details")

def donothing():
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()

def temp_extract():

   frame2.destroy()
   frame3.destroy()
   
   global frame1
   frame1 = Frame(root)
   frame1.pack(side='top',fill='x')
         
   Label(frame1,text="TEMPERATURE EXTRACTION", relief='flat', padx=10,pady=10).grid(row=0,column=0, columnspan=3)
   Label(frame1,text="STEPS", relief='ridge',width=15,bg='white').grid(row=1,column=0)
   Label(frame1,text="PROCESS", relief='ridge',width=25,bg='white').grid(row=1,column=1)
   Label(frame1,text="VALUES", relief='ridge',width=15,bg='white').grid(row=1,column=2)
   steps = ['Step1','Step2','Step3']
   r = 2
   for c in steps:
     Label(frame1,text=c, relief='ridge',width=15).grid(row=r,column=0)
     r = r + 1

   extract_start_date1 = tk.Button(frame1, text ="Input Start Date", command = extract_start_date)
   extract_end_date1 = tk.Button(frame1, text ="Input End Date", command = extract_end_date)
   extract_location1 = tk.Button(frame1,text = "Choose Location", command= choose_loc_temp)
   TE = tk.Button(frame1, text = "Extract Temperature Data", command = gettemp)
   extract_start_date1.grid(row=2,column=1)
   extract_end_date1.grid(row=3,column=1)
   extract_location1.grid(row=4,column=1)
   TE.grid(row=5,column=0,columnspan=3)
   # frame1.grid_remove()

def pp_extract():

   frame1.destroy()
   frame3.destroy()
   
   global frame2
   frame2 = Frame(root)
   frame2.pack(side='top',fill='x')
      
   Label(frame2,text="PLANET POSITION EXTRACTION", relief='flat', padx=10,pady=10).grid(row=0,column=0, columnspan=3)
   Label(frame2,text="STEPS", relief='ridge',width=15,bg='white').grid(row=1,column=0)
   Label(frame2,text="PROCESS", relief='ridge',width=25,bg='white').grid(row=1,column=1)
   Label(frame2,text="VALUES", relief='ridge',width=15,bg='white').grid(row=1,column=2)
   steps = ['Step1','Step2','Step3']
   r = 2
   for c in steps:
     Label(frame2,text=c, relief='ridge',width=15).grid(row=r,column=0)
     r = r + 1

   extract_start_date2 = tk.Button(frame2, text ="Input Start Date", command = extract_start_date_pp)
   extract_end_date2 = tk.Button(frame2, text ="Input End Date", command = extract_end_date_pp)
   extract_location2 = tk.Button(frame2,text = "Choose Location", command= choose_loc_pp)
   PP = tk.Button(frame2, text = "Extract Planet Position Data", command = getpp)
   extract_start_date2.grid(row=2,column=1)
   extract_end_date2.grid(row=3,column=1)
   extract_location2.grid(row=4,column=1)
   PP.grid(row=5,column=0,columnspan=3)
   # Label(frame2,command=esd1.grid_forget())
   

def weatherprediction():

   frame1.destroy()
   frame2.destroy()
   
   global frame3
   frame3 = Frame(root)
   frame3.pack(side='top',fill='x')

   Label(frame3,text="WEATHER PREDICTION", relief='flat', padx=10,pady=10).grid(row=0,column=0, columnspan=3)
   Label(frame3,text="STEPS", relief='ridge',width=15,bg='white').grid(row=1,column=0)
   Label(frame3,text="PROCESS", relief='ridge',width=25,bg='white').grid(row=1,column=1)
   Label(frame3,text="VALUES", relief='ridge',width=15,bg='white').grid(row=1,column=2)
   steps = ['Step1','Step2','Step3','Step4','Step5','Step6','Step7']
   r = 2
   for c in steps:
     Label(frame3,text=c, relief='ridge',width=15).grid(row=r,column=0)
     r = r + 1

   tr_start1 = tk.Button(frame3, text ="Input Training Set Start Date", command = getTrainSD)
   tr_end1 = tk.Button(frame3, text ="Input Training Set End Date", command = getTrainED)
   test_start1 = tk.Button(frame3, text ="Input Test Set Start Date", command = getTestSD)
   test_end1 = tk.Button(frame3, text ="Input Test Set End Date", command = getTestED)
   dep_var_idx1 = tk.Button(frame3, text ="Choose What to Predict", command = getPredictor)
   indep_var_idx1 = tk.Button(frame3, text ="Choose Plantes", command = getIPredictor)

   def getBool(): # get rid of the event argument
    global feature_scaling
    feature_scaling=boolvar.get()
    Label(frame3,text=feature_scaling, relief='flat',width=15).grid(row=8,column=2)
    print(feature_scaling)

   boolvar = BooleanVar()
   boolvar.set(False)
   boolvar.trace('w', lambda *_: print("The value was changed"))
   cb = tk.Checkbutton(frame3, text = "Is Feature Scaling applicable?", variable = boolvar, command = getBool)
   
   LR = tk.Button(frame3, text = "Run Prediction", command = MLR )

   tr_start1.grid(row=2,column=1)
   tr_end1.grid(row=3,column=1)
   test_start1.grid(row=4,column=1)
   test_end1.grid(row=5,column=1)
   dep_var_idx1.grid(row=6,column=1)
   indep_var_idx1.grid(row=7,column=1)
   cb.grid(row=8,column=1)
   LR.grid(row=9,column=0,columnspan=3)

def clear_frame():
   frame1.destroy()
   frame2.destroy()
   frame3.destroy()

def helpfunc():
   messagebox.showinfo("Help","Choose the application from the File menu")

def aboutfunc():
   messagebox.showinfo("About","This application has been created to demostrate the Dissertation Work carried out for availing M.Tech Degree")
   

if __name__ == "__main__":

   # Default Values
   # indep_var_idx = [3,4,5,9,13]   # dep_var_idx = 6   # feature_scaling = False
   # tr_start = "2014-06-01"   # tr_end = "2016-12-1"   # test_start = "2017-01-01"   # test_end = "2017-02-1"

   root = tk.Tk()
   root.title("Weather Application BITS Dissertation - 2016HT12799")
   width=600
   height=350
   sw=root.winfo_screenwidth()
   sh=root.winfo_screenheight()
   x_coord=(sw/2)-(width/2)
   y_coord=(sh/2)-(height/2)
   root.geometry("%dx%d+%d+%d" %(width,height,x_coord,y_coord))

   global frame1
   frame1 = Frame(root)
   frame1.pack(side='top',fill='x')
   
   global frame3
   frame3 = Frame(root)
   frame3.pack(side='top',fill='x')
   
   global frame2   
   frame2 = Frame(root)
   frame2.pack(side='top',fill='x')
  
   # rootimage = tk.PhotoImage(file='Root_img.png')
   # label_image=Label(root,image=rootimage)
   # label_image.place(anchor='center')
   
   menubar = Menu(root)
   filemenu = Menu(menubar, tearoff=0)
   filemenu.add_command(label="Temperature Extraction", command=temp_extract)
   filemenu.add_command(label="Plantes Position Extraction", command=pp_extract)
   filemenu.add_command(label="Weather Prediction", command=weatherprediction)
   filemenu.add_separator()
   filemenu.add_command(label="Exit", command=root.quit)
   menubar.add_cascade(label="File", menu=filemenu)

   editmenu = Menu(menubar, tearoff=0)
   editmenu.add_command(label="Clear", command=clear_frame)
   menubar.add_cascade(label="Edit", menu=editmenu)

   helpmenu = Menu(menubar, tearoff=0)
   helpmenu.add_command(label="Help Index", command=helpfunc)
   helpmenu.add_command(label="About", command=aboutfunc)
   menubar.add_cascade(label="Help", menu=helpmenu)

   
   root.config(menu=menubar)

   root.mainloop()