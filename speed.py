# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:36:52 2019

@author: mifuad
"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
#from datetime import timedelta
import numpy as np
#import pprint
import pymongo
plt.close('all')

#%%

def GetBondData(month, day, starttime):  # compile high level data from each Bond Event into a dataframe
    #Import Bond Data from MongoDB
    print("Getting new Bond Data Starting from {}/{}/2019".format(month, day))
    from pymongo import MongoClient
    client = MongoClient('10.6.50.91', 27017)
    db = client.UltraWire
    BondDataCo = db.BondData
    WaferDataCo = db.WaferData
    timeoff = 7
    d = datetime(2020, month, day, starttime+timeoff)
    files3 = BondDataCo.find({"StartTime":{"$gt":d}})
    filesW = WaferDataCo.find({"PlaceTime":{"$gt":d}})
    BD = pd.DataFrame(list(files3))
    WD = pd.DataFrame(list(filesW))
    BondDataPoint = pd.DataFrame(BD)
    WaferDataPoint = pd.DataFrame(WD)
    cols = BondDataPoint.columns.tolist()
    return BondDataPoint, WaferDataPoint

def CompileDF(BondDF): # compiles LVDT, Load Cell, Temperature, and Motor torque data for each Bond event in the Bond DataFrame
    print("Compiling LVDT and Load cell data")
    from pymongo import MongoClient
    client = MongoClient('10.6.50.91', 27017)
    db = client.UltraWire
    LVDTcollection = db.LvdtData
    LCcollection = db.LoadCellData
    n = 0
    _runs = len(BondDF["StartTime"])
    LCFiles = {}
    LVFiles = {}
    print("There are {} runs being compiled)".format(_runs))
    while n < _runs:
        starttime = BondDF["StartTime"].iloc[n]
        endtime = BondDF["EndTime"].iloc[n]
        LVfiles = LVDTcollection.find({"LogTime": {"$gt": starttime, "$lt":endtime}})
        LCfiles = LCcollection.find({"LogTime": {"$gt": starttime, "$lt":endtime}})
        LV = pd.DataFrame(list(LVfiles))
        LC = pd.DataFrame(list(LCfiles))
        LVd = pd.DataFrame(LV)
        LCd = pd.DataFrame(LC)
        LVd[['West', 'East']] = pd.DataFrame(LVd['SensorInstantaneous'].values.tolist())
        LCd[['Lower East', 'Upper East', 'Upper West', 'Lower West']] = pd.DataFrame(LCd['SensorInstantaneous'].values.tolist())
        LVFiles[n] = LVd
        LCFiles[n] = LCd
        print("There are {} runs remaining".format(_runs-n-1))
        n += 1
    keys = BondDF["BondCounter"]
    LVFiles = dict((keys[key],value) for (key, value) in LVFiles.items())
    LCFiles = dict((keys[key],value) for (key, value) in LCFiles.items())
    return LVFiles, LCFiles

#%%  Plot LVDT and LC data

def plotLVandLCdata(run_start, run_end, xaxis, LVFiles, LCFiles, num_runs, BondDF):
    print("Plotting data for Bond run {} through {}".format(run_start, run_end))
    #plt.close()
    #offset = 50
    #offset2 = 75
    bondheadweight = 185
    LVoffset = 1000
    #run_start = 6
    #run_end = 7
    num_plots = run_end-run_start+1
    #plt.subplot(num_runs,2,1)
    #plt.figure(1)
    plt.figure(figsize=(15, 35))
    temp = 1
    for x in BondDF["EndTime"]:
        if run_start > run_end:
            break

        #plt.plot(LCshort["HotPlatePosition"], LCshort["West"],"*")
        #plt.plot(LCshort["HotPlatePosition"], LCshort["East"],"*")
        #plt.plot(LCshort["Time"], LCshort["Lower West"],"*")
        #plt.plot(LCshort["Time"], LCshort["Lower East"],"*")
        
        ax = plt.subplot(num_plots, 2, temp)
        LCbond = LCFiles[run_start]
        LVbond = LVFiles[run_start]
        x1 = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'East'].mean()
        x2 = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'West'].mean()
        x3 = LCbond.loc[LCbond['HotPlatePosition'].between(975,1460), 'Upper East'].mean()
        x4 = LCbond.loc[LCbond['HotPlatePosition'].between(975,1460), 'Upper West'].mean()
        LVaverage = (x1+x2)/20
        LCaverage = (x3+x4)/2
        xaxis = "HotPlatePosition"
        #ax = plt.plot(LCbond[xaxis], LCbond["Upper West"] - LCbond["Lower West"]+bondheadweight, "*")
        #ax = plt.plot(LCbond[xaxis], LCbond["Upper East"] - LCbond["Lower East"]+bondheadweight, "*")
        ax = plt.plot(LCbond[xaxis], LCbond["Upper West"] , "-")
        ax = plt.plot(LCbond[xaxis], LCbond["Upper East"] , "-")
        ax = plt.plot(LCbond[xaxis], LCbond["Lower West"] , "-")
        ax = plt.plot(LCbond[xaxis], LCbond["Lower East"] , "-")
        ax = plt.plot(LCbond[xaxis], LCbond["CurrentPercentage"]/10 , "-")
        ax = plt.plot(LCbond[xaxis], LCbond["EastPressCurrent"]/10 , "-")
        ax = plt.plot(LCbond[xaxis], LCbond["WestPressCurrent"]/10 , "-")
        ax = plt.legend(["UW", "UE", "LW", "LE", "Current","East Curr","West Curr"])
        ax = plt.title("LC Run {}".format(run_start))
        ax = plt.xlabel(xaxis+" (mm)")
        ax = plt.ylabel("Force (Newtons)")
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=.2, hspace=0.2+num_plots/12)
        plt.ylim([0,500])
        #plt.ylim([LCaverage-100, LCaverage+100])
        if xaxis == "HotPlatePosition":
            plt.xlim([950, 1500])
        ax = plt.subplot(num_plots, 2, temp+1)
        ax = plt.plot(LVbond[xaxis], LVbond["West"]/10+LVoffset, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["East"]/10, "*")
        
        ax = plt.plot(LVbond[xaxis], LVbond["BondheadTemp"], "*")
        ax = plt.plot(LVbond[xaxis], LVbond["BondheadTemp2"], "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateNorth"], "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateCenter"], "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateSouth"], "*")
        
        ax = plt.plot(LVbond[xaxis], LVbond["BondheadMV"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["BondheadMV2"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateNorthMV"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateCenterMV"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateSouthMV"]/10, "*")
        
        ax = plt.legend(["W", "E","BHTemp1","BHTemp2","North","Center","South","BH1MV","BH2MV","NorthMV","CenterMV","SouthMV"])
        ax = plt.xlabel(xaxis+" (mm)")
        sp = BondDF["BondSpeed"].loc[BondDF["BondCounter"] == run_start].values.tolist()
        bond = BondDF["BondType"].loc[BondDF["BondCounter"] == run_start].values.tolist()
        ax = plt.ylabel("LVDT Pos (\u03bcm)")
        #plt.ylim([average/10000 - .1400,average/10000+.0700])
        if xaxis == "HotPlatePosition":
            plt.xlim([950, 1500])
        #plt.ylim([LVaverage, LVaverage+500])
        plt.ylim([-10,105])
        ax = plt.title("LVDT Run {}, Bond Speed {} mm/s, {}".format(run_start, sp, bond))
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=.2, hspace=0.2+num_plots/10)
        #print(temp)
        print(LVbond["LogTime"].iloc[1])
        temp += 2
        run_start += 1
        #LVavg = LVbond.groupby()

    return

#plt.subplots_adjust(wspace = 0.15, hspace = 0.35, left = .07, right = .97, top = .95)
#fig.tight_layout()
#plt.plot.show
#%%
def compare_runs(run1, run2 , LVFiles, BondDF, LCFiles):
    fig = plt.figure(figsize = (15,35))
    print("Comparing Runs {} and {}".format(run1, run2))
    LVDTshort1 = LVFiles[run1]
    LVDTshort2 = LVFiles[run2]
    LCShort1 = LCFiles[run1]
    LCShort2 = LCFiles[run2]
    east1 = LVDTshort1.loc[LVDTshort1['HotPlatePosition'].between(1140,1280), 'East'].mean()
    east2 = LVDTshort2.loc[LVDTshort2['HotPlatePosition'].between(1140,1280), 'East'].mean()
    offset = (east2-east1)/10
    offset = 0
    avg = (east2+east1)/20
    ax = plt.plot(LVDTshort1["HotPlatePosition"], LVDTshort1["West"]/10+600+offset, "-")
    ax = plt.plot(LVDTshort1["HotPlatePosition"], LVDTshort1["East"]/10+offset, "-")
    ax = plt.plot(LVDTshort2["HotPlatePosition"], LVDTshort2["West"]/10+600, "-")
    ax = plt.plot(LVDTshort2["HotPlatePosition"], LVDTshort2["East"]/10, "-")
    

    
    ax = plt.legend(["W1","E1","W2","E2"])
    ax = plt.xlabel("Hot Plate Position (mm)")
    ax = plt.ylabel("LVDT Pos (\u03bcm)")
    
    avg = 1000
    plt.ylim([avg-150,avg+150])
    ax = plt.xlim([970, 1470])
    bond1 = BondDF["BondType"].loc[BondDF["BondCounter"] == run1].values
    bond2 = BondDF["BondType"].loc[BondDF["BondCounter"] == run2].values
    ax = plt.title("LVDT Run {} ({}) and {} ({})".format(run1, bond1, run2, bond2))
    
    
    ax2 = plt.twinx()
#    ax = plt.plot(LCShort1["HotPlatePosition"], LCShort1["Upper West"]-LCShort1["Lower West"], "-")
#    ax = plt.plot(LCShort1["HotPlatePosition"], LCShort1["Upper East"]-LCShort1["Lower East"], "-")
#    ax = plt.plot(LCShort2["HotPlatePosition"], LCShort2["Upper West"]-LCShort2["Lower West"], "-")
#    ax = plt.plot(LCShort2["HotPlatePosition"], LCShort2["Upper East"]-LCShort2["Lower East"], "-")
    ax = plt.plot(LCShort1["HotPlatePosition"], LCShort1["Lower West"], "-")
    ax = plt.plot(LCShort1["HotPlatePosition"], LCShort1["Lower East"], "-")
    ax = plt.plot(LCShort2["HotPlatePosition"], LCShort2["Lower West"], "-")
    ax = plt.plot(LCShort2["HotPlatePosition"], LCShort2["Lower East"], "-")
    
    ax2.plot(LCShort1["HotPlatePosition"], LCShort1["CurrentPercentage"]/10 , "-",color = ".6")
    ax2.plot(LCShort2["HotPlatePosition"], LCShort2["CurrentPercentage"]/10 , "-",color=".8")
    ax2.set_ylim([100,220])
    ax2.legend(["LW1", "LE1","LW2","LE2","Run 1","Run 2"])
    
    x = np.linspace(973, 1474, 1000)
    dat = []
    for i in x:
       dat.append(GenerateWaferOutline(i))
    ax2.plot(x,dat,".")
    p = len(LVDTshort1["West"])
    #ax3 = plt.figure()
    #ax3 = plt.plot(LVDTshort1["HotPlatePosition"], (LVDTshort1["East"]-LVDTshort2[0:p]["East"])/10)
    #ax3 = plt.plot(LCShort1["HotPlatePosition"], LCShort1["CurrentPercentage"]/10-LCShort2[0:p]["CurrentPercentage"]/10)
    #ax3 = plt.title("East 1 - East 2")
    #ax3 = plt.legend("LVDT East","Current Percentage")
    #plt.ylim([-30,30])
    #plt.xlim([973,1485])
    #ax2.ylabel("Current Percentage")
    #plt.ylim([average/10000 - .1400,average/10000+.0700])
#    plt.figure()
#    plt.plot(LVDTshort1["LogTime"], LVDTshort1["HotPlatePosition"])
#    plt.plot(LVDTshort2["LogTime"], LVDTshort2["HotPlatePosition"])
    
    #ax = plt.title("LVDT Run {} (Profile Bond) and {} (Manual Bond)".format(run1, run2))
    #print(LVDTshort1['LogTime'].iloc[1])
    #print(LVDTshort2['LogTime'].iloc[1])
    xpos = 974
    h = 0
    means = {}
    #mean[["HotPlateX","East1","West1","East2","West2"]] = pd.DataFrame()
    #mean["East1"].iloc[0] = 0
    while xpos < 1466.5 :
        x1 =  LVDTshort1.loc[LVDTshort1['HotPlatePosition'].between(xpos,xpos+5), 'East'].mean()
        x2 = LVDTshort1.loc[LVDTshort1['HotPlatePosition'].between(xpos,xpos+5), 'West'].mean()
        x3 = LVDTshort2.loc[LVDTshort2['HotPlatePosition'].between(xpos,xpos+5), 'East'].mean()
        x4 = LVDTshort2.loc[LVDTshort2['HotPlatePosition'].between(xpos,xpos+5), 'West'].mean()
        means[h] = [xpos+2.5,x1,x2,x3,x4]
        #mean["HotPlateX"].iloc[h] = xpos+2.5
        xpos += 5
        h += 1
    #mean = pd.DataFrame(means)
    runmeans = pd.DataFrame.from_dict(means, orient = 'index', columns=["HotPlateX","East1","West1","East2","West2"])
    #print("The mean is {} ".format(mean))
    return runmeans

def plotTemperature(run, LVFiles, LCFiles, BondDF):
    #print("Plotting temp data for Bond run {}".format(run))
    #plt.close()

    plt.figure()

    LCbond = LCFiles[run]
    LVbond = LVFiles[run]
    xaxis = "LogTime"
    xaxis = "HotPlatePosition"
    ax = plt.plot(LCbond[xaxis], LCbond["CurrentPercentage"]/10 , "*")
    ax = plt.plot(LVbond[xaxis], LVbond["East"]/10 , "*")
    ax = plt.plot(LVbond[xaxis], LVbond["West"]/10+800 , "*")
    #ax = plt.plot(LVbond["LogTime"], LVbond["BondheadTemp"], "*")
    #ax = plt.plot(LVbond["LogTime"], LVbond["BondheadTemp2"], "*")
    #ax = plt.plot(LVbond["LogTime"], LVbond["HotPlateNorth"], "*")
    #ax = plt.plot(LVbond["LogTime"], LVbond["HotPlateCenter"], "*")
    #ax = plt.plot(LVbond["LogTime"], LVbond["HotPlateSouth"], "*")
    if (run>195):
        ax = plt.plot(LVbond[xaxis], LVbond["BondheadMV"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["BondheadMV2"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateNorthMV"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateCenterMV"]/10, "*")
        ax = plt.plot(LVbond[xaxis], LVbond["HotPlateSouthMV"]/10, "*")
    
    force_hp = LCbond.loc[LCbond['HotPlatePosition'].between(975,1460), 'CurrentPercentage'].median()/10
    LVEast = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'East'].median()/10
    LVWest = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'West'].median()/10
    LVDT = (LVEast+LVWest)/2
    #print("The median HP current for run {} was {} percent".format(run,force_hp))
    #print("The median LVDT pos for run {} was {} ".format(run,LVDT))

    
    #ax = plt.legend(["UW", "UE", "LW", "LE", "Current","East Curr","West Curr"])
    ax = plt.title("Temp Data for Run {}".format(run))
    ax = plt.xlabel("Time (s)")
    ax = plt.ylabel("Percentage")
    #plt.subplots_adjust(bottom=0.1, top=0.9, wspace=.2, hspace=0.2+num_plots/12)
    plt.ylim([100,185])
    plt.xlim([963,1481])
    #plt.ylim([LCaverage-100, LCaverage+100])

    #ax = plt.plot(LVbond[xaxis], LVbond["West"]/10+LVoffset, "*")
    #ax = plt.plot(LVbond[xaxis], LVbond["East"]/10, "*")

    ax = plt.legend(["Current","East","West","BH1MV","BH2MV","NorthMV","CenterMV","SouthMV"])

    #print(temp)
    #print(LVbond["LogTime"].iloc[1])
    #LVavg = LVbond.groupby()
    LowerWest_Northwafer = LCbond.loc[LCbond['HotPlatePosition'].between(1310,1410), 'Upper West'].mean()
    LowerEast_Northwafer = LCbond.loc[LCbond['HotPlatePosition'].between(1310,1410), 'Upper East'].mean()
    LowerWest_Southwafer = LCbond.loc[LCbond['HotPlatePosition'].between(980,1110), 'Upper West'].mean()
    LowerEast_Southwafer = LCbond.loc[LCbond['HotPlatePosition'].between(980,1110), 'Upper East'].mean()
    print(run)
    print("North")
    print("The lower west average on the north wafer was {}".format(LowerWest_Northwafer))
    print("The lower east average on the north wafer was {}".format(LowerEast_Northwafer))
    print("South")
    print("The lower west average on the south wafer was {}".format(LowerWest_Southwafer))
    print("The lower east average on the south wafer was {}".format(LowerEast_Southwafer))

    return

#%%
    # Calculates average or median values for various metrics and compiles into a dataframe
def GetAverageHotPlateCurrent(start, end, LVFiles, LCFIles, BondDF):
    run = start
    AverageData = []
    skiplist = [50, 54, 168, 283, 289, 445] # add Bond events here as needed to skip (in case the data does not exist)
    while run<end+1:    
        if (skiplist.__contains__(run)):
            run = run+1
            continue
        LCbond = LCFiles[run]
        LVbond = LVFiles[run]
        n_hp = LCbond.loc[LCbond['HotPlatePosition'].between(1308,1465), 'CurrentPercentage'].mean()/10
        c_hp = LCbond.loc[LCbond['HotPlatePosition'].between(1142,1301), 'CurrentPercentage'].mean()/10
        s_hp = LCbond.loc[LCbond['HotPlatePosition'].between(975,1135), 'CurrentPercentage'].mean()/10
        force_hp = (n_hp+c_hp+s_hp)/3
        LVEast = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'East'].median()/10
        LVWest = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'West'].median()/10
        LCWest = LCbond.loc[LCbond['HotPlatePosition'].between(975,1460), 'Upper West'].median()
        LCEast = LCbond.loc[LCbond['HotPlatePosition'].between(975,1460), 'Upper East'].median()
        BHMV = 0
        HPMV = 0
        if (run>176):
            BHMV1 = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'BondheadMV'].mean()/10 
            BHMV2 = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'BondheadMV2'].mean()/10 
            BHMV = (BHMV1+BHMV2)/2
            HPCMV = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'HotPlateCenterMV'].mean()/10 
            HPSMV = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'HotPlateSouthMV'].mean()/10 
            HPNMV = LVbond.loc[LVbond['HotPlatePosition'].between(975,1460), 'HotPlateNorthMV'].mean()/10 
            HPMV = (HPCMV+HPNMV+HPSMV)/3
        LVDT = (LVEast+LVWest)/2
        Speed = BondDF["BondSpeed"].loc[BondDF["BondCounter"] == run].values
        HP = BondDF["HPSetPointTemp"].loc[BondDF["BondCounter"] == run].values
        BH = BondDF["BondHeadTempSV"].loc[BondDF["BondCounter"] == run].values
        Notes = BondDF["BondEventNotes"].loc[BondDF["BondCounter"] == run].values
        #print("The median HP current for run {} was {} percent".format(run,force_hp))
        #print("The median LVDT pos for run {} was {} ".format(run,LVDT))
        newdata = [run, force_hp, LVDT, LCWest, LCEast,BH, HP, BHMV, HPMV, Speed, Notes]
        AverageData.append(newdata)
        #rundata.append(run)
        #HPdata.append(force_hp)
        #LVDTdata.append(LVDT)
        run = run+1
    AverageData = pd.DataFrame(list(AverageData))
    AverageData.columns = ["BondCounter","HotplateCurrent","LVDT","UpperWest","UpperEast","BHTemp","HPTemp","BHMV","HPMV","Speed", "Notes"]
    return AverageData

#%%
c = 13.9 #(amount of chamfer)
g = 4 # not used
w = 161.7 # wafer dimension
o = 165.7 # wafer to next wafer idstance
top = 179
st = 970 # starting x position
def GenerateWaferOutline(x):
    y = top
    if(st<x<(st+c)): y = top - c+(x-st)  #south wafer first chamfer
    #if(973+c<x<973+w-c): y = top # south wafer
    if(st+w-c<x<st+w): y = top - (x-(st+w-c)) #south wafer second chamfer
    if(st+o<x<st+o+c): y = top - c+(x-(st+o)) # center wafer 1st
    if(st+o+w-c<x<st+o+w): y =  top - (x-(st+o+w-c)) # center wafer 2nd
    if(st+o*2<x<st+o*2+c): y = top - c+(x-(st+o*2)) #north wafer 1st
    if(st+o*2+w-c<x<st+o*2+w): y = y =  top - (x-(st+o*2+w-c)) #north wafer 2nd
    #else: y = top
    return y


#%%\
    #Set Variables for data collection and plotting
month = 3
day = 4
time = 1

plot_start = 675
plot_end = 680

# Comment out the next two lines if new data does not need to be collected
BondDF, WaferDF = GetBondData(month, day, time) # month, date, hour (military time)
LVFiles, LCFiles = CompileDF(BondDF)
number_runs = len(BondDF["StartTime"])
print("There were {} Bond Events Since {}/{}/2019".format(number_runs, month, day))
plotLVandLCdata(675, 680, "HotPlatePosition", LVFiles, LCFiles, number_runs, BondDF)  # run start, run end, x axis

ALLData = GetAverageHotPlateCurrent(675,680, LVFiles, LCFiles, BondDF)






