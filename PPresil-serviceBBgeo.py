
# coding: utf-8

# In[43]:


from pandas import read_csv, concat, DataFrame, to_datetime
import scipy as sc
import pandas as pd
import os
import numpy as np
from multiprocessing import Pool
idx = pd.IndexSlice

import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy.contrasts import Sum

from osgeo import gdal


# In[65]:


#Parameters
#Boundingbox


startrow=0
endrow=200
startcol=0
endcol=200


#Number of nodes in parallel
cpuNode=8
#max size of pixel square given to a worker node in a given time
maxchuckSize=100
#Path for the tempfile of memory mapping
MemFilePath="/Tmp/MSaviFeat"

#path to get input stack in envi format
path="Dati_SatPeneda"
file="MSAVI_AOI_MASKED_STACK_NEW"
 west,south,east,north =read_csv("BBfile.txt", index_col=None, header=None,delim_whitespace=True).values.flatten().tolist()
if !(south==west==north==east==0):
    ds = gdal.Open(path+file)
    ds = gdal.Translate(path+"BB_"+file, ds, projWin = [south,west,north,east])
    ds = None
    file="BB_"+file
file=path+"/"+file
#Output path
outputName='new_imageYAMNN2b.hdr'


# In[45]:


import spectral

import spectral.io.envi as envi


# In[46]:


from scipy import optimize
def MsaviError(NIR,RED):
    C=(2*NIR-1)**2 +8*(NIR-RED)
    return (4*(C**0.5 -2*NIR-1)**2 +16)/C
def Msavi(NIR,RED):
    return  (2 * NIR + 1 -  ((2 * NIR + 1)**2- 8 * (NIR - RED))**0.5) / 2
def fl(x,params,k=[1,2,3]):
    c=params[0]
    step=1
    n=(len(params)-1)/2
    m=0
    if n!=int(n):
        m=params[1]
        step=2
        n=(len(params)-2)/2
    n=int(n)
    gamma=np.array(params[step:(n+step)])
    theta=np.array(params[(step+n):])
    #c,gamma1,gamma2,gamma3,theta1,theta2,theta3=params
    freq=365
    K=np.repeat(np.array([k]),len(x),axis=0)
    theta=np.repeat(np.array([theta] ),len(x),axis=0)
    gamma=np.repeat(np.array([gamma]),len(x),axis=0)
    X=np.array([x]).transpose()
    Fix=(2*np.pi*K*X)/freq
    return c+m*x+np.sum(gamma*np.sin(Fix)+theta*np.cos(Fix),axis=1)

def residualfl(p, x, y):
    return np.nansum((y - fl(x, *p))**2)

def MakeLinear(c,a,delta):
    gamma=a*np.cos(delta)
    theta=a*np.sin(delta)
    return [c]+list(gamma)+list(theta)
def ReverseLinear(pl):
    c=pl[0]
    n=(len(pl)-1)/2
    m=None
    step=1
    if n!=int(n):
        m=pl[1]
        step=2
        n=(len(pl)-2)/2
    n=int(n)
    gamma=np.array(pl[step:(n+step)])
    theta=np.array(pl[(step+n):])
    assert len(gamma)==len(theta)
    a=(gamma**2+theta**2)**0.5
    delta=np.arctan(theta/gamma)
    res=pd.Series([c,a,delta], index=["c","a","delta"])
    if m:
        res=pd.Series([c,m,a,delta], index=["c","m","a","delta"])
    return res
def GetSinus(t,y):
    c0=1
    a0=[5,1,1]
    delta0=[1]*3
    pl0=MakeLinear(c0,a0,delta0)
    Afl = optimize.minimize(residualfl, pl0, method='L-BFGS-B',args=(t,y))
    Afl.Rsq=(1-(Afl.fun/len(y))/y.var())
    return Afl
def MakeLinearSinus(t, k=[1,2,3], trend=False, YAM=False):
    N=len(t)
    freq=365
    K=np.repeat(np.array([k]),N, axis=0)
    fix=2*np.pi/freq
    Fix=t.reshape(N,1)*fix*K
    #print(Fix.shape)
    if trend:
        Xm=np.concatenate([np.array([1]*N).reshape(N,1),
                           t.reshape(N,1),np.sin(Fix), np.cos(Fix)],axis=1)
    elif YAM:
        year=((t/365).astype("int"))
        year = Sum().code_without_intercept(list(set(year))).matrix[year,:]
        #year=sm.tools.categorical(year, drop=True)
        Xm=np.concatenate([np.array([1]*N).reshape(N,1),
                           year,np.sin(Fix), np.cos(Fix)],axis=1)
    else:
        Xm=np.concatenate([np.array([1]*N).reshape(N,1),
                           np.sin(Fix), np.cos(Fix)],axis=1)
    return Xm
def GetLinearSinus(t,y,w=None, cov_type='nonrobust', robust=False, k=[1,2,3], trend=False, YAM=False): 
    Xm=MakeLinearSinus(t,k=k, trend=trend, YAM=YAM)

    if w is not None:
        results=sm.WLS(y,Xm,weights=w, missing="drop").fit()
    elif robust:
        results=sm.robust.robust_linear_model.RLM(y,Xm, missing="drop").fit()
    else:
        results=sm.OLS(y,Xm, missing="drop").fit(cov_type=cov_type)
#     if YAM:
#         #recenter intercept and 
#         meany=np.nanmean(y)
#         c=meany-results.params[0]
#         L=len(results.params)
#         results.params[0]=meany
#         results.params[1:(L-len(k))]=results.params[1:(L-len(k))]
    return results
    

def FeatureExtraction(y,func=fl,freq=365, step=1,obs=False, k=[1,2,3],years=1, trend=False, YAM=False):
    if not obs:
        y=y.predict(MakeLinearSinus(np.arange(0,freq*years,step), k=k, trend=trend, YAM=YAM))
        y=pd.Series(y)
        #y=func(np.arange(freq*years), y, k=k)
    elif y.__class__==pd.Series().__class__:
        y=pd.Series(y.values,index=y.index.get_level_values("Days").values)
    M=y.mean()
    CV=(y.var()**0.5)/M
    anomal=[]
    pos=(y.idxmax()*step) % 365.
    if years>1:
        Y=(np.arange(0,freq*years,step)/freq).astype("int")
        anomal=pd.Series(y).groupby(Y).mean()
        #print((np.arange(freq*years)/freq).astype("int"))
        #print(anomal)
        anomal.index=[ "year"+str(x+1) for x in anomal.index]
        anomal=anomal-M
        anomal["anomalVar"]=(anomal.var())**0.5
        #pos=y.argmax()
    temp=pd.Series([M,CV,pos], index=["mean", "CV","pos"], name="feature")
    if len(anomal)>1:
        temp=pd.concat([temp,anomal])
    return temp
def GetFit(y,t):
    Afl=GetSinus(t,y)
    res=FeatureExtraction(Afl.x)
    res.append(Afl.Rsq)
    return res


# In[47]:


from statsmodels.tsa.stattools import acf
def G(V,att="f_pvalue"):
    temp=GetLinearSinus(X.index.get_level_values("Days").values,V)
    return getattr(temp,att)

def GF(V):
    temp=GetLinearSinus(X.index.get_level_values("Days").values,V).params
    res=FeatureExtraction(temp)
    return res
def GG(V, w=None, robust=None, k=[1,2,3], trend=False, YAM=False, step=15):
    try:
        if "Days" in V.index.names:
            t=V.index.get_level_values("Days").values
            V=V
        else:
            t=x
    except AttributeError:
        t=X
    temp=GetLinearSinus(t,V.values,w=w,robust=robust, k=k,trend=trend, YAM=YAM)
    params=temp.params
    nyear=1
    if YAM:
        year=(t/365).astype("int")
        nyear=len(set(year))
        params=np.concatenate([temp.params[0:1],temp.params[(nyear):]])
    featu=FeatureExtraction(temp, k=k, years=nyear,trend=trend, YAM=YAM, step=step)
    ttemp=acf(temp.resid, unbiased=True, qstat=True)
    featu["liuPvalue"]=ttemp[2].max()
    featu["liu"]=ttemp[1].mean()
    try:
        featu["pvalue"]=temp.f_pvalue
    except AttributeError:
        featu["pvalue"]=np.nan
    try:
        featu["rsq"]=temp.rsquared
    except AttributeError:
        featu["rsq"]=temp.fittedvalues.var()/V.var()
    if YAM:
        check=V.groupby(year).mean().isnull()
        check.index=["year"+str(c+1) for c in range(nyear)]
        #for c,y in enumerate(temp.params[1:nyear]):
        #    featu["year"+str(c+1)]=y
        #featu["year"+str(c+1+1)]=-np.sum(temp.params[1:nyear])
        #print(featu[check.index][check])
        featu[check.index[check]]=np.nan
        #featu["varAnom"]=featu[check.index].var()
        #print(featu[check.index][check])
    featuRaw=FeatureExtraction(V,obs=True)
    featu=pd.concat([featu,featuRaw],keys=["Mod","raw"])
    return featu
def SDonFeatu(FModel, Featu,obs=False):
    
    if obs:
        sd=np.var(FModel,ddof=1)**0.5
        N=len(FModel)
        y=FModel
    else:
        sd=((FModel.mse_resid*FModel.df_resid)/(FModel.nobs-1))**0.5
        N=FModel.nobs
        res=FModel.get_prediction(exog=MakeLinearSinus(np.arange(365)))
        Y=res.summary_frame()
        y=Y["mean"]
    alpha95=scipy.stats.t.ppf(0.975, N)
    se=((sd**2/N)**0.5)*alpha95
    Featu["Umean"]=Featu["mean"]+se
    Featu["Lmean"]=Featu["mean"]-se
    sdCV=Featu["CV"]*(1+1/(4*N))/(2*N)**0.5
    Featu["Ucv"]=Featu["CV"]+(sdCV*alpha95)
    Featu["Lcv"]=Featu["CV"]-(sdCV*alpha95)
    np.where(Y["obs_ci_upper"]>Y["mean"].max())[0][[1,-1]]
    R=np.where(y>(y[Featu["pos"]]-(sd*1.96)))[0]
    #R=np.where((y+se)>y[Featu["pos"]])[0]
    Featu["Upos"]=R.max()
    Featu["Lpos"]=R.min()
    return Featu


# In[48]:


import itertools
def GGnp(a,robust=False, trend=False, YAM=False):
    try:
        temp=GG(pd.Series(a,index=timeless),robust=robust, trend=trend, YAM=YAM)
    except ZeroDivisionError:
        temp=np.repeat(0,10)
    return temp
def GGGl(x,y,w=100,option={}):
    #x,y=coord
    #MSAVI.read_subregion
    Y=MSAVI.read_subregion((x,x+w),(y,y+w)).astype("float64")
    #SHAPE=list(Y.shape[:-1])+[10]
    #RES=np.zeros((w*w,10),Y.dtype)
    #Y=MSAVI[x,y,:].flatten().astype("float")
    Y[Y==-32768]=np.nan
    Y[Y==-20000]=np.nan
    Y=Y/10000
    coox=[]
    cooy=[]
    cox=[]
    coy=[]
    count=0
    y1,x1,oops=Y.shape
    RES=np.apply_along_axis(GGnp,-1,Y,**option) 
    for xx,yy in itertools.product(range(y1),range(x1)):
        #RES[count]=GG(pd.Series(Y[xx,yy],index=timeless))
        count+=1
        coox.append(xx+x)
        cox.append(xx)
        cooy.append(yy+y)
        coy.append(yy)
    fp[coox,cooy]=RES[cox,coy]
def GGG(x,y,option={}):
    #x,y=coord
    #MSAVI.read_subregion
    #Y=MSAVI.read_subregion((x,x+w),(y,y+w)).astype("float64")
    #RES=np.zeros(Y.shape,Y.dtype)
    try:
        Y=MSAVI[x,y,:].flatten().astype("float")
    except EOFError:
        print(x,y)
    Y[Y==-32768]=np.nan
    Y[Y==-20000]=np.nan
    Y=Y/10000
    return (x,y),GG(pd.Series(Y,index=timeless),**option).values


# In[67]:


#Reading stacked ENVI files with time information in bandnames
#Assuming that cell value are integer, that need to be divided by 10000, and 
MSAVI=envi.open(file+'.hdr',file+'.envi')

meta=MSAVI.metadata
Index=pd.Series(meta['band names'])


# In[68]:


#Getting Date and extracting Days since 1st january first year
#Assuming that time stamp is first token after "_" splitting, and format is Year/month/day
Index=Index.str.split("_", expand=True)
Index=Index.astype("O")
Index["Date"]=to_datetime(Index.iloc[:,0], format="%Y/%m/%d")
START=pd.to_datetime(pd.DataFrame({"year":[Index.Date[0].year],"day":[1],"month":[1]}))[0]
Index["Days"]=(Index.Date-START).apply(lambda x: x.days)
Index.set_index("Days", inplace=True)
timeless=Index.index
nyears=len(Index.Date.apply(lambda x: x.year).unique())


# In[69]:


#Setting name output
ModFeatures=["Mod_mean","Mod_CV","Mod_pos"]
Years=["year"+str(x) for x in range(1,nyears+1)]
Resilience=["anomalSD"]
Stat=["liuPvalue","liu","pvalue","rsq"]
RawFeatures=["raw_mean","raw_CV","raw_pos"]
outputparams=ModFeatures+Years+Resilience+Stat+RawFeatures


# In[70]:


#Setting input memory map
lat,long,time=MSAVI.shape

fp = np.memmap(MemFilePath, dtype='float64', mode='w+', shape=(lat,long,len(outputparams)))


# In[74]:


#Working 
startrow=0
endrow=lat
startcol=0
endcol=long
import itertools

p=Pool(cpuNode)
p.starmap(GGGl,itertools.product(range(startrow,endrow,maxchuckSize),range(startcol,endcol,maxchuckSize),[maxchuckSize],[{"YAM":True}]))
p.close()


# In[72]:


#Writing output
meta=MSAVI.metadata
meta["band names"]=outputparams
meta['bands']=str(len(meta["band names"]))
meta['data type']='4'
#del meta['description']

img = envi.create_image(outputName, meta, force=True)
mm = img.open_memmap(writable=True)
mm[:,:,:]=fp[:,:,:]

