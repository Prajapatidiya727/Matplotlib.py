#import matplotlib.pyplot as plt
#%matplotlib inline


#apl_price=[45,23,78,99,64,55,77,99]
#ms_pric=[78,96,43,52,79,63,74,85,85]
#year=[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]


#plt.plot(year , apl_price)
#plt.show()


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd





#Functional plot
x_1=np.linspace(0,5,10)
y_1=x_1**2

plt.plot  (x_1,y_1)
plt.title('Days Squared Chart')
plt.xlabel('Days')
plt.ylabel('Squared Days')
plt.show()


# print multiple plots
plt.subplot(1,2,1)
plt.plot(x_1,y_1,'b')
plt.subplot(1,2,2)
plt.plot(x_1,y_1,'g')



# using figure objects
fig_1 = plt.figure(figsize=(5,4),dpi=100)#dpi means dots per inch
axes_1 = fig_1.add_axes([0.15,0.15,0.9,0.9])
axes_1.set_xlabel('days')
axes_1.set_ylabel('days squred')
axes_1.set_title('days squred chart')
axes_1.plot(x_1,y_1,label='x/x2')
axes_1.plot(y_1,x_1,label='x2/x')
axes_1.legend(loc=0)


axes_2= fig_1.add_axes([0.45,0.45,0.4,0.3])
axes_2.set_xlabel('days')
axes_2.set_ylabel('days squred')
axes_2.set_title('days squred chart')
axes_2.plot(x_1,y_1,'r')

axes_2.text(0,40,'message')



# subplots
fig_2,axes_2= plt.subplots(figsize=(8,4),nrows=1,ncols=3)
plt.tight_layout()
axes_2[1].set_title('plot 2')
axes_2[1].set_xlabel('x')
axes_2[1].set_ylabel('x Squared')
axes_2[1].plot(x_1,y_1)


#fig_3 = plt.figure(figsize=(6,4))
#axes_3=fig_3.add_axes([0,0,1,1])
#axes_3.plot(x_1,y_1,color='navy',alpha=75,lw=2,ls='_',marker='o',markerize=7,markerfacecolor='y',markeredgecolor='y',markeredgewidth=4)

#save visualization to a file
#fig_3.savefile

#WORKING  WITH PANDAS DATAFRAM


from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.plot([1,2,4],[3,5,1])
plt.title('info')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
x=[5,8,10]
y=[12,6,6]
x2=[6,9,11]
y2=[6,15,7]
plt.plot(x,y,'g',label='line one',linewidth=5)
plt.plot(x2,y2,'c',label='line two',linewidth=5)
plt.title('epic info')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.grid(True ,color='k')
plt.show()



# Bar plot
plt.bar([11,8,5,6,7],[5,6,9,7,4],label='example one',color='b')
plt.bar([2,9,6,7,4],[4,6,11,9,5],label='example two',color='g')
plt.title('info')
plt.xlabel('bar numbers',color='b')
plt.ylabel('bar heights',color='b')
plt.legend()
plt.grid(True,color='k')
plt.show()

# Histo plot
population_ages=[22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population_ages,bins,histtype='bar',rwidth=0.9)
plt.xlabel('x Axis')
plt.ylabel('y Axis')
plt.title('Histogram')
plt.show()

# Sactter Plot
x=[1,2,3,4,5,6,7,8]
y=[5,2,4,2,1,4,5,2]
plt.scatter(x,y,label='python',color='k')
plt.title('info')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()

# Stack plot
days=[1,2,3,4,5]
sleeping=[7,8,6,11,7]
eating=[2,3,4,3,2]
working=[7,8,7,2,2]
playing=[8,5,7,8,13]
plt.plot([],[],color='g',label='sleeping',linewidth=5)
plt.plot([],[],color='c',label=' eating',linewidth=5)
plt.plot([],[],color='r',label=' working',linewidth=5)
plt.plot([],[],color='k',label=' playing',linewidth=5)
plt.stackplot(days,sleeping,eating,working,playing,colors=['g','c','r','k'])
plt.title('info')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()

# Pie Plot
slices=[7,11,150,113]
activities=['sleeping','eating','working','playing']
cols=['c','m','r','b']
plt.pie(slices,labels=activities,colors=cols,startangle=90,shadow=True,explode=[0,0.1,0,0],autopct='%1.1f%%')
plt.title('pie plot')
plt.show()

#line plot
import numpy as np

def f(t):
  return np.exp(-t)*np.cos(2*np.pi*t)
t1=np.arange(0.0,5.0,0.1)
t2=np.arange(0.0,5.0,0.02)
plt.subplot(2,1,1)
plt.plot(t1,f(t1),'bo',t2,f(t2))
plt.subplot(2,1,2)
plt.plot(t2,np.cos(2*np.pi*t2))
plt.show()

import pandas as pd


# Basic Graph
x=[0,1,2,3,4]
y=[0,2,4,6,8]
plt.plot(x,y,linewidth=3,marker='.',linestyle='--',markersize=20,markeredgecolor='b')



plt.show()

from matplotlib import pyplot as plt
a=[0,-100,25,67,-323]
b=[0,3,7,3,9]
plt.plot(a,b)

plt.axis([-50,80,2,8])
plt.plot(a,b)

from matplotlib import pyplot as plt
import numpy as np


x=np.linspace(0,25,30)
y=np.sin(x)+0.1*np.random.randn(len(x))
plt.plot(x,y,'o--',color='purple',lw=1,ms=5)#ls will change size of line,ms will change size of dots and '--' will make doted line and'o' will make dots at the place where function is intersacting.



plt.figure(figsize=(8,3))#Figure size will change size of this box
plt.plot(x,y,'o--',color='red',lw=0.7,ms=5,label='component 1')
plt.legend(loc='upper right',fontsize=15)#legend is to show name of ploted thing and loc means location which shows that where will the legend be placed.Fontsize means size og legend.
plt.grid(True,color='k')

#x2=np.linespace(0,15,100)
#y2.sin(x2)

#plt.figure(figsize=(8,3))
#plt.plot(x,y,'o--',color='red',lw=0.7,ms=5,label='component 1')
#plt.plot(x2,y2,'--',color='green',lw=0.7,ms=5,label='component 2')

from matplotlib import pyplot as plt
import numpy as np
x2= np.linspace(0,20,100)
y2=np.sin(x2)
plt.figure(figsize=(8,5))
plt.plot(x,y,'o--',color='purple',lw=0.9,ms=5,label='Data')
plt.plot(x2,y2,color='green',lw=0.9,ms=5,label='Theory')
plt.legend(loc='upper right',fontsize=15,ncol=2)
plt.grid(True,color='k')
plt.legend()
plt.xlabel('voltage[v]')
plt.ylabel('Time[s]')

 # Histograms
res = np.random.rand(1000)*0.2+0.4
res

# Quick Histogram
plt.figure(figsize=(8,3))
plt.hist(res,bins=30,density=True)

res2=np.random.rand(1000)*0.2+0.4

#change num of bins
plt.figure(figsize=(8,3))
plt.hist(res,bins=30,density=True,histtype='step')
plt.hist(res2, bins=30,density=True,histtype='step')# if we will change type from normal to step type this will look like this.
plt.show()# How to plot 2 plots togather.

#single Axes in one Figure.
fig,ax=plt.subplots(1,1,figsize=(12,4))
ax.plot(x,y,'o--')
ax.set_xlabel('asd')
ax.set_ylabel('asd mahina')
plt.show()



#fig.ax=plt.subplots(3,2,figsize=(12,12))
#ax= axes[0][0]
#ax.plot(x,y,'o--')
#ax.set_xlabel('asd')
#ax.set_ylabel('asd mahina')
#ax= axes[1][1]
#ax.hist(res,bins=30,density=True,histtype='step')
#ax.set_xlabl('asd')
#ax.set_ylabel('asd mahina')
#plt.show()

res_a1=0.2*np.random.randn(1000)+0.4
res_b1=0.25*np.random.randn(1000)+0.4
res_a2=0.21*np.random.randn(1000)+0.3
res_b2=0.22*np.random.randn(1000)+0.3

plt.hist(res_a1,bins=100)
plt.hist(res_b1,bins=100)
plt.show()

#fig,axes=plt.subplots(1,2,figsize=(10,3.5))
#ax=axes[0]# This 0 will show plot in firt axes maeans 0 axes.
#ax.hist(res_a1,bins=30,density=True,histtype='step',label='a')
#ax.hist(res_b1,bins=30,density=True,histtype='step',label='b')
#ax.legend(loc='upper right')
#plt.show()


# 2D Plot
_=np.linspace(-1,1,100)
x,y=np.meshgrid(_,_)
z=x**2+x*y
plt.show()

#Filled in contour plots
plt.contourf(x,y,z,levels=30,vmax='0.7',cmap='plasma')# plasma is color map which color you like for that there is one tabel of color in google.
plt.colorbar(label='asd')#plt.contourf will show you colored plot
plt.show()


cs=plt.contour(x,y,z,levels=30)#only plt.contour will show you lined plot
plt.clabel(cs,fontsize=8)# This will add value inside the lines

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot_surface(x,y,z,cmap='coolwarm')

#stream plots
w=3
_=np.meshgrid(_,_)
U=-1-x**2+y
V=-1+x-y**2
speed=np.sqrt(U**2+V**2)


fig,axes=plt.subplots(2,2,figsize=(7,7))
ax=axes[0][0]
ax.streamplot(x,y,U,V)
ax=axes[0][1]
ax.streamplot(x,y,U,V,color=speed)
ax=axes[1][0]
lw=5*speed/speed.max()
ax.streamplot(x,y,U,V,linewidth=lw)
ax=axes[1][1]
lw=5*speed/speed.max()
seedpoints=np.array([[0,1],[1,0]])
ax.streamplot(x,y,U,V,start_points=seedpoints)


#image reading
# in image reading you just need to write im=plt.imread(and then csv file name) and then plt.imshow(im)


#Animations
def f(x,t):
  return np.sin(x-3*t)

x=np.linspace(0,10*np.pi,1000)


plt.plot(x,f(x,0))
plt.plot(x,f(x,0.1))








