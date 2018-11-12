import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import random

xy=[]

maxs=0.15

lineXend=1
lineYend=2
lineXstart=0
lineYstart=0

maxs=maxs*2
h=math.sqrt(maxs)
nx=int((lineXend)/h+1)

ny=int((lineYend)/h+1)


for i in range(nx):
    for j in range(ny):
        if(h*i>lineXstart and h*j>lineYstart):
            xy.append([h*i,h*j])

xy.append([lineXend,lineYend])
xy.append([lineXstart,lineYstart])
xy.append([lineXstart,lineYend])
xy.append([lineXend,lineYstart])

for i in range(0,nx):
    if(h*i>lineXstart):
        xy.append([h*i,lineYend])
        xy.append([h*i,lineYstart])
for i in range(0,ny):
    if(h*i>lineYstart):
        xy.append([lineXend,h*i])
        xy.append([lineXstart,h*i])

def SortCoord(lst):
    a=lst[0]
    b=lst[1]
    c=lst[2]
    right=a
    left=b
    res=1
    for item in lst:
        if(item[0]>right[0]):
            right=item
        if(item[0]<left[0]):
            left=item
    for item in lst:
        if(item!=left and item!=right):
            middle=item
    #print(left,' ',middle,' ',right)
    res=((middle[0]-left[0])*(right[1]-left[1])-(middle[1]-left[1])*(right[0]-left[0]))
    if(res>0):
        t=[right,left,middle]
    if(res<0):
        t=[right,middle,left]
    return t


xy = np.array(xy)

#x=np.array(xy[:,0])
#y=np.array(xy[:,1])

layer1x=[]
layer1y=[]

for elem in xy:
    layer1x.append(elem[0])
    layer1y.append(elem[1])

#Xview=0.5
#Yview=0.5

#Xview=5
#Yview=5

triang = Delaunay(np.array([layer1x,layer1y]).T,)

alltri1 = triang.simplices.tolist()
allpoints1 = triang.points.tolist()


for t in alltri1:
    temp=[[allpoints1[t[0]][0],allpoints1[t[0]][1]],
          [allpoints1[t[1]][0],allpoints1[t[1]][1]],
          [allpoints1[t[2]][0],allpoints1[t[2]][1]]]
    sort=SortCoord(temp)
    triang.points.tolist()[t[0]][0]=sort[0][0]
    triang.points.tolist()[t[0]][1]=sort[0][1]
    triang.points.tolist()[t[1]][0]=sort[1][0]
    triang.points.tolist()[t[1]][1]=sort[1][1]
    triang.points.tolist()[t[2]][0]=sort[2][0]
    triang.points.tolist()[t[2]][1]=sort[2][1]


allpoints1= triang.points.tolist()

Squares=[]

layer2x=[]
layer2y=[]



for t in alltri1:
    layer2x.append(((allpoints1[t[0]][0]+allpoints1[t[1]][0]+allpoints1[t[2]][0]))/3)
    layer2y.append(((allpoints1[t[0]][1]+allpoints1[t[1]][1]+allpoints1[t[2]][1]))/3)
    Squares.append(((allpoints1[t[0]][0]*allpoints1[t[1]][1]+allpoints1[t[1]][0]*allpoints1[t[2]][1]
                     +allpoints1[t[2]][0]*allpoints1[t[0]][1])-(allpoints1[t[0]][1]*allpoints1[t[1]][0]
                        +allpoints1[t[1]][1]*allpoints1[t[2]][0]+allpoints1[t[2]][1]*allpoints1[t[0]][0]))*0.5)

s=0
for i in Squares:
    s=s+i
print(s)
print()


layer2x=np.array(layer2x)
layer2y=np.array(layer2y)

for i in range(len(allpoints1)):
    plt.text(allpoints1[i][0],allpoints1[i][1],i,color='black')

for i in range(len(layer2x)):
    #plt.text(layer2x[i],layer2y[i],i+1,color='green' ,bbox=dict(facecolor='red', alpha=0.5))
    plt.text(layer2x[i],layer2y[i],i+1,color='white')

#for t in alltri1:
#    print('[',allpoints1[t[0]][0],',',allpoints1[t[0]][1],'](',t[0],')\n[',
#          allpoints1[t[1]][0],',',allpoints1[t[1]][1],'](',t[1],')\n[',
#          allpoints1[t[2]][0],',',allpoints1[t[2]][1],'](',t[2],')\n')

#---------------------------- Сhappter 2 Start -----------------------------------------------------#
iterator=0

#Параметри нашого варіанту з листків
a11=4
a12=0
a21=0
a22=3
d=1
fxi1=2
fxj2=2
fxm3=2
f=2

n=len(allpoints1)
k=len(alltri1)
print('К-сть точок= ',n,'  К-сть трикутників=',)

#Ініціалізація вектора матриць
Matrix=[[0] * n] * n
#Ініціалізація вектора векторів
rpart=[[0] * n]

#Вектор масивів K
K=[]

#Вектор масивів M
M=[]

#Вектор масивів Q
Q=[]

#ітеруємось по кожному трикутнику
for t in alltri1:
    #використовуючи формулу з формул базисних векторів перераховуємо
    bi1=allpoints1[t[1]][1]-allpoints1[t[2]][1]
    bj2=allpoints1[t[2]][1]-allpoints1[t[0]][1]
    bm3=allpoints1[t[0]][1]-allpoints1[t[1]][1]
    ci1=allpoints1[t[2]][0]-allpoints1[t[1]][0]
    cj2=allpoints1[t[0]][0]-allpoints1[t[2]][0]
    cm3=allpoints1[t[1]][0]-allpoints1[t[0]][0]

    #Ініціалізація  за ось цими формулами використовуючи обчислене зверху та переметри нашого варіанту
    K.append([
        [ a11*bi1**2+a22*ci1**2, a11*bi1*bj2+a22*ci1*cj2, a11*bi1*bm3+a22*ci1*cm3],
        [ a11*bi1*bj2+a22*ci1*cj2, a11*bj2**2+a22*cj2**2, a11*bj2*bm3+a22*cj2*cm3],
        [ a11*bi1*bm3+a22*ci1*cm3, a11*bj2*bm3+a22*cj2*cm3, a11*bm3**2+a22*cm3**2]
        ])
    #Ініціалізація  константами по формулі
    M.append([
        [2,1,1],
        [1,2,1],
        [1,1,2]
        ])

    #Ініціалізація  константами по формулі
    Q.append(
        [1,1,1]
        )


    #Тепер для кожного проініціалізованого вектора чи матриці ковбасимо наступне
    for z1 in range(3):
        for z2 in range(3):
            #Ітеруючись по кожному елементу матриці
            # Ітеруючись по кожному елементу вектора домножуємо на коеф який вираховується як :  помножити на Дельта (2 * Площа трик.) /2
            K[iterator][z1][z2]=K[iterator][z1][z2]*(1/(2*Squares[iterator]*2))
            # Ітеруючись по кожному елементу вектора домножуємо на коеф який вираховується як :  помножити на d (задано) * на Дельта (2 * Площа трик.) /24
            M[iterator][z1][z2]=M[iterator][z1][z2]*(d*Squares[iterator]*2/24)

        # Ітеруючись по кожному елементу вектора домножуємо на коеф який вираховується як :  помножити на f (задано) * на Дельта (2 * Площа трик.) /6
        Q[iterator][z1]=Q[iterator][z1]*(Squares[iterator]*2*f/6)
    iterator=iterator+1

#print(K)
#print()
#print(M)
#print()
#print(Q)

print()
print('----------------- Друкуємо вектор матриць K ------------------')
print()
for i in K:
    for j in range(len(i)):
        print(i[j])
    print('\n')
print('----------------- Друкуємо вектор матриць M ------------------')
print()
for i in M:
    for j in range(len(i)):
        print(i[j])
    print('\n')
print('----------------- Друкуємо вектор векторів Q ------------------')
print()
for i in Q:
    for j in range(len(i)):
        print(i[j])
    print('\n')

#for t in alltri1:



#-------------------- Chapter 2 End ----------------------------------------#


colors=[]
for i in range(len(triang.simplices)):
    colors.append(random.randint(0,100))


plt.tripcolor(layer1x,layer1y,triang.simplices.copy(), colors, 'k')


plt.show()

input()

