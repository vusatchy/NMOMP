import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import random
import scipy
import copy
####################################################
####################################################
import copy


def gaussFunc(a):
    eps = 1e-16

    c = np.array(a)

    a = np.array(a)

    len1 = len(a[:, 0])

    len2 = len(a[0, :])

    vectB = copy.deepcopy(a[:, len1])

    for g in range(len1):

        max = abs(a[g][g])

        my = g

        t1 = g

        while t1 < len1:

            if abs(a[t1][g]) > max:
                max = abs(a[t1][g])

                my = t1

            t1 += 1

        if abs(max) < eps:
            raise DetermExeption("Check determinant")

        if my != g:
            # a[g][:], a[my][:] = a[my][:], a[g][:]

            # numpy.swapaxes(a, 1, 0)

            b = copy.deepcopy(a[g])

            a[g] = copy.deepcopy(a[my])

            a[my] = copy.deepcopy(b)

        amain = float(a[g][g])

        z = g

        while z < len2:
            a[g][z] = a[g][z] / amain

            z += 1

        j = g + 1

        while j < len1:

            b = a[j][g]

            z = g

            while z < len2:
                a[j][z] = a[j][z] - a[g][z] * b

                z += 1

            j += 1

    a = backTrace(a, len1, len2)

    return a


class DetermExeption(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def backTrace(a, len1, len2):
    a = np.array(a)

    i = len1 - 1

    while i > 0:

        j = i - 1

        while j >= 0:
            a[j][len1] = a[j][len1] - a[j][i] * a[i][len1]

            j -= 1

        i -= 1

    return a[:, len2 - 1]


def vectorN(c, a, len1, vectB):
    c = np.array(c)

    a = np.array(a)

    vectB = np.array(vectB)

    b = np.zeros((len1))

    i = 0

    while i < len1:

        j = 0

        while j < len1:
            b[i] += c[i][j] * a[j]

            j += 1

        i = i + 1

    c = copy.deepcopy(b)

    print("!")

    for i in range(len1):
        c[i] = abs(c[i] - vectB[i])

    return c


# ---------------------------- Сhappter 1 Start -----------------------------------------------------#

xy = []

maxs = 0.55

lineXend = 1
lineYend = 2
lineXstart = 0
lineYstart = 0

maxs = maxs * 2
h = math.sqrt(maxs)
nx = int((lineXend) / h + 1)

ny = int((lineYend) / h + 1)

for i in range(nx):
    for j in range(ny):
        if (h * i > lineXstart and h * j > lineYstart):
            xy.append([h * i, h * j])

xy.append([lineXend, lineYend])
xy.append([lineXstart, lineYstart])
xy.append([lineXstart, lineYend])
xy.append([lineXend, lineYstart])

for i in range(0, nx):
    if (h * i > lineXstart):
        xy.append([h * i, lineYend])
        xy.append([h * i, lineYstart])
for i in range(0, ny):
    if (h * i > lineYstart):
        xy.append([lineXend, h * i])
        xy.append([lineXstart, h * i])


def SortCoord(lst):
    a = lst[0]
    b = lst[1]
    c = lst[2]
    right = a
    left = b
    res = 1
    for item in lst:
        if (item[0] > right[0]):
            right = item
        if (item[0] < left[0]):
            left = item
    for item in lst:
        if (item != left and item != right):
            middle = item
    # print(left,' ',middle,' ',right)
    res = ((middle[0] - left[0]) * (right[1] - left[1]) - (middle[1] - left[1]) * (right[0] - left[0]))
    if (res > 0):
        t = [right, left, middle]
    if (res < 0):
        t = [right, middle, left]
    return t


xy = np.array(xy)

# x=np.array(xy[:,0])
# y=np.array(xy[:,1])

layer1x = []
layer1y = []

for elem in xy:
    layer1x.append(elem[0])
    layer1y.append(elem[1])

# Xview=0.5
# Yview=0.5

# Xview=5
# Yview=5

triang = Delaunay(np.array([layer1x, layer1y]).T, )

alltri1 = triang.simplices.tolist()
allpoints1 = triang.points.tolist()

for t in alltri1:
    temp = [[allpoints1[t[0]][0], allpoints1[t[0]][1]],
            [allpoints1[t[1]][0], allpoints1[t[1]][1]],
            [allpoints1[t[2]][0], allpoints1[t[2]][1]]]
    sort = SortCoord(temp)
    triang.points.tolist()[t[0]][0] = sort[0][0]
    triang.points.tolist()[t[0]][1] = sort[0][1]
    triang.points.tolist()[t[1]][0] = sort[1][0]
    triang.points.tolist()[t[1]][1] = sort[1][1]
    triang.points.tolist()[t[2]][0] = sort[2][0]
    triang.points.tolist()[t[2]][1] = sort[2][1]

allpoints1 = triang.points.tolist()

Squares = []

layer2x = []
layer2y = []

for t in alltri1:
    layer2x.append(((allpoints1[t[0]][0] + allpoints1[t[1]][0] + allpoints1[t[2]][0])) / 3)
    layer2y.append(((allpoints1[t[0]][1] + allpoints1[t[1]][1] + allpoints1[t[2]][1])) / 3)
    Squares.append(((allpoints1[t[0]][0] * allpoints1[t[1]][1] + allpoints1[t[1]][0] * allpoints1[t[2]][1]
                     + allpoints1[t[2]][0] * allpoints1[t[0]][1]) - (allpoints1[t[0]][1] * allpoints1[t[1]][0]
                                                                     + allpoints1[t[1]][1] * allpoints1[t[2]][0] +
                                                                     allpoints1[t[2]][1] * allpoints1[t[0]][0])) * 0.5)

s = 0
for i in Squares:
    s = s + i
print(s)
print()

layer2x = np.array(layer2x)
layer2y = np.array(layer2y)

for i in range(len(allpoints1)):
    plt.text(allpoints1[i][0], allpoints1[i][1], i, color='black')

for i in range(len(layer2x)):
    # plt.text(layer2x[i],layer2y[i],i+1,color='green' ,bbox=dict(facecolor='red', alpha=0.5))
    plt.text(layer2x[i], layer2y[i], i + 1, color='white')

# for t in alltri1:
#    print('[',allpoints1[t[0]][0],',',allpoints1[t[0]][1],'](',t[0],')\n[',
#          allpoints1[t[1]][0],',',allpoints1[t[1]][1],'](',t[1],')\n[',
#          allpoints1[t[2]][0],',',allpoints1[t[2]][1],'](',t[2],')\n')
# ---------------------------- Сhappter 1 End -------------------------------------------------------#
# ---------------------------- Сhappter 2 Start -----------------------------------------------------#
iterator = 0

# Параметри нашого варіанту з листків
a11 = 3
a12 = 0
a21 = 0
a22 = 2
d = 1
fxi1 = 2
fxj2 = 2
fxm3 = 2
f = 2

n = len(allpoints1)
k = len(alltri1)
print('n= ', n, '  k=', k)
print()
Matrix = []
rpart = []

for i in range(n):
    Matrix.append([])
for i in range(n):
    for j in range(n):
        Matrix[i].append(0)
    rpart.append(0)
# Matrix=np.array(Matrix)
# rpart=np.array(rpart)
# print(Matrix)
# print(rpart)

KM = []
K = []
M = []
Q = []

# ітеруємось по кожному трикутнику
for t in alltri1:
    # використовуючи формулу з формул базисних векторів перераховуємо
    bi1 = allpoints1[t[1]][1] - allpoints1[t[2]][1]
    bj2 = allpoints1[t[2]][1] - allpoints1[t[0]][1]
    bm3 = allpoints1[t[0]][1] - allpoints1[t[1]][1]
    ci1 = allpoints1[t[2]][0] - allpoints1[t[1]][0]
    cj2 = allpoints1[t[0]][0] - allpoints1[t[2]][0]
    cm3 = allpoints1[t[1]][0] - allpoints1[t[0]][0]

    # Ініціалізація  за ось цими формулами використовуючи обчислене зверху та переметри нашого варіанту
    K.append([
        [a11 * bi1 ** 2 + a22 * ci1 ** 2, a11 * bi1 * bj2 + a22 * ci1 * cj2, a11 * bi1 * bm3 + a22 * ci1 * cm3],
        [a11 * bi1 * bj2 + a22 * ci1 * cj2, a11 * bj2 ** 2 + a22 * cj2 ** 2, a11 * bj2 * bm3 + a22 * cj2 * cm3],
        [a11 * bi1 * bm3 + a22 * ci1 * cm3, a11 * bj2 * bm3 + a22 * cj2 * cm3, a11 * bm3 ** 2 + a22 * cm3 ** 2]
    ])
    # Ініціалізація  константами по формулі
    M.append([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])
    # Ініціалізація  константами по формулі
    Q.append(
        [1, 1, 1]
    )
    # Тепер для кожного проініціалізованого вектора чи матриці ковбасимо наступне
    for z1 in range(3):
        for z2 in range(3):
            # Ітеруючись по кожному елементу матриці
            # Ітеруючись по кожному елементу вектора домножуємо на коеф який вираховується як :  помножити на Дельта (2 * Площа трик.) /2
            K[iterator][z1][z2] = K[iterator][z1][z2] * (1 / ((2 * Squares[iterator]) ** 2))
            # Ітеруючись по кожному елементу вектора домножуємо на коеф який вираховується як :  помножити на d (задано) * на Дельта (2 * Площа трик.)
            M[iterator][z1][z2] = M[iterator][z1][z2] * (d * Squares[iterator] * 2 / 24)
        # Ітеруючись по кожному елементу вектора домножуємо на коеф який вираховується як :  помножити на f (задано) * на Дельта (2 * Площа трик.) /6
        Q[iterator][z1] = Q[iterator][z1] * (Squares[iterator] * 2 * f / 6)
    temp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            temp[i][j] = K[iterator][i][j] + M[iterator][i][j]
    KM.append(temp)
    iterator = iterator + 1

# -------------------- Chapter 2 End ------------------------------------------#
# -------------------- Chapter 3 Start ----------------------------------------#






iterator = 0
for t in alltri1:
    # ітеруємось по трикутниках
    for i in range(3):
        for j in range(3):
            # за номерами вершин трикутників додаємо до попередніх
            # і додаємо KM матрицю матриць (обчислена з матриць K і M з частини 2)
            Matrix[t[i]][t[j]] = Matrix[t[i]][t[j]] + KM[iterator][i][j]
            Matrix1 = np.array(Matrix)
        # будуємо вектор розвязок по вершинах трикутників
        # і матриці Q яку ми обчислили в частині 2
        rpart[t[i]] = rpart[t[i]] + Q[iterator][i]
    # print('-----------------------Matrix-------------------------')
    #        print(Matrix1)
    iterator = iterator + 1

# print(Matrix)
# iterator=0
# for t in alltri1:
#    e00=KM[iterator][0][0]
#    e01=KM[iterator][0][1]
#    e02=KM[iterator][0][2]
#    e10=KM[iterator][1][0]
#    e11=KM[iterator][1][1]
#    e12=KM[iterator][1][2]
#    e20=KM[iterator][2][0]
#    e21=KM[iterator][2][1]
#    e22=KM[iterator][2][2]

#    ei=int(t[0])
#    ej=int(t[1])
#    em=int(t[2])

#    Matrix[ei][ei]=Matrix[ei][ei]+e00
#    Matrix[ei][ej]=Matrix[ei][ej]+e01
#    Matrix[ei][em]=Matrix[ei][em]+e02
#    Matrix[ej][ei]=Matrix[ej][ei]+e10
#    Matrix[ej][ej]=Matrix[ej][ej]+e11
#    Matrix[ej][em]=Matrix[ej][em]+e12
#    Matrix[em][ei]=Matrix[em][ei]+e20
#    Matrix[em][ej]=Matrix[em][ej]+e21
#    Matrix[em][em]=Matrix[em][em]+e22

#    KM1=np.array(KM[iterator])
#    Matrix1=np.array(Matrix)
#    print('-----------------------t-------------------------')
#    print(t[0],',',t[1],',',t[2])
#    print('-----------------------KM-------------------------')
#    print(KM1)
#    print('-----------------------Matrix-------------------------')
#    print(Matrix1)
#    iterator=iterator+1

KM = np.array(KM)
K = np.array(K)
M = np.array(M)
Q = np.array(Q)

Matrix = np.array(Matrix)
rpart = np.array(rpart)

# Робимо матрицю 2 для експерементів з власним Гаусом
Matrix2 = []

for i in range(n):
    Matrix2.append([])
for i in range(n):
    for j in range(n + 1):
        Matrix2[i].append(0)

for i in range(n):
    for j in range(n):
        Matrix2[i][j] = Matrix[i][j]
for i in range(n):
    Matrix2[i][n] = rpart[i]

# print(Matrix2)
# Власний Гаус для перевірки з біблотечним
RES_2 = gaussFunc(Matrix2)

# переводимо в numpy формат
Matrix = np.array(Matrix)
rpart = np.array(rpart)

# Розвязок з використанням бібліотечного Гауса
RES = np.linalg.solve(Matrix, rpart)

# print('-----------------------K-------------------------')
# print(K)
# print('-----------------------KM-------------------------')
# print(KM)
# print('-----------------------Matrix-------------------------')
print(Matrix)
print('-----------------------rpart-------------------------')
print(rpart)
print('-----------------------DET-------------------------')
# print(scipy.linalg.det(Matrix))
print('-----------------------RES-------------------------')
print(RES)
print('-----------------------RES2-------------------------')
print(RES_2)

# print()
# print('-----------------KKKKKKK------------------')
# print()
# for i in K:
#    for j in range(len(i)):
#         print(i[j])
#    print('\n')
# print('-----------------MMMMMMM------------------')
# print()
# for i in M:
#    for j in range(len(i)):
#        print(i[j])
#    print('\n')
# print('-----------------QQQQQQQ------------------')
# print()
# for i in Q:
#    for j in range(len(i)):
#        print(i[j])
#    print('\n')

# for t in alltri1:



# -------------------- Chapter 3 End ----------------------------------------#


colors = []
for i in range(len(triang.simplices)):
    colors.append(random.randint(0, 100))

plt.tripcolor(layer1x, layer1y, triang.simplices.copy(), colors, 'k')

plt.show()

input()
