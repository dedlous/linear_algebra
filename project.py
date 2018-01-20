import sys
import unittest
import numpy as np

from decimal import *

# def print_log(content_to_print):
#     f_handler = open('out.log', 'w')
#     f_handler.write(str(content_to_print))

f_handler=open('out.log', 'w')
sys.stdout=f_handler

def create_matrix(n):
    '''

    :param n: matrix line columb number
    :return: matrix with n lines and  columbs
    '''
    line_num = columb_numb = n
    I = []
    matrix_line = []
    for i in range(line_num):
        for j in range(columb_numb):
            matrix_line.append(1)
        I.append(matrix_line)
        matrix_line = []
    return I


def shape1(input_matrix):
    # line_num=0
    line_num=len(input_matrix)
    columb_num=0
    line=input_matrix[0]
    # columb_num=len(list(line))

    if isinstance(line,list):
        for columb in line:
            columb_num+=1
    else:
        columb_num=1

    return line_num,columb_num


def shape(input_matrix):
    line_num = len(input_matrix)
    columb_num = 0
    line = input_matrix[0]

    if isinstance(line, list):
        for columb in line:
            columb_num += 1
    else:
        columb_num = 1

    return line_num, columb_num

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound1(M, decPts=4):
    line=M[0]
    for i in range(len(M)):
        for j in range(len(line)):
            M[i][j]=round(M[i][j],decPts)
    pass

def matxRound(M, decPts=4):
    r,c=shape(M)
    for i in range(r):
        for j in range(c):
            M[i][j]=round(M[i][j],decPts)
    pass

# def trans(M):
#     line_num,columb_num=shape(M)
#     print(line_num,columb_num)
#
#     MT = [[0]  for i in range(columb_num)]
#     line_num1, columb_num1 = shape(MT)
#     print(MT)
#     print(line_num1, columb_num1)
#
#     for line in M:
#         for j in range(len(line)):
#             MT[j].append(line[j])
#             print (MT)
#
#     # print (M)
#     return None

def transpose(m):
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    m=a
    return a

# def trans2(m):
#     return zip(*m)

# def transpose(M):
#     MT = [[] for i in M[0]]
#     for i in M:
#         for j in range(len(i)):
#             MT[j].append(i[j])
#     M=MT
#
#     return None

class LinearRegressionTestCase(unittest.TestCase):

    def test_transpose(self):
        for _ in range(5):
            r, c = np.random.randint(low=1, high=25, size=2)
            matrix = np.random.random((r, c))


            mat = matrix.tolist()

            t = np.array(transpose(mat))

            # print(t.shape)
            self.assertEqual(t.shape, (c, r), "Expected shape{}, but got shape{}".format((c, r), t.shape))
            self.assertTrue((matrix.T == t).all(), 'Wrong answer')

    def test_matxMultiply(self):

        for _ in range(100):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d))
            mat2 = np.random.randint(low=-5,high=5,size=(d,c))
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1.tolist(),mat2.tolist()))
            print ('dp.shape=',dp.shape)
            self.assertEqual(dotProduct.shape, dp.shape,
                             'Wrong answer, expected shape{}, but got shape{}'.format(dotProduct.shape, dp.shape))
            self.assertTrue((dotProduct == dp).all(),'Wrong answer')

        mat1 = np.random.randint(low=-10,high=10,size=(r,5))
        mat2 = np.random.randint(low=-5,high=5,size=(4,c))
        mat3 = np.random.randint(low=-5,high=5,size=(6,c))
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
        	matxMultiply(mat1.tolist(),mat2.tolist())
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
        	matxMultiply(mat1.tolist(),mat3.tolist())

    def test_augmentMatrix(self):

        for _ in range(50):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))
            Amat = A.tolist()
            bmat = b.tolist()
            print("Amat,bmat",Amat,bmat)
            print ('shape',len(Amat),len(Amat[0]),len(bmat))

            t=augmentMatrix(Amat,bmat)
            ab = np.hstack((A, b))
            Ab = np.array(augmentMatrix(Amat,bmat))


            self.assertTrue(A.tolist() == Amat,"Matrix A shouldn't be modified")
            self.assertEqual(Ab.shape, ab.shape,
                             'Wrong answer, expected shape{}, but got shape{}'.format(ab.shape, Ab.shape))
            self.assertTrue((Ab == ab).all(),'Wrong answer')


    def test_swapRows(self):
        for _ in range(10):
            r, c = np.random.randint(low=1, high=25, size=2)
            matrix = np.random.random((r, c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0, r, size=2)
            swapRows(mat, r1, r2)

            matrix[[r1, r2]] = matrix[[r2, r1]]

            self.assertTrue((matrix == np.array(mat)).all(), 'Wrong answer')


    def test_scaleRow(self):
        for _ in range(10):
            r, c = np.random.randint(low=1, high=25, size=2)
            matrix = np.random.random((r, c))

            mat = matrix.tolist()

            rr = np.random.randint(0, r)
            with self.assertRaises(ValueError):
                scaleRow(mat, rr, 0)

            scale = np.random.randint(low=1, high=10)
            scaleRow(mat, rr, scale)
            matrix[rr] *= scale
            print (mat)
            self.assertTrue((matrix == np.array(mat)).all(), 'Wrong answer')

    def test_addScaledRow(self):

        for _ in range(10):
            r, c = np.random.randint(low=1, high=25, size=2)
            matrix = np.random.random((r, c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0, r, size=2)

            scale = np.random.randint(low=1, high=10)
            addScaledRow(mat, r1, r2, scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all(), 'Wrong answer')

    def test_gj_Solve(self):

        for _ in range(9999):
            r = np.random.randint(low=3, high=10)
            A = np.random.randint(low=-10, high=10, size=(r, r))
            b = np.arange(r).reshape((r, 1))

            x = gj_Solve(A.tolist(), b.tolist(), epsilon=1.0e-8)

            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x, None, "Matrix A is singular")
            else:
                self.assertNotEqual(x, None, "Matrix A is not singular")
                self.assertEqual(np.array(x).shape, (r, 1),
                                 "Expected shape({},1), but got shape{}".format(r, np.array(x).shape))
                Ax = np.dot(A, np.array(x))
                loss = np.mean((Ax - b) ** 2)
                self.assertTrue(loss < 0.1, "Bad result.")


test=LinearRegressionTestCase()
# test.test_transpose()

def get_n_columb(x,n):
    return [line[n] for line in x]

def matxMultiply(A, B):
    # try:
        line_a,columb_a=shape(A)
        line_b,columb_b=shape(B)
        # print (' line_a columb_b ',line_a,columb_b)
        # print ('columb_a  line_b',columb_a,line_b)
        if columb_a != line_b:
            raise ValueError
        # print ('A=',A)
        # print ('B=',B)

        return_product= [[] for i in range(line_a)]

        for i in range(line_a):
            for j in range(columb_b):
                # print ('A[i]=',A[i])
                # print ('get_n_columb(B,j)=',get_n_columb(B,j))
                # element=[]
                element=[x*y for x, y in zip(A[i], get_n_columb(B, j))]
                # for x, y in zip(A[i], get_n_columb(B, j)):
                #     element.append(x*y)
                # print ('element=',element)
                # print ('return_product',return_product)
                # print ('return_product sum',sum(element))
                #
                # print ('return_product',return_product)
                return_product[i].append(sum(element))
        print (return_product)
        return return_product
    # except Exception as e:
    #     if columb_a!=line_b:
    #         raise ValueError
    #     else:
    #         print (e)
import copy

def augmentMatrix(A, b):
    B=copy.deepcopy(A)
    for i in range(len(A)):
        B[i].extend(b[i])

    return B

def swapRows(M, r1, r2):
    M[r1],M[r2]=M[r2],M[r1]
    pass

def scaleRow(M, r, scale):
    if scale==0:
        raise ValueError
    l,c=shape(M)
    M[r]=[M[r][i]*scale for i in range(c)]
    pass

def addScaledRow(M, r1, r2, scale):

    line=copy.deepcopy(M[r2])
    scale_row=[scale]*len(M[r2])
    line=[line[i]*scale for i in range(len(line))]

    # scaleRow(line,0,scale)
    M[r1]=[M[r1][j] +i for i,j in zip(line,range(len(line)))]

    pass
# test.test_matxMultiply()
# test.test_scaleRow()
# test .test_addScaledRow()

def print_list_to_matrix(M):
    print ('print matrix+++++++++++++++++++++++++++++++++++++')
    for line in M:
        # print ('print_list_to_matrix',line)
        for e in range(len(line)):
            print('%10f'%line[e],end=' ')
        print ('\r')
    print('print matrix---------------------------------------')

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16

    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""
#
# def find_not_zero_columb(m):
#     line=[]
#     for row in m:
#         for e,index in zip(row,range(len(row))):
#             # print ('e,index',e,index)
#             if e!=0:
#                 line.append(index)
#                 # print('line',line)
#                 break
#
#     return line

def sort_matrix_by_none_zero_value(m,epsilon):
    r,c=shape(m)
    none_zero_line=[]
    print('sort_matrix_by_none_zero_value before sort ')
    print_list_to_matrix(m)
    for row in m:
        for e,index in zip(row,range(len(row))):
            # print ('e,index',e,index)
            if (not float_equal(e,0,epsilon)):
                none_zero_line.append(index)
                # print('line',line)
                break
    print ('none_zero_line=',none_zero_line)
    for i in range(len(none_zero_line)-1):
        for j in range(len(none_zero_line)-1):
            if none_zero_line[j] > none_zero_line[j + 1]:
                none_zero_line[j], none_zero_line[j + 1] = none_zero_line[j + 1], none_zero_line[j]
                swapRows(m,j,j+1)
    print ('sort_matrix_by_none_zero_value after sort ')
    print_list_to_matrix(m)
    return m

def  get_upper_triangular_matrix1(m):
    c, r = shape(m)
    sort_matrix_by_none_zero_value(m)
    scale_matrix_elemeng_to_one(m,0,0)
    print ('get_upper_triangular_matrix+++++++++++++++++++++++++++++++++++++++++++++')
    print_list_to_matrix(m)
    for i in range(c):
        # print ('i',i)
        # not_zero_list=find_not_zero_columb(m)
        # sort_matrix_by_not_zero_list(not_zero_list,m)
        for j in range (i+1):
            print ('m[i][j]:',i,j,m[i][j])
            # if i-1 <0:
            #     print ('break')
            #     break
            if j<i:
                addScaledRow(m, i, j, (-(m[i][j])))
            if j==i:
                while (m[i][j]==0):
                    print('m[i][j]==0:sort_matrix_by_none_zero_value')
                    print_list_to_matrix(m)
                    sort_matrix_by_none_zero_value(m)
                    print_list_to_matrix(m)
                scale_matrix_elemeng_to_one(m, i, j)
            # if m[i][j] !=1:
            #     print ('scale!')
            #
            #     addScaledRow(m,i,j,(-(m[i][j])))
            #     scale_matrix_elemeng_to_one(m,i,)
            print ('after addScaledRow',m[i])
            print('addScaledRow+++++++++++++++++++++++++++++++++++++++++++++')
    return m

def  get_upper_triangular_matrix(m,epsilon):
    print ('get_upper_triangular_matrix+++++++++++++++++++++++++++++++++++++')
    r, c = shape(m)
    sort_matrix_by_none_zero_value(m,epsilon)

    for j in range(c):
        for i in range(r):
            print('m[%d][%d]=%10f' % (i, j, m[i][j]))
            if i==j:
                if float_equal(m[i][j],0,epsilon):
                    return None
                scale_matrix_elemeng_to_one(m, i, j)
            elif i>j:
                addScaledRow(m, i, j, (-(m[i][j])))
        sort_matrix_by_none_zero_value(m,epsilon)
        print_list_to_matrix(m)
    return m

def float_equal(i,j,epsilon):
    return ((abs(i-j))<epsilon)

def get_diagonal_matrix(m,epsilon):
    r,c= shape(m)
    print ('r,c',r,c)
    for i in list(range(r)[::-1]):
        for j in list(range(r)[::-1]):
            print('m[%d][%d]=%10f' % (i, j, m[i][j]))
            if (m[i][j]!=0) and (i!=j):
                print('m[%d][%d]=%10f' % (i, j, m[i][j]))
                addScaledRow(m, i, j, (-(m[i][j])))
                print ('m[%d][%d]=%10f'%(i,j,m[i][j]))

    return m

def scale_matrix_elemeng_to_one(m,r,c):


    if m[r][c]!=1:
        scaleRow(m,r,1/m[r][c])
        return m

epsilon=1.0e-16

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    c,r = shape(A)
    c_b ,r_b =shape(b)
    if c!= c_b:
        return None
    print ('start solve 000000000000000000000000000000000000000000000000000000000000000000000')
    m= augmentMatrix(A,b)
    print_list_to_matrix(m)

    # not_zero_list=find_not_zero_columb(m)
    # print ('not_zero_list')
    # print (not_zero_list)
    #
    # m=sort_matrix_by_none_zero_value(m,epsilon)
    # print_list_to_matrix(m)

    m=get_upper_triangular_matrix(m,epsilon)
    if m==None:
        return None
    print_list_to_matrix(m)

    m=get_diagonal_matrix(m,epsilon)
    # for i in range(c):
    #     print ('------------------------------------------------------scale_matrix_elemeng_to_one')
    #     scale_matrix_elemeng_to_one(m,i)
    #     print_list_to_matrix(m)
    print_list_to_matrix(m)
    matxRound(m,decPts)
    print_list_to_matrix(m)
    return_list=[[line[c]] for line in m]

    print (return_list)
    return return_list

from helper import *

from helper import *
from matplotlib import pyplot as plt
# %matplotlib inline

X,Y = generatePoints(seed=9999,num=100)

# ## 可视化
# plt.xlim((-5,5))
# plt.xlabel('x',fontsize=18)
# plt.ylabel('y',fontsize=18)
# plt.scatter(X,Y,c='b')
# plt.show()
#
#
# m1 = 3.22
# b1 = 7.2
#
# # 不要修改这里！
# plt.xlim((-5,5))
# x_vals = plt.axes().get_xlim()
# y_vals = [m1*x+b1 for x in x_vals]
# plt.plot(x_vals, y_vals, '-', color='r')
#
# plt.xlabel('x',fontsize=18)
# plt.ylabel('y',fontsize=18)
# plt.scatter(X,Y,c='b')
#
# plt.show()

def calculateMSE(X,Y,m,b):
    se=[ (x*m+b-y)**2 for x,y in zip(X,Y)]
    sum=0
    b=len(se)
    for i in se:
        sum=sum +i

    return sum/b

# print(calculateMSE(X,Y,m1,b1))
# TODO 实现线性回归
'''
参数：X, Y 存储着一一对应的横坐标与纵坐标的两个一维数组
返回：m，b 浮点数
'''
def linearRegression(X,Y):

    x_list=[[e] for e in X]
    print(x_list)
    b= [[1] for _ in X]
    print (b)

    x_list=augmentMatrix(x_list,b)
    print (x_list)

    x_list_t=transpose(x_list)
    print (x_list_t)

    y_list=[[e] for e in Y]

    a,b=gj_Solve(matxMultiply(x_list_t,x_list),matxMultiply(x_list_t,y_list))

    return a[0],b[0]

m2,b2 = linearRegression(X,Y)
assert isinstance(m2,float),"m is not a float"
assert isinstance(b2,float),"b is not a float"
print(m2,b2)

# test.test_gj_Solve()

# test.test_transpose()