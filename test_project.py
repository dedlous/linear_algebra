# import math
# # import sys
# #
# # f_handler=open('out.log', 'w')
# # sys.stdout=f_handler
# # a=[]
# # element=[-9, -7, -7, 9, -2, 2, -1, 1, 2, -5, -2, -4, -9, -5, 6, -6, -7, -8, 2, -1, 7]
# # s = 0
# # for x in element:
# #     s += x
# #     print('sum', s)
# #     print(s)
# #     a.append(s)
# #     print (a)
#
# # print ('%4f'%3.15871544444)
# #
# # print('PI=%20f'%math.pi)
# #
# # print ([j for j in list(range(10)[::-1])])
# #
# # epsilon=1.0e-16
# # print (abs(1.0000000000000000000005-0)>epsilon)
# #
# # array = [1,3,2,6,0,5,4]
# # for i in range(len(array)):
# #     print ('array[i]=',array[i])
# #     for j in range(i):
# #         print ('array[j]=array[%d]=%d'%(j,array[j]))
# #         if array[j]> array[j + 1]:
# #             array[j], array[j + 1] = array[j + 1], array[j]
# #             print(array)
#
#
# def bubbleSort(myList):
#     # 首先获取list的总长度,为之后的循环比较作准备
#     length = len(myList)
#
#     # 一共进行几轮列表比较,一共是(length-1)轮
#     for i in range(0, length - 1):
#
#         # 每一轮的比较,注意range的变化,这里需要进行length-1-长的比较,注意-i的意义(可以减少比较已经排好序的元素)
#         for j in range(0, length - 1 - i):
#             print(myList)
#             print ('myList[j],myList[j+1]',myList[j],myList[j+1])
#             # 交换
#
#             if myList[j] > myList[j + 1]:
#                 tmp = myList[j]
#                 myList[j] = myList[j + 1]
#                 myList[j + 1] = tmp
#                 print(myList)
#         print (myList)
#         # # 打印每一轮交换后的列表
#         # for item in myList:
#         #     print(item)
#         # print("=============================")
#
#
# print("Bubble Sort: ")
# myList = [1,8, 4,3,9,7, 5, 0, 6,2]
# bubbleSort(myList)


#
# def sort_matrix_by_none_zero_value(m,epsilon):
#     r,c=shape(m)
#     none_zero_line=[]
#
#     for row in m:
#         for e,index in zip(row,range(len(row))):
#
#             if (not float_equal(e,0,epsilon)):
#                 none_zero_line.append(index)
#
#                 break
#
#     for i in range(len(none_zero_line)-1):
#         for j in range(len(none_zero_line)-1):
#             if none_zero_line[j] > none_zero_line[j + 1]:
#                 none_zero_line[j], none_zero_line[j + 1] = none_zero_line[j + 1], none_zero_line[j]
#                 swapRows(m,j,j+1)
#
#     return m
#
# def  get_upper_triangular_matrix1(m):
#     c, r = shape(m)
#     sort_matrix_by_none_zero_value(m)
#     scale_matrix_elemeng_to_one(m,0,0)
#
#     for i in range(c):
#
#         for j in range (i+1):
#
#             if j<i:
#                 addScaledRow(m, i, j, (-(m[i][j])))
#             if j==i:
#                 while (m[i][j]==0):
#
#                     sort_matrix_by_none_zero_value(m)
#
#                 scale_matrix_elemeng_to_one(m, i, j)
#
#     return m
#
# def  get_upper_triangular_matrix(m,epsilon):
#
#     r, c = shape(m)
#     sort_matrix_by_none_zero_value(m,epsilon)
#
#     for j in range(c):
#         for i in range(r):
#
#             if i==j:
#                 if float_equal(m[i][j],0,epsilon):
#                     return None
#                 scale_matrix_elemeng_to_one(m, i, j)
#             elif i>j:
#                 addScaledRow(m, i, j, (-(m[i][j])))
#         sort_matrix_by_none_zero_value(m,epsilon)
#
#     return m
#
# def float_equal(i,j,epsilon):
#     return ((abs(i-j))<epsilon)
#
# def get_diagonal_matrix(m,epsilon):
#     r,c= shape(m)
#
#     for i in list(range(r)[::-1]):
#         for j in list(range(r)[::-1]):
#
#             if (m[i][j]!=0) and (i!=j):
#
#                 addScaledRow(m, i, j, (-(m[i][j])))
#
#
#     return m
#
# def scale_matrix_elemeng_to_one(m,r,c):
#
#
#     if m[r][c]!=1:
#         scaleRow(m,r,1/m[r][c])
#         return m
#
# epsilon=1.0e-16
#
# def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
#     c,r = shape(A)
#     c_b ,r_b =shape(b)
#     if c!= c_b:
#         return None
#
#     m= augmentMatrix(A,b)
#
#     m=get_upper_triangular_matrix(m,epsilon)
#     if m==None:
#         return None
#
#
#     m=get_diagonal_matrix(m,epsilon)
#
#     matxRound(m,decPts)
#
#     return_list=[[line[c]] for line in m]
#
#
#     return return_list
#
def shape(input_matrix):
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


def transpose(m):
    line_num,columb_num=shape(m)
    print(line_num,columb_num)
    a = [[] for i in range(columb_num)]
    print (a)
    for i in m:
        for j in range(line_num):
            a[j].append(i[j])
            print (a)
    m=a
    return a

print (transpose([5,8,9,7,5]))
