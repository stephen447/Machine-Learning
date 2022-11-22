a = 5
b = 3
#Array = [[0 for x in range(a)] for y in range(a)]
Array = [[1, 1, 1, 2, 8], [3, 2, 3, 2, 2], [7, 1, 4, 0, 9], [8, 4, 5, 0, 5], [6, 5, 6, 7, 0]]
#Kernel = [[0 for x in range(b)] for y in range(b)]
Kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
Kernel1 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
Kernel2 = [[0, -1, 0], [-1, 8, -1], [0, -1, 0]]


def convolution(array1, kernel1):
    n = len(array1)
    k = len(kernel1)
    con_total = 0

    Convo = [[0 for x in range(n-k+1)] for y in range(n-k+1)]
    indent = int(k/2)
    e = 0
    f = 0
    for x in range(indent, n-indent):
        for y in range(indent, n-indent):
            con_total = 0

            c = 0
            d = 0
            for a in range(x-indent,x+indent+1):
                for b in range(y-indent,y+indent+1):

                    arr = array1[a][b]
                    #print("Array",arr)
                    ker = kernel1[d][c]
                    #print("Kernel", ker)
                    con_total = con_total+(arr*ker)
                    #print("Total Con",con_total)
                    if(c<2):
                        c = c+1
                    else:
                        c = 0
                        d = d+1
            Convo[f][e] = con_total
            print("Conv array:", Convo[e][f])
            #if(e<(n-2*indent)):
            if(e<n-2*indent-1):
                e = e + 1
            else:
                e = 0
                f = f+1
        print("Convo:", Convo)
        print(np.matrix(Convo))
        rows = len(Convo)
        columns = len(Convo[0])
        print(rows)
        print(columns)
    return Convo


import numpy as np
from PIL import Image
#im = Image.open('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Lab 4/testimg.jpeg')
im = Image.open('/Users/stephenbyrne/Documents/College Year 5/Machine Learning/Labs/Lab 4/test2.png')
rgb = np.array(im.convert('RGB'))
r = rgb[:, :, 0] # array of R pixels
#np.set_printoptions(threshold=np.inf)
#print(r)
Image.fromarray(np.uint8(r)).show()
#print(r)
dimension = rgb.shape
dimension1 = r.shape
print(dimension1)
convolution(Array, Kernel)

Convo = convolution(Array, Kernel)
print("Result is:", Convo)
#Image.fromarray(np.uint8(Convo)).show()
#Convo = convolution(r, Kernel2)
#Image.fromarray(np.uint8(Convo)).show()


