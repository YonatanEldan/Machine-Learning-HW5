import numpy as np
import time
import datetime
import itertools

# generate 20,000 vectors of 20 features each
A = np.random.randn(20000, 20)
squareOfTwo = np.sqrt(2)


# kernel function - K(x,y) = (x*y +1) ^ 2
def K(x ,y):
    return (np.inner(x, y) + 1) ** 2

# Gram matrix using kernel function
B = np.zeros(shape = (20000, 20000))
start = time.time()

for i in range(20000):
    for j in range(20000):
        B[i][j] = K(A[i], A[j])
    print('first' , i)
end = time.time()
k_time = end - start

# phi function for the same kernel
def phi(x):
    combinationsArr = list(itertools.combinations(x, 2))
    temp1 = map(lambda a: a * squareOfTwo, x)
    temp2 = map(lambda a: a * a, x)
    temp3 = map(lambda (a, b): a * b * squareOfTwo, combinationsArr)
    return np.concatenate(([1], temp1, temp2, temp3))


print(time.time() - start)
C = np.zeros(shape=(20000, 20000))

start = time.time()


phiValueArray = np.zeros(shape=(20000, 231))

for i in range(20000):
    phiValueArray[i] = phi(A[i])


for i in range(20000):
    for j in range(20000):
        C[i][j] = np.inner(phiValueArray[i],  phiValueArray[j])
    print('second' , i)
    end = time.time()
    phi_time = end - start

print(56)
print(np.isclose(B, C))
print(np.allclose(B, C))

str1 = str(datetime.timedelta(seconds=k_time))
str2 = str(datetime.timedelta(seconds=phi_time))

print('time of calculating the matrix by using the kernel is: ' + str1 +
      '\n time of calculating the matrix by using the phi is: ' + str2)

