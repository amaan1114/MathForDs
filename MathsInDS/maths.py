import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Matrix and Vector Operations
A=np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([1,2,3])

#Matrix-vector multiplication
Multiplication = A@B
print("Matrix-Vector Multiplication: ",Multiplication)


#Trace of A
trace = np.trace(A)
print("Trace of A: ",trace)

#Eigen Value and Eigen Vecotrs
evalue,evector = np.linalg.eig(A)
print("Eigen Value: ",evalue)
print("Eigen Vectors: ",evector,"\n")

#Updating and Findind Determinant
A[len(A)-1]=[10,11,12]
determinant = np.linalg.det(A)
print("Determinant of Updated Matrix:\n",determinant)

#Checking for singular or non Singular
if(determinant==0):
    print("Updated A is singular\n")
else:
    print("Updated A is not Singular\n")


#Invertibility of Matrices
if(determinant==0):
    print("Updated A is non invertible\n")
else:
    inverse = np.linalg.inv(A)

#Solving Linear equation AX=B
B=B.reshape(-1,1)
if(determinant==0):
    print("Determinant of A is zero so system has no unique solutions\n\n")
else:
    X = np.linalg.solve(A,B)






#Practical Matrix Operation
C = np.random.randint(1, 21, size=(4, 4))
#Calculating Rank
rank = np.linalg.matrix_rank(C)
print("Rank of C: ",rank,"\n")
#Making a submatrix
submatrix = C[0:2,2:4]
print("Submatrix with 1st 2 rows and last 2 columns: \n",submatrix,"\n")
#Forbenius norm
FNorm = np.linalg.norm(C,'fro')
print("Forbenius Norm of Matrix C: ",FNorm,"\n")

#Matrix Multiplication A and C
C=C[0:3,0:3]
#Checking if the matrix multiplication is possible
if(A.shape[1]==C.shape[0]):
    Matrix_Multiplication = A@C

#If not possible resizing C to make it possible.
else:
    C=C[0:A.shape[1],:]
    Matrix_Multiplication = A@C
print("Matrix Multiplication of A and C: \n",Matrix_Multiplication)






#Data Science Context
D = np.array([[3, 5, 7, 9, 11],
                 [2, 4, 6, 8, 10],
                 [1, 3, 5, 7, 9],
                 [4, 6, 8, 10, 12],
                 [5, 7, 9, 11, 13]])
Standardise = StandardScaler()
X=Standardise.fit_transform(D)

print("Matrix after Standardising: \n")
print(X)

CovarianceMat = np.cov(D)
print("Covariance Matrix: \n")
print(CovarianceMat)

print("Eigen vectors of covariance matrix: ", np.linalg.eig(CovarianceMat),"\n\n")

#Performing Principal Component Analysis
pca = PCA(n_components=2)
pca.fit(D)
AfterPca = pca.transform(D)
print("After Principal Component Analysis:\n",AfterPca)

