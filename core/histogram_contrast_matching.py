import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression

def rectify_histogram(movFile, refFile, outFile):
    # Reads inputs
    ref = nib.load(refFile)
    R = ref.get_data()
    mov = nib.load(movFile)
    M = mov.get_data()
    
    M = M * np.std(R[R!=0])/np.std(M[M!=0])

    # Isolate elements
    elt = np.logical_and(R > np.quantile(R, 0.1), M > np.quantile(M, 0.1))

    # Regression
    x, y = R[elt], M[elt]

    # This fit was giving [ 0.37157987 42.94802671]
    #model = np.polyfit(x, y, 1)  # returns coefficients in decreasing degree order
    #print(model)  # check parameters

    #This fit gives [5.880295] and [[0.69228363]]
    X = R[elt].reshape(-1, 1)  # values converts it into a numpy array
    Y = M[elt].reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    intercept = linear_regressor.intercept_[0]
    slope = linear_regressor.coef_[0,0]

    # Calculate result
    res = np.copy(M)
    res[M != 0] = M[M != 0] - slope * R[M != 0] - intercept + R[M != 0]

    # Writes ouptput
    out_nii = nib.Nifti1Image(res, affine=ref.affine, header=ref.header)  # Je cree le Nifti
    nib.save(out_nii, outFile)  # Je le save
