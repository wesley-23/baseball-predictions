from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import math


"""
    Wrapper class for a logistic regression model to predict hit probabilities.
    kwargs:
        years: list of ints representing the years to train model on
"""
class Logistic_Regression:

    def __init__(self, **kwargs):
        self.years = kwargs.get('years', None)
        if not isinstance(self.years, list) or not isinstance(self.years[0], int):
            raise TypeError('Years must be a nonempty list of integers')
        self.create_data_frame()
    
    def create_data_frame(self):
        li = []
        for year in self.years:
            path = 'data/' + str(year) + '_pbp_all/' + str(year) + '_pbp_conglomerate.csv'
            df = pd.read_csv(path, skipinitialspace=True)
            li.append(df)
        frame = pd.concat(li, axis = 0, ignore_index=True)
        frame = frame[(frame['EV'] != 'None') & (frame['LA'] != 'None') & (frame['Outcome'] != 'Catcher Inteference') & (frame['Sprint_Speed'] != 'None')]
        self.Y = frame['Outcome']
        self.Y[(self.Y == 'Single') | (self.Y == 'Double') | (self.Y == 'Triple') | (self.Y == 'Home Run')] = 1
        self.Y[self.Y != 1] = 0
        self.X = pd.DataFrame({
            'intercept': np.ones(frame['EV'].shape),
            'EV': frame['EV'].astype(float),
            'LA': frame['LA'].astype(float),
            'sprint_speed': frame['Sprint_Speed'].astype(float)
        })
        # print((self.Y))
        # print(self.X)
        self.features = 4

    """
        Calculates P(Outcome = Hit|X=x) under the logistic regression model. Numerator is exp(dot product between observation and coffecients).
        Denominator is 1 + numerator as we are solving a 2 class case.
    """
    def calc_likelihoods(self, obs):
        numerator = 0
        for i in range(len(obs)):
            numerator += self.coeffs[i] * obs[i]
        numerator = math.exp(numerator)
        denominator = 1 + numerator
        if denominator == 0:
            print(numerator, denominator)
            print(obs)
        return numerator / denominator
        
    """
        Return an N X 1 2d numpy array in which the ith element is the probability that the ith outcome is a hit given the current parameters.
    """
    def create_probabilities_vector(self, X):
        p = np.empty((len(X)))
        for i in range(len(X)):
            obs = X[i]
            prob = self.calc_likelihoods(obs)
            p[i] = prob
        return p

    """
        Returns an NxN diagonal matrix W and its inverse with ith diagonal element P(outcome = hit|coeffs) * (1 - P(outcome = hit|coeffs)). The 
        matrices will be returned in a vectorized form to save space, where the ith element corresponds to the i, ith diagonals of the matrix.
    """
    def create_weight_matrix(self, p):
        W = np.zeros((len(p)))
        W_inv = np.zeros((len(p)))
        for i in range(len(p)):
            W[i] = p[i] * (1 - p[i])
            W_inv[i] = 1/W[i]
        return W, W_inv

    """
        Multiplies the vectorized form of a diagonal matrix represented by W with a vector b.
    """
    def mult_vec_with_vec(self, W, b):
        ans = np.zeros(len(b))
        for i in range(len(b)):
            ans[i] = b[i] * W[i]  
        return ans
    
    """
        Multiplies a matrix with the vectorized form of a diagonal matrix, represented by the vector b.
    """
    def mult_vectorized(self, A, b):
        ans = np.zeros((len(A), len(A[0])))
        for i in range(len(A)):
            ans[i] = self.mult_vec_with_vec(b, A[i])
        return ans

    """
        Fit logistic model by Newton Raphson Method. Store the coefficient values for prediction and graphing purposes and store the covariance matrix
        (inverse of the negative second derivative) for inference purposes.
    """
    def fit(self):
        self.coeffs = []
        for i in range(self.features):
            self.coeffs.append(0) ## Initialize all coefficients to equal 0
        X = self.X.to_numpy()
        Y = self.Y.to_numpy().transpose()  ## Get features and outcomes as numpy array for faster indexing

        for i in range(0, 100):
            p = self.create_probabilities_vector(X)
            Xt = np.transpose(X)
            y_p = np.subtract(Y, p)
            W, W_inv = self.create_weight_matrix(p)
            second_der = np.matmul(self.mult_vectorized(Xt, W), X) ## second derivative of log-lik is -X'WX (ignore negative sign for now)
            z = np.add(np.dot(X, self.coeffs), self.mult_vec_with_vec(W_inv, y_p))  ## define z as (XB + W_inv(y - p)) where B are the coeffs
            self.covar = np.linalg.inv(second_der)  ## Store covariance matrix

            ## By Newton-Raphson method, new coeffs = inv(X'WX)X'Wz
            self.coeffs = np.dot(self.covar, np.dot(Xt, self.mult_vec_with_vec(W, z)))
            first_der = np.dot(Xt, y_p) 
            if (np.linalg.norm(first_der) < 0.0001):
                print(self.coeffs)
                ## Exit when first derivative is very close to 0. Coefficient values and covariance matrix have been saved
                return True
        return False

        
            
    

