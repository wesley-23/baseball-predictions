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
            'Sprint_Speed': frame['Sprint_Speed'].astype(float)
        })
        # print((self.Y))
        # print(self.X)
        self.tensor_prod_spline()
        self.normalize_predictors()
        self.features = len(self.X.columns)
        # print(self.X)

    """
        Normalizes predictors by scaling them to be between 0 - 1. Prevents overflow in math calculations
    """
    def normalize_predictors(self):
        new_x = {}
        mins = []
        ranges = []
        for col in self.X.columns:
            max = self.X[col].max()
            min = self.X[col].min()
            rng = max - min
            if rng != 0:
                new_x[col] = (self.X[col] - min) / rng
            else:
                new_x[col] = self.X[col]
            mins.append(max)
            ranges.append(rng)
        self.mins = mins
        self.ranges = ranges
        self.X = pd.DataFrame(new_x)

    """
        Expands the basis by creating a spline in all 3 predictors and using the tensor product between their basis functions as the basis functions
        for the model.
    """
    def tensor_prod_spline(self):
        ev_basis = {}
        ev_basis['EV'] = self.X['EV'].to_numpy()
        ev_basis['EV^2'] = ev_basis['EV']**2
        ev_basis['EV^3'] = ev_basis['EV']**3
        for i in range(1, 5):
            basis = []
            knot = (120 * i) / 5
            for ev in self.X['EV']:
                if ev <= knot:
                    basis.append(0)
                else:
                    basis.append((ev - knot)**3)
            ev_basis['EVk'+str(knot)] = basis
        la_basis = {}
        la_basis['LA'] = self.X['LA'].to_numpy()
        la_basis['LA^2'] = la_basis['LA']**2
        la_basis['LA^3'] = la_basis['LA']**3
        for i in range(1, 5):
            basis = []
            knot = 90 - (180 * i / 5)
            for la in self.X['LA']:
                if la <= knot:
                    basis.append(0)
                else:
                    basis.append((la - knot)**3)
            la_basis['LAk' + str(knot)] = basis
        sprint_basis = {}
        sprint_basis['S'] = self.X['Sprint_Speed'].to_numpy()
        sprint_basis['S^2'] = sprint_basis['S']**2
        sprint_basis['S^3'] = sprint_basis['S']**3
        for i in range(1, 3):
            basis = []
            knot = 25 + (2 * i)
            for s in self.X['Sprint_Speed']:
                if s <= knot:
                    basis.append(0)
                else:
                    basis.append((s - knot)**3)
            sprint_basis['Sk'+str(knot)] = basis
        s = pd.DataFrame(sprint_basis)
        new_x = {}
        new_x['intercept'] = self.X['intercept']
        for key in ev_basis:
            val = ev_basis[key]
            new_x[key] = val
            for lakey in la_basis:
                laval = la_basis[lakey]
                new_x[lakey] = laval
                new_x[key+lakey] = self.mult_vec_with_vec(val, laval)
                for skey in sprint_basis:
                    sval = sprint_basis[skey]
                    new_x[skey] = sval
                    new_x[lakey + skey] = self.mult_vec_with_vec(laval, sval)
            for skey in sprint_basis:
                new_x[key+skey] = self.mult_vec_with_vec(val, sprint_basis[skey])
        self.X = pd.DataFrame(new_x)
 
    """
        Calculates P(Outcome = Hit|X=x) under the logistic regression model. Numerator is exp(dot product between observation and coffecients).
        Denominator is 1 + numerator as we are solving a 2 class case.
    """
    def calc_likelihoods(self, obs):
        numerator = 0
        for i in range(len(obs)):
            numerator += self.coeffs[i] * obs[i]
        
        try:    
            numerator = math.exp(numerator)
        except OverflowError:
            return 0.999999
        denominator = 1 + numerator
        output = numerator / denominator
        if output < 0.000001:
            return 0.000001
        if output > 0.999999:
            return 0.999999
        return output
        
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
        return W

    """
        Outputs a new vector whos ith entry is the elementwise product of the ith entries of the 2 input vectors.
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
    
    def calc_log_likelihood(self, X, Y):
        output = 0
        for i in range(len(Y)):
            if Y[i] == 1:
                output += math.log(self.calc_likelihoods(X[i]))
            else:
                output += math.log((1  - self.calc_likelihoods(X[i])))
        return output
    
    """
        Fit logistic model by gradient ascent.
    """
    def fit_by_gradient_ascent(self, learning_rate):
        self.coeffs = np.zeros(self.features)
        X = self.X.to_numpy()
        Y = self.Y.to_numpy()
        X_t = np.transpose(X)

        for i in range(1000):
            print(self.calc_log_likelihood(X, Y))
            p = self.create_probabilities_vector(X)
            gradient = np.dot(X_t, np.subtract(Y, p))
            self.coeffs = np.add(self.coeffs, learning_rate * gradient)
            mag = np.linalg.norm(gradient)
            print(mag)
            if mag < 0.0001:
                break

    """
        Fit logistic model by Newton Raphson Method. Store the coefficient values for prediction and graphing purposes and store the covariance matrix
        (inverse of the negative second derivative) for inference purposes.
    """
    def fit(self):
        self.coeffs = np.random.rand(self.features)  ## Initialize all coefficients to equal 0

        X = self.X.to_numpy()
        Y = self.Y.to_numpy()  ## Get features and outcomes as numpy array for faster indexing

        for i in range(0, 100):
            loglik = self.calc_log_likelihood(X, Y)
            print(loglik)
            p = self.create_probabilities_vector(X)
            Xt = np.transpose(X)
            y_p = np.subtract(Y, p)
            first_der = np.dot(Xt, y_p) ## First derivative is X'(Y - p)
            W = self.create_weight_matrix(p)
            second_der = np.matmul(self.mult_vectorized(Xt, W), X) ## second derivative of log-lik is -X'WX (ignore negative sign for now)
            print(np.linalg.cond(second_der))
            self.covar = np.linalg.inv(second_der)  ## Store covariance matrix
            update = 0.5 * np.dot(self.covar, first_der)

            self.coeffs = np.add(self.coeffs, update)

            mag = np.linalg.norm(first_der)
            # print(self.coeffs)
            print(mag)
            if (mag < 0.0001):
                ## Exit when first derivative is very close to 0. Coefficient values and covariance matrix have been saved
                return True
        return False

    def predict(self, ev, la, sprint):
        vec = (1, ev, la, sprint)
        return np.dot(vec, self.coeffs) >= 0.5
            
    

