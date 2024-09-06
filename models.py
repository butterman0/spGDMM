import numpy as np
from scipy.stats import invgamma
import pymc as pm

# expcov function to calculate the exponential covariance matrix
def expcov(dists, rho, sigma):
    """
    Calculate the exponential covariance matrix based on distance between points.

    Key properties:
        - Positive definite
        - Stationarity: i.e. Covariance depends ONLY on distance between and not absolute locations.
        - Smoothness i.e. continuous
        - Correlation decay i.e. correlation decays as a function of distance between points

    Parameters:
    dists (ndarray): 
    rho (float): The range parameter. Determines the rate at which covariance decreases as the distance increases. 
    sigma (float): The standard deviation. Represents the scale of the covariance.

    Returns:
    ndarray: The exponential covariance matrix.
    """
    n = dists.shape[0]
    result = np.zeros((n, n))
    sigma2 = sigma ** 2

    # Calculate the covariance between each pair of points
    for i in range(n-1):
        for j in range(i+1, n):
            temp = sigma2 * np.exp(-dists[i, j] / rho)
            result[i, j] = temp
            result[j, i] = temp

    # Set the diagonal elements to sigma^2
    for i in range(n):
        result[i, i] = sigma2

    return result

def model_1_pymc(X, p, n, c, log_V, censored):
    """
    Implement the first model using PyMC. Provided parameters are for a specific site-pair.
    
    V ~ N(mu, sigma^2)
    mu = beta_0 + beta * h(||s[i] - s[j]||) + sum of the covariate distances

    Spatial Random Effects -    None
    Variance -                  sigma^2

    Parameters:
    X (ndarray): The feature matrix.
    p (int): The number of features.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.

    Returns:
    trace: The trace of the sampled posterior.
    """
    with pm.Model() as model:
        # Define the priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
        beta = pm.Lognormal('beta', mu=0, sigma=10, shape=p)
        
        # Calculate the linear predictor
        linpred = pm.math.dot(X[:, :p], beta
        
        # Define the likelihood
        sigma2 = pm.InverseGamma('sigma2', alpha=1, beta=1)
        mu = beta_0 + linpred
        log_V_obs = pm.Normal('log_V_obs', mu=mu, sigma=pm.math.sqrt(sigma2), observed=log_V)
        
        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=True, progressbar=True)
        
    return trace

def model_1(X, p, n, c, log_V, censored):
    """
    Implement the first model. Provided parameters are for a specific site-pair.
    
    V ~ N(mu, sigma^2)
    mu = beta_0 + beta * h(||s[i] - s[j]||) + sum of the covariate distances

    Spatial Random Effects -    None
    Variance -                  sigma^2

    Parameters:
    X (ndarray): The feature matrix.
    p (int): The number of features.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    beta = np.exp(np.random.normal(0, 10, size=p))

    # Calculate the linear predictor
    linpred = np.dot(X[:, :p], beta)

    # Sample the error variance
    sigma2 = invgamma.rvs(1, 1)

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i]
        log_V[i] = np.random.normal(mu, np.sqrt(sigma2))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code2 function
def nimble_code2(X, p, p_sigma, n, c, log_V, censored, X_sigma):
    """
    Implement the second NIMBLE code block.

    Parameters:
    X (ndarray): The feature matrix.
    p (int): The number of features.
    p_sigma (int): The number of features for the variance model.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    X_sigma (ndarray): The feature matrix for the variance model.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    beta = np.exp(np.random.normal(0, 10, size=p))
    beta_sigma = np.random.normal(0, 10, size=p_sigma)

    # Calculate the linear predictor and the variance
    linpred = np.dot(X[:, :p], beta)
    var_out = np.exp(np.dot(X_sigma[:, :p_sigma], beta_sigma))

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i]
        log_V[i] = np.random.normal(mu, np.sqrt(var_out[i]))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code3 function
def nimble_code3(x, p, p_sigma, n, c, log_V, censored):
    """
    Implement the third NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    p_sigma (int): The number of features for the variance model.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    beta = np.exp(np.random.normal(0, 10, size=p))
    beta_sigma = np.random.normal(0, 10, size=p_sigma)

    # Calculate the linear predictor
    linpred = np.dot(x[:, :p], beta)

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i]
        var_out = np.exp(beta_sigma[0] + mu * beta_sigma[1] + mu ** 2 * beta_sigma[2] + mu ** 3 * beta_sigma[3])
        log_V[i] = np.random.normal(mu, np.sqrt(var_out))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code4 function
def nimble_code4(x, p, n, c, log_V, censored, row_ind, col_ind, R_inv, n_loc):
    """
    Implement the fourth NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    row_ind (ndarray): The row indices for the spatial random effects.
    col_ind (ndarray): The column indices for the spatial random effects.
    R_inv (ndarray): The inverse of the spatial correlation matrix.
    n_loc (int): The number of spatial locations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    sig2_psi = invgamma.rvs(1, 1)
    prec_use = R_inv / sig2_psi
    psi = np.random.multivariate_normal(np.zeros(n_loc), prec_use)
    beta = np.exp(np.random.normal(0, 10, size=p))
    sigma2 = invgamma.rvs(1, 1)

    # Calculate the linear predictor
    linpred = np.dot(x[:, :p], beta)

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i] + np.abs(psi[row_ind[i]] - psi[col_ind[i]])
        log_V[i] = np.random.normal(mu, np.sqrt(sigma2))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code5 function
def nimble_code5(x, p, p_sigma, n, c, log_V, censored, row_ind, col_ind, R_inv, n_loc):
    """
    Implement the fifth NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    p_sigma (int): The number of features for the variance model.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    row_ind (ndarray): The row indices for the spatial random effects.
    col_ind (ndarray): The column indices for the spatial random effects.
    R_inv (ndarray): The inverse of the spatial correlation matrix.
    n_loc (int): The number of spatial locations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    sig2_psi = invgamma.rvs(1, 1)
    prec_use = R_inv / sig2_psi
    psi = np.random.multivariate_normal(np.zeros(n_loc), prec_use)
    beta = np.exp(np.random.normal(0, 10, size=p))
    beta_sigma = np.random.normal(0, 10, size=p_sigma)

    # Calculate the linear predictor and the variance
    linpred = np.dot(x[:, :p], beta)
    var_out = np.exp(np.dot(X_sigma[:, :p_sigma], beta_sigma))

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i] + np.abs(psi[row_ind[i]] - psi[col_ind[i]])
        log_V[i] = np.random.normal(mu, np.sqrt(var_out[i]))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code6 function
def nimble_code6(x, p, p_sigma, n, c, log_V, censored, row_ind, col_ind, R_inv, n_loc):
    """
    Implement the sixth NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    p_sigma (int): The number of features for the variance model.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    row_ind (ndarray): The row indices for the spatial random effects.
    col_ind (ndarray): The column indices for the spatial random effects.
    R_inv (ndarray): The inverse of the spatial correlation matrix.
    n_loc (int): The number of spatial locations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    sig2_psi = invgamma.rvs(1, 1)
    prec_use = R_inv / sig2_psi
    psi = np.random.multivariate_normal(np.zeros(n_loc), prec_use)
    beta = np.exp(np.random.normal(0, 10, size=p))
    beta_sigma = np.random.normal(0, 10, size=p_sigma)

    # Calculate the linear predictor
    linpred = np.dot(x[:, :p], beta)

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i] + np.abs(psi[row_ind[i]] - psi[col_ind[i]])
        var_out = np.exp(beta_sigma[0] + mu * beta_sigma[1] + mu ** 2 * beta_sigma[2] + mu ** 3 * beta_sigma[3])
        log_V[i] = np.random.normal(mu, np.sqrt(var_out))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code7 function
def nimble_code7(x, p, n, c, log_V, censored, row_ind, col_ind, R_inv, n_loc):
    """
    Implement the seventh NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    row_ind (ndarray): The row indices for the spatial random effects.
    col_ind (ndarray): The column indices for the spatial random effects.
    R_inv (ndarray): The inverse of the spatial correlation matrix.
    n_loc (int): The number of spatial locations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    sig2_psi = invgamma.rvs(1, 1)
    prec_use = R_inv / sig2_psi
    psi = np.random.multivariate_normal(np.zeros(n_loc), prec_use)
    beta = np.exp(np.random.normal(0, 10, size=p))
    sigma2 = invgamma.rvs(1, 1)

    # Calculate the linear predictor
    linpred = np.dot(x[:, :p], beta)

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i] + (psi[row_ind[i]] - psi[col_ind[i]])**2
        log_V[i] = np.random.normal(mu, np.sqrt(sigma2))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code8 function
def nimble_code8(x, p, p_sigma, n, c, log_V, censored, row_ind, col_ind, R_inv, n_loc):
    """
    Implement the eighth NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    p_sigma (int): The number of features for the variance model.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    row_ind (ndarray): The row indices for the spatial random effects.
    col_ind (ndarray): The column indices for the spatial random effects.
    R_inv (ndarray): The inverse of the spatial correlation matrix.
    n_loc (int): The number of spatial locations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    sig2_psi = invgamma.rvs(1, 1)
    prec_use = R_inv / sig2_psi
    psi = np.random.multivariate_normal(np.zeros(n_loc), prec_use)
    beta = np.exp(np.random.normal(0, 10, size=p))
    beta_sigma = np.random.normal(0, 10, size=p_sigma)

    # Calculate the linear predictor and the variance
    linpred = np.dot(x[:, :p], beta)
    var_out = np.exp(np.dot(X_sigma[:, :p_sigma], beta_sigma))

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i] + (psi[row_ind[i]] - psi[col_ind[i]])**2
        log_V[i] = np.random.normal(mu, np.sqrt(var_out[i]))
        censored[i] = np.random.uniform(log_V[i], c[i])

# nimble_code9 function
def nimble_code9(x, p, p_sigma, n, c, log_V, censored, row_ind, col_ind, R_inv, n_loc):
    """
    Implement the ninth NIMBLE code block.

    Parameters:
    x (ndarray): The feature matrix.
    p (int): The number of features.
    p_sigma (int): The number of features for the variance model.
    n (int): The number of observations.
    c (ndarray): The censoring values.
    log_V (ndarray): The log-transformed response variable.
    censored (ndarray): The censored observations.
    row_ind (ndarray): The row indices for the spatial random effects.
    col_ind (ndarray): The column indices for the spatial random effects.
    R_inv (ndarray): The inverse of the spatial correlation matrix.
    n_loc (int): The number of spatial locations.

    Returns:
    None
    """
    # Define the priors
    beta_0 = np.random.normal(0, 10)
    sig2_psi = invgamma.rvs(1, 1)
    prec_use = R_inv / sig2_psi
    psi = np.random.multivariate_normal(np.zeros(n_loc), prec_use)
    beta = np.exp(np.random.normal(0, 10, size=p))
    beta_sigma = np.random.normal(0, 10, size=p_sigma)

    # Calculate the linear predictor
    linpred = np.dot(x[:, :p], beta)

    # Loop through the observations and sample the response and censored values
    for i in range(n):
        mu = beta_0 + linpred[i] + (psi[row_ind[i]] - psi[col_ind[i]])**2
        var_out = np.exp(beta_sigma[0] + mu * beta_sigma[1] + mu ** 2 * beta_sigma[2] + mu ** 3 * beta_sigma[3])
        log_V[i] = np.random.normal(mu, np.sqrt(var_out))
        censored[i] = np.random.uniform(log_V[i], c[i])