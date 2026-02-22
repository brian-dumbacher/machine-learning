# Name:     model_error_simulation.py
# Purpose:  Conduct simulation to explore the different types of model error in machine learning
# Author:   Brian Dumbacher
# Date:     May 31, 2025

import matplotlib.pyplot as plt
import random
from sklearn import linear_model

# Name:        f
# Purpose:     Calculate value of function that relates the X and Y
# Parameters:  x
# Returns:     Value of the function f(x) = x^2 - 8x + 20

def f(x):
    return x**2 - 8 * x + 20

# Name:        evu
# Purpose:     Calculate E[U^m], where U ~ Uniform(0, 10)
# Parameters:  m
# Returns:     10**m/(m+1)

def evu(m):
    return 10**m/(m+1)

# Name:        main
# Purpose:     Conduct simulation
# Parameters:
# Returns:

def main():
    ######################
    ###   Parameters   ###
    ######################

    # Noise standard deviation and variance
    sigma  = 5
    sigma2 = sigma**2
    # Training set size
    n = 25
    # Number of training sets
    B = 5000
    # Random number seed
    random.seed(12345)

    ###########################
    ###   Data structures   ###
    ###########################

    # Training sets
    xs_train = {}
    ys_train = {}

    # Estimated model parameters
    betas_const  = {"beta0": {}}
    betas_linear = {"beta0": {}, "beta1": {}}
    betas_quad   = {"beta0": {}, "beta1": {}, "beta2": {}}
    betas_cubic  = {"beta0": {}, "beta1": {}, "beta2": {}, "beta3": {}}
    betas_quart  = {"beta0": {}, "beta1": {}, "beta2": {}, "beta3": {}, "beta4": {}}
    betas_quint  = {"beta0": {}, "beta1": {}, "beta2": {}, "beta3": {}, "beta4": {}, "beta5": {}}

    # Model errors
    # Training error
    errs_train_const  = {}
    errs_train_linear = {}
    errs_train_quad   = {}
    errs_train_cubic  = {}
    errs_train_quart  = {}
    errs_train_quint  = {}

    # In-sample error
    errs_in_const  = {}
    errs_in_linear = {}
    errs_in_quad   = {}
    errs_in_cubic  = {}
    errs_in_quart  = {}
    errs_in_quint  = {}

    # Test error
    errs_test_const  = {}
    errs_test_linear = {}
    errs_test_quad   = {}
    errs_test_cubic  = {}
    errs_test_quart  = {}
    errs_test_quint  = {}

    ######################
    ###   Simulation   ###
    ######################

    for b in range(B):
        # Generate training set from joint distribution of X and Y
        xs  = [random.uniform(0, 10) for i in range(n)]
        ys  = [random.normalvariate(f(x), sigma) for x in xs]
        fxs = [f(x) for x in xs]
        xs_train[b] = xs
        ys_train[b] = ys

        # Fit regression models
        # Constant
        mean_ys = sum(ys)/n
        betas_const["beta0"][b] = mean_ys
        yhats_const = [mean_ys for val in xs]

        # Linear
        xs_feats_linear = [[x] for x in xs]
        mod_linear = linear_model.LinearRegression(fit_intercept=True)
        mod_linear.fit(xs_feats_linear, ys)
        betas_linear["beta0"][b] = mod_linear.intercept_
        betas_linear["beta1"][b] = mod_linear.coef_[0]
        yhats_linear = [val for val in mod_linear.predict(xs_feats_linear)]

        # Quadratic
        xs_feats_quad = [[x, x**2] for x in xs]
        mod_quad = linear_model.LinearRegression(fit_intercept=True)
        mod_quad.fit(xs_feats_quad, ys)
        betas_quad["beta0"][b] = mod_quad.intercept_
        betas_quad["beta1"][b] = mod_quad.coef_[0]
        betas_quad["beta2"][b] = mod_quad.coef_[1]
        yhats_quad = [val for val in mod_quad.predict(xs_feats_quad)]

        # Cubic
        xs_feats_cubic = [[x, x**2, x**3] for x in xs]
        mod_cubic = linear_model.LinearRegression(fit_intercept=True)
        mod_cubic.fit(xs_feats_cubic, ys)
        betas_cubic["beta0"][b] = mod_cubic.intercept_
        betas_cubic["beta1"][b] = mod_cubic.coef_[0]
        betas_cubic["beta2"][b] = mod_cubic.coef_[1]
        betas_cubic["beta3"][b] = mod_cubic.coef_[2]
        yhats_cubic = [val for val in mod_cubic.predict(xs_feats_cubic)]

        # Quartic
        xs_feats_quart = [[x, x**2, x**3, x**4] for x in xs]
        mod_quart = linear_model.LinearRegression(fit_intercept=True)
        mod_quart.fit(xs_feats_quart, ys)
        betas_quart["beta0"][b] = mod_quart.intercept_
        betas_quart["beta1"][b] = mod_quart.coef_[0]
        betas_quart["beta2"][b] = mod_quart.coef_[1]
        betas_quart["beta3"][b] = mod_quart.coef_[2]
        betas_quart["beta4"][b] = mod_quart.coef_[3]
        yhats_quart = [val for val in mod_quart.predict(xs_feats_quart)]

        # Quintic
        xs_feats_quint = [[x, x**2, x**3, x**4, x**5] for x in xs]
        mod_quint = linear_model.LinearRegression(fit_intercept=True)
        mod_quint.fit(xs_feats_quint, ys)
        betas_quint["beta0"][b] = mod_quint.intercept_
        betas_quint["beta1"][b] = mod_quint.coef_[0]
        betas_quint["beta2"][b] = mod_quint.coef_[1]
        betas_quint["beta3"][b] = mod_quint.coef_[2]
        betas_quint["beta4"][b] = mod_quint.coef_[3]
        betas_quint["beta5"][b] = mod_quint.coef_[4]
        yhats_quint = [val for val in mod_quint.predict(xs_feats_quint)]

        # Calculate training error
        errs_train_const[b]  = sum([(yhats_const[i]  - ys[i])**2 for i in range(n)])/n
        errs_train_linear[b] = sum([(yhats_linear[i] - ys[i])**2 for i in range(n)])/n
        errs_train_quad[b]   = sum([(yhats_quad[i]   - ys[i])**2 for i in range(n)])/n
        errs_train_cubic[b]  = sum([(yhats_cubic[i]  - ys[i])**2 for i in range(n)])/n
        errs_train_quart[b]  = sum([(yhats_quart[i]  - ys[i])**2 for i in range(n)])/n
        errs_train_quint[b]  = sum([(yhats_quint[i]  - ys[i])**2 for i in range(n)])/n

        # Calculate in-sample error
        errs_in_const[b]  = sigma2 + sum([(yhats_const[i]  - fxs[i])**2 for i in range(n)])/n
        errs_in_linear[b] = sigma2 + sum([(yhats_linear[i] - fxs[i])**2 for i in range(n)])/n
        errs_in_quad[b]   = sigma2 + sum([(yhats_quad[i]   - fxs[i])**2 for i in range(n)])/n
        errs_in_cubic[b]  = sigma2 + sum([(yhats_cubic[i]  - fxs[i])**2 for i in range(n)])/n
        errs_in_quart[b]  = sigma2 + sum([(yhats_quart[i]  - fxs[i])**2 for i in range(n)])/n
        errs_in_quint[b]  = sigma2 + sum([(yhats_quint[i]  - fxs[i])**2 for i in range(n)])/n

        # Calculate test error
        errs_test_const[b] =  sigma2
        errs_test_const[b] += (20 - betas_const["beta0"][b])**2
        errs_test_const[b] += -16 * (20 - betas_const["beta0"][b]) * evu(1)
        errs_test_const[b] += (64 + 2 * (20 - betas_const["beta0"][b])) * evu(2)
        errs_test_const[b] += -16 * evu(3)
        errs_test_const[b] += evu(4)

        errs_test_linear[b] =  sigma2
        errs_test_linear[b] += (20 - betas_linear["beta0"][b])**2
        errs_test_linear[b] += -2 * (20 - betas_linear["beta0"][b]) * (8 + betas_linear["beta1"][b]) * evu(1)
        errs_test_linear[b] += (40 - 2*betas_linear["beta0"][b] + (8 + betas_linear["beta1"][b])**2) * evu(2)
        errs_test_linear[b] += -2 * (8 + betas_linear["beta1"][b]) * evu(3)
        errs_test_linear[b] += evu(4)

        errs_test_quad[b] =  sigma2
        errs_test_quad[b] += (20 - betas_quad["beta0"][b])**2
        errs_test_quad[b] += -2 * (20 - betas_quad["beta0"][b]) * (8 + betas_quad["beta1"][b]) * evu(1)
        errs_test_quad[b] += (2 * (20 - betas_quad["beta0"][b]) * (1 - betas_quad["beta2"][b]) + (8 + betas_quad["beta1"][b])**2) * evu(2)
        errs_test_quad[b] += -2 * (8 + betas_quad["beta1"][b]) * (1 - betas_quad["beta2"][b]) * evu(3)
        errs_test_quad[b] += (1 - betas_quad["beta2"][b])**2 * evu(4)

        errs_test_cubic[b] =  sigma2
        errs_test_cubic[b] += (20 - betas_cubic["beta0"][b])**2
        errs_test_cubic[b] += -2 * (20 - betas_cubic["beta0"][b]) * (8 + betas_cubic["beta1"][b]) * evu(1)
        errs_test_cubic[b] += (2 * (20 - betas_cubic["beta0"][b]) * (1 - betas_cubic["beta2"][b]) + (8 + betas_cubic["beta1"][b])**2) * evu(2)
        errs_test_cubic[b] += -2 * ((20 - betas_cubic["beta0"][b]) * betas_cubic["beta3"][b] + (8 + betas_cubic["beta1"][b]) * (1 - betas_cubic["beta2"][b])) * evu(3)
        errs_test_cubic[b] += (2 * (8 + betas_cubic["beta1"][b]) * betas_cubic["beta3"][b] + (1 - betas_cubic["beta2"][b])**2) * evu(4)
        errs_test_cubic[b] += -2 * (1 - betas_cubic["beta2"][b]) * betas_cubic["beta3"][b] * evu(5)
        errs_test_cubic[b] += (betas_cubic["beta3"][b])**2 * evu(6)

        errs_test_quart[b] =  sigma2
        errs_test_quart[b] += (20 - betas_quart["beta0"][b])**2
        errs_test_quart[b] += -2 * (20 - betas_quart["beta0"][b]) * (8 + betas_quart["beta1"][b]) * evu(1)
        errs_test_quart[b] += ((8 + betas_quart["beta1"][b])**2 + 2 * (20 - betas_quart["beta0"][b]) * (1 - betas_quart["beta2"][b])) * evu(2)
        errs_test_quart[b] += -2 * ((20 - betas_quart["beta0"][b]) * betas_quart["beta3"][b] + (8 + betas_quart["beta1"][b]) * (1 - betas_quart["beta2"][b])) * evu(3)
        errs_test_quart[b] += ((1 - betas_quart["beta2"][b])**2 - 2 * (20 - betas_quart["beta0"][b]) * betas_quart["beta4"][b] + 2 * (8 + betas_quart["beta1"][b]) * betas_quart["beta3"][b]) * evu(4)
        errs_test_quart[b] += 2 * ((8 + betas_quart["beta1"][b]) * betas_quart["beta4"][b] - (1 - betas_quart["beta2"][b]) * betas_quart["beta3"][b]) * evu(5)
        errs_test_quart[b] += ((betas_quart["beta3"][b])**2 - 2 * (1 - betas_quart["beta2"][b]) * betas_quart["beta4"][b]) * evu(6)
        errs_test_quart[b] += 2 * betas_quart["beta3"][b] * betas_quart["beta4"][b] * evu(7)
        errs_test_quart[b] += (betas_quart["beta4"][b])**2 * evu(8)

        errs_test_quint[b] =  sigma2
        errs_test_quint[b] += (20 - betas_quint["beta0"][b])**2
        errs_test_quint[b] += -2 * (20 - betas_quint["beta0"][b]) * (8 + betas_quint["beta1"][b]) * evu(1)
        errs_test_quint[b] += ((8 + betas_quint["beta1"][b])**2 + 2 * (20 - betas_quint["beta0"][b]) * (1 - betas_quint["beta2"][b])) * evu(2)
        errs_test_quint[b] += -2 * ((20 - betas_quint["beta0"][b]) * betas_quint["beta3"][b] + (8 + betas_quint["beta1"][b]) * (1 - betas_quint["beta2"][b])) * evu(3)
        errs_test_quint[b] += ((1 - betas_quint["beta2"][b])**2 - 2 * (20 - betas_quint["beta0"][b]) * betas_quint["beta4"][b] + 2 * (8 + betas_quint["beta1"][b]) * betas_quint["beta3"][b]) * evu(4)
        errs_test_quint[b] += -2 * ((20 - betas_quint["beta0"][b]) * betas_quint["beta5"][b] - (8 + betas_quint["beta1"][b]) * betas_quint["beta4"][b] + (1 - betas_quint["beta2"][b]) * betas_quint["beta3"][b]) * evu(5)
        errs_test_quint[b] += ((betas_quint["beta3"][b])**2 + 2 * (8 + betas_quint["beta1"][b]) * betas_quint["beta5"][b] - 2 * (1 - betas_quint["beta2"][b]) * betas_quint["beta4"][b]) * evu(6)
        errs_test_quint[b] += 2 * (betas_quint["beta3"][b] * betas_quint["beta4"][b] - (1 - betas_quint["beta2"][b]) * betas_quint["beta5"][b]) * evu(7)
        errs_test_quint[b] += ((betas_quint["beta4"][b])**2 + 2 * betas_quint["beta3"][b] * betas_quint["beta5"][b]) * evu(8)
        errs_test_quint[b] += 2 * betas_quint["beta4"][b] * betas_quint["beta5"][b] * evu(9)
        errs_test_quint[b] += (betas_quint["beta5"][b])**2 * evu(10)

        # Cleanup
        del xs, xs_feats_linear, xs_feats_quad, xs_feats_cubic, xs_feats_quart, xs_feats_quint
        del ys, yhats_const, yhats_linear, yhats_quad, yhats_cubic, yhats_quart, yhats_quint
        del fxs, mean_ys
        del mod_linear, mod_quad, mod_cubic, mod_quart, mod_quint

    # Estimate expected training error
    err_exp_train_const  = sum([errs_train_const[b]  for b in range(B)])/B
    err_exp_train_linear = sum([errs_train_linear[b] for b in range(B)])/B
    err_exp_train_quad   = sum([errs_train_quad[b]   for b in range(B)])/B
    err_exp_train_cubic  = sum([errs_train_cubic[b]  for b in range(B)])/B
    err_exp_train_quart  = sum([errs_train_quart[b]  for b in range(B)])/B
    err_exp_train_quint  = sum([errs_train_quint[b]  for b in range(B)])/B

    print("")
    print("Estimates of expected training error")
    print("------------------------------------")
    print("Constant:   {0:>6.2f}".format(err_exp_train_const))
    print("Linear:     {0:>6.2f}".format(err_exp_train_linear))
    print("Quadratic:  {0:>6.2f}".format(err_exp_train_quad))
    print("Cubic:      {0:>6.2f}".format(err_exp_train_cubic))
    print("Quartic:    {0:>6.2f}".format(err_exp_train_quart))
    print("Quintic:    {0:>6.2f}".format(err_exp_train_quint))

    # Estimate expected in-sample error
    err_exp_in_const  = sum([errs_in_const[b]  for b in range(B)])/B
    err_exp_in_linear = sum([errs_in_linear[b] for b in range(B)])/B
    err_exp_in_quad   = sum([errs_in_quad[b]   for b in range(B)])/B
    err_exp_in_cubic  = sum([errs_in_cubic[b]  for b in range(B)])/B
    err_exp_in_quart  = sum([errs_in_quart[b]  for b in range(B)])/B
    err_exp_in_quint  = sum([errs_in_quint[b]  for b in range(B)])/B

    print("")
    print("Estimates of expected in-sample error")
    print("-------------------------------------")
    print("Constant:   {0:>6.2f}".format(err_exp_in_const))
    print("Linear:     {0:>6.2f}".format(err_exp_in_linear))
    print("Quadratic:  {0:>6.2f}".format(err_exp_in_quad))
    print("Cubic:      {0:>6.2f}".format(err_exp_in_cubic))
    print("Quartic:    {0:>6.2f}".format(err_exp_in_quart))
    print("Quintic:    {0:>6.2f}".format(err_exp_in_quint))

    # Estimate expected test error
    err_exp_test_const  = sum([errs_test_const[b]  for b in range(B)])/B
    err_exp_test_linear = sum([errs_test_linear[b] for b in range(B)])/B
    err_exp_test_quad   = sum([errs_test_quad[b]   for b in range(B)])/B
    err_exp_test_cubic  = sum([errs_test_cubic[b]  for b in range(B)])/B
    err_exp_test_quart  = sum([errs_test_quart[b]  for b in range(B)])/B
    err_exp_test_quint  = sum([errs_test_quint[b]  for b in range(B)])/B

    print("")
    print("Estimates of expected test error")
    print("--------------------------------")
    print("Constant:   {0:>6.2f}".format(err_exp_test_const))
    print("Linear:     {0:>6.2f}".format(err_exp_test_linear))
    print("Quadratic:  {0:>6.2f}".format(err_exp_test_quad))
    print("Cubic:      {0:>6.2f}".format(err_exp_test_cubic))
    print("Quartic:    {0:>6.2f}".format(err_exp_test_quart))
    print("Quintic:    {0:>6.2f}".format(err_exp_test_quint))

    #################
    ###   Plots   ###
    #################

    # Example training set
    b0 = 2
    vals_x = [i/100 for i in range(1001)]
    vals_f = [f(x) for x in vals_x]
    vals_const  = [betas_const["beta0"][b0] for x in vals_x]
    vals_linear = [betas_linear["beta0"][b0] + betas_linear["beta1"][b0] * x for x in vals_x]
    vals_quad   = [betas_quad["beta0"][b0]   + betas_quad["beta1"][b0]   * x + betas_quad["beta2"][b0]  * x**2 for x in vals_x]
    vals_cubic  = [betas_cubic["beta0"][b0]  + betas_cubic["beta1"][b0]  * x + betas_cubic["beta2"][b0] * x**2 + betas_cubic["beta3"][b0] * x**3 for x in vals_x]
    vals_quart  = [betas_quart["beta0"][b0]  + betas_quart["beta1"][b0]  * x + betas_quart["beta2"][b0] * x**2 + betas_quart["beta3"][b0] * x**3 + betas_quart["beta4"][b0] * x**4 for x in vals_x]
    vals_quint  = [betas_quint["beta0"][b0]  + betas_quint["beta1"][b0]  * x + betas_quint["beta2"][b0] * x**2 + betas_quint["beta3"][b0] * x**3 + betas_quint["beta4"][b0] * x**4 + betas_quint["beta5"][b0] * x**5 for x in vals_x]

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(vals_x, vals_f,      color="black",      linewidth=1.5, linestyle="dashed", alpha=1,   label=r"$f(x) = x^2 - 8x + 20$", zorder=1)
    ax.plot(vals_x, vals_const,  color="red",        linewidth=1.5, linestyle="solid",  alpha=0.5, label="Constant Fit",            zorder=2)
    ax.plot(vals_x, vals_linear, color="gold",       linewidth=1.5, linestyle="solid",  alpha=0.5, label="Linear Fit",              zorder=3)
    ax.plot(vals_x, vals_quad,   color="lime",       linewidth=1.5, linestyle="solid",  alpha=0.5, label="Quadratic Fit",           zorder=4)
    ax.plot(vals_x, vals_cubic,  color="dodgerblue", linewidth=1.5, linestyle="solid",  alpha=0.5, label="Cubic Fit",               zorder=5)
    ax.plot(vals_x, vals_quart,  color="magenta",    linewidth=1.5, linestyle="solid",  alpha=0.5, label="Quartic Fit",             zorder=6)
    ax.plot(vals_x, vals_quint,  color="sienna",     linewidth=1.5, linestyle="solid",  alpha=0.5, label="Quintic Fit",             zorder=7)
    ax.scatter(xs_train[b0], ys_train[b0], color="black", zorder=8)
    ax.set_title("Example Training Set (n={})".format(n))
    ax.set_xlabel("X")
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("Y", rotation="horizontal")
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    plt.legend(loc="upper left")
    plt.savefig("ml_model_errors_fig1.png", dpi=200)

    # Training and test error as a function of model complexity/flexibility
    fig, ax = plt.subplots(figsize=(8, 8))
    for b in range(100):
        ax.plot([0, 1, 2, 3, 4, 5], [ errs_test_const[b],  errs_test_linear[b],  errs_test_quad[b],  errs_test_cubic[b],  errs_test_quart[b],  errs_test_quint[b]], linewidth=0.5, alpha=0.1, color="tab:orange")
        #ax.plot([0, 1, 2, 3, 4, 5], [  errs_in_const[b],     errs_in_linear[b],    errs_in_quad[b],    errs_in_cubic[b],    errs_in_quart[b],    errs_in_quint[b]], linewidth=0.5, alpha=0.1, color="tab:olive")
        ax.plot([0, 1, 2, 3, 4, 5], [errs_train_const[b], errs_train_linear[b], errs_train_quad[b], errs_train_cubic[b], errs_train_quart[b], errs_train_quint[b]], linewidth=0.5, alpha=0.1, color="tab:blue")
    ax.plot([], [], linewidth=1, color="tab:orange", label="Test Error")
    #ax.plot([], [], linewidth=1, color="tab:olive",  label="In-Sample Error")
    ax.plot([], [], linewidth=1, color="tab:blue",   label="Training Error")
    ax.plot([0, 1, 2, 3, 4, 5], [ err_exp_test_const,  err_exp_test_linear,  err_exp_test_quad,  err_exp_test_cubic,  err_exp_test_quart,  err_exp_test_quint], linewidth=3, color="tab:orange", label="Expected Test Error")
    #ax.plot([0, 1, 2, 3, 4, 5], [   err_exp_in_const,    err_exp_in_linear,    err_exp_in_quad,    err_exp_in_cubic,    err_exp_in_quart,    err_exp_in_quint], linewidth=3, color="tab:olive",  label="Expected In-Sample Error")
    ax.plot([0, 1, 2, 3, 4, 5], [err_exp_train_const, err_exp_train_linear, err_exp_train_quad, err_exp_train_cubic, err_exp_train_quart, err_exp_train_quint], linewidth=3, color="tab:blue",   label="Expected Training Error")
    ax.plot([0, 5], [sigma2, sigma2], linewidth=1, linestyle="dashed", color="black", label="Noise Variance " + r"$\sigma^2$")
    ax.set_title("Simulation Training and Test Error")
    ax.set_xlabel("Polynomial Regression Model Complexity/Flexibility", labelpad=10)
    ax.set_xlim(-0.1, 5.1)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["Constant", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"])
    ax.set_ylabel("Mean\nSquared\nError", rotation="horizontal", va="center", labelpad=20)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    plt.legend(loc="upper right")
    plt.savefig("ml_model_errors_fig2.png", dpi=200)

    print("")
    return

if __name__ == "__main__":
    main()
