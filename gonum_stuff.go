package main

import (
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/optimize/functions"
)

func solveMonoObjectiveProblem() {

	// #############################################
	// Define a problem
	// #############################################

	// problem := optimize.Problem{
	// 	Func: func(x []float64) float64 {},
	// 	Grad: func(grad, x []float64) {},
	// }
	problem := optimize.Problem{
		Func: functions.ExtendedRosenbrock{}.Func,
		Grad: functions.ExtendedRosenbrock{}.Grad,
	}

	// #############################################
	// Define a starting point
	// #############################################
	x := []float64{1.3, 0.7, 0.8, 1.9, 1.2}

	// #############################################
	// Define a method
	// #############################################

	// Linesearchers
	// ls := optimize.Backtracking{} // Search for Armijo condition. Few gradient evaluations.
	// ls := optimize.Bisection{}    // Search for weak Wolfe conditions
	ls := optimize.MoreThuente{} // Search for strong Wolfe conditions. More minor iterations, solution closer to the local minima at each major.

	// Optimizers
	// m := optimize.NelderMead{}                       // simplex algorithm for gradient-free NLP
	// m := optimize.Newton{Linesearcher: &ls}          // modified Newton's method. Applies regularization when the Hessian is not positive definite.
	// m := optimize.BFGS{Linesearcher: &ls}            // quasi-Newton method, o(n²) in memory
	m := optimize.LBFGS{Linesearcher: &ls} // idem, lower memory complexity for large problems
	// m := optimize.CG{Linesearcher: &ls}              // nonlinear conjugate gradient
	// m := optimize.CmaEsChol{}                        // covariance matrix adaptation evolution strategy (CMA-ES) based on the Cholesky decomposition
	// m := optimize.GradientDescent{Linesearcher: &ls} // steepest descent
	// m := optimize.GuessAndCheck{}                    // Evaluates the function at random points (for comparison purposes)
	// m := optimize.ListSearch{}                       // Find the optimum in a user-provded list of locations

	c := optimize.FunctionConverge{
		Absolute:   1e-6,
		Relative:   0.001, // 0.1%
		Iterations: 100,
	} // Stops if after Iterations iterations, the improvement is less than Absolute and less than Relative of the objective value.
	s := optimize.Settings{
		InitValues:        nil,  // Use nil, the function will be evaluated at the starting point
		GradientThreshold: 1e-6, // stop if all gradients are smaller
		Converger:         &c,   // Stop cirterions, evaluated after every major iteration
		MajorIterations:   0,    // No limit
		Runtime:           0,    // No limit
		FuncEvaluations:   0,    // No limit
		GradEvaluations:   0,    // No limit
		HessEvaluations:   0,    // No limit
		Recorder:          nil,
		Concurrent:        2, // 2 evaluations max in parallel
	}

	// #############################################
	// solve it
	// #############################################

	solve(&problem, x, &s, &m)
}
