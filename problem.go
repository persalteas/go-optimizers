package main

import (
	"math"

	"gorgonia.org/tensor"
)

// ########################################################
// 		DEFINE YOUR PROBLEM'S FUNCTIONS HERE
// ########################################################

// Problem The struct which stores the optimisation problem definition,
// by defining its function and derivatives
type Problem struct {
	nVars     int
	nDims     int
	f         func([]float64) *tensor.Dense
	jacobianf func([]float64) *tensor.Dense
	hessianf  func([]float64) *tensor.Dense
	equations *[]string
}

// Replace the numbers below to match the number of variables and number of objective functions in your problem.
var eq = []string{"(x-y)**3+2*x**2+y**2-x+2*y-500", "x**4 - x**3 -20*x**2 + x + y**4 - y**3 -20*y**2 + y - 100"}
var p = Problem{2, 2, f, jacobianf, hessianf, &eq}

// The function to minimize, R^M -> R^N
func f(v []float64) *tensor.Dense {
	x, y := v[0], v[1]
	xsq := x * x
	ysq := y * y

	f1 := math.Pow(x-y, 3) + 2.0*xsq + ysq - x + 2.0*y - 500.0
	f2 := xsq*xsq - xsq*x - 20.0*xsq + x + math.Pow(ysq, 2) - ysq*y - 20.0*ysq + y - 100
	var result = tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{f1, f2}))

	return result
}

// Its gradients (partial derivatives) at some point, R^M -> M(N, M)
func jacobianf(v []float64) *tensor.Dense {
	x, y := v[0], v[1]
	xsq := x * x
	ysq := y * y

	df1dx := 3.0*xsq - 2.0*x*y + 3.0*ysq - 1.0
	df2dx := 4.0*xsq*x - 3.0*xsq - 40.0*x + 1.0
	df1dy := -3.0*ysq - 3.0*xsq + 6.0*x*y + 2.0*y + 2.0
	df2dy := 4.0*ysq*y - 3.0*ysq - 40.0*y + 1.0
	var result = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{df1dx, df1dy, df2dx, df2dy}))

	return result
}

// Its Hessian matrix computed at some point, R^M -> T(N,M,M)
func hessianf(v []float64) *tensor.Dense {
	x, y := v[0], v[1]

	df1dxdx := 6.0*x - 2.0*y
	df1dxdy := -2.0*x + 6.0*y
	df1dydx := df1dxdy
	df1dydy := -6.0*y + 6.0*x + 2.0
	df2dxdx := 12.0*x*x - 6.0*x - 40.0
	df2dxdy := 0.0
	df2dydx := df1dxdy
	df2dydy := 12.0*y*y - 6.0*y - 40.0
	var result = tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{df1dxdx, df1dxdy, df1dydx, df1dydy, df2dxdx, df2dxdy, df2dydx, df2dydy}))

	return result
}

// Getter for the number of dimensions of the optimisation problem
func (p *Problem) getDims() (int, int) {
	return p.nVars, p.nDims
}

// ##############################################################
// 			OR USE ONE OF THESE KNOWN FUNCTIONS
// ##############################################################

// Code copy pasted and adapted without permission from gonum/optimize/functions.

var bealeEq = []string{"(1.5-x+x*y)**2+(2.25-x+x*y*y)**2+(2.625-x+x*y*y*y)**2"}

// BealeProblem implements the Beale's function.
var BealeProblem = Problem{2, 1, beale, jacobianBeale, hessianBeale, &bealeEq}

//
// Standard starting points:
//  Easy: [1, 1]
//  Hard: [1, 4]
//
// References:
//  - Beale, E.: On an Iterative Method for Finding a Local Minimum of a
//    Function of More than One Variable. Technical Report 25, Statistical
//    Techniques Research Group, Princeton University (1958)
//  - More, J., Garbow, B.S., Hillstrom, K.E.: Testing unconstrained
//    optimization software. ACM Trans Math Softw 7 (1981), 17-41

func beale(x []float64) *tensor.Dense {
	f1 := 1.5 - x[0]*(1-x[1])
	f2 := 2.25 - x[0]*(1-x[1]*x[1])
	f3 := 2.625 - x[0]*(1-x[1]*x[1]*x[1])
	var result = tensor.New(tensor.WithShape(1, 1), tensor.WithBacking([]float64{f1*f1 + f2*f2 + f3*f3}))

	return result
}

func jacobianBeale(x []float64) *tensor.Dense {
	t1 := 1 - x[1]
	t2 := 1 - x[1]*x[1]
	t3 := 1 - x[1]*x[1]*x[1]

	f1 := 1.5 - x[0]*t1
	f2 := 2.25 - x[0]*t2
	f3 := 2.625 - x[0]*t3

	grad1 := -2 * (f1*t1 + f2*t2 + f3*t3)
	grad2 := 2 * x[0] * (f1 + 2*f2*x[1] + 3*f3*x[1]*x[1])
	var result = tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{grad1, grad2}))
	return result
}

func hessianBeale(x []float64) *tensor.Dense {
	t1 := 1 - x[1]
	t2 := 1 - x[1]*x[1]
	t3 := 1 - x[1]*x[1]*x[1]
	f1 := 1.5 - x[1]*t1
	f2 := 2.25 - x[1]*t2
	f3 := 2.625 - x[1]*t3

	h00 := 2 * (t1*t1 + t2*t2 + t3*t3)
	h01 := 2 * (f1 + x[1]*(2*f2+3*x[1]*f3) - x[0]*(t1+x[1]*(2*t2+3*x[1]*t3)))
	h11 := 2 * x[0] * (x[0] + 2*f2 + x[1]*(6*f3+x[0]*x[1]*(4+9*x[1]*x[1])))
	var result = tensor.New(tensor.WithShape(1, 2, 2), tensor.WithBacking([]float64{h00, h01, h01, h11}))
	return result
}
