package main

import (
	"math"

	"gorgonia.org/tensor"
)

// // tensor.Dense Alias for tensor.Dense
// type tensor.Dense tensor.Dense

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
