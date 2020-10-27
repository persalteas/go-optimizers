package main

import (
	// "fmt"

	"gorgonia.org/tensor"
)

// // tensor.Dense Alias for tensor.Dense
// type tensor.Dense tensor.Dense

// Point A struct to store all the information about a point in the problem's spaces.
type Point struct {
	inputs   []float64
	images   *tensor.Dense
	gradient *tensor.Dense
	gradNorm *tensor.Dense
	Problem  *Problem
}

func (p *Problem) evaluate(newInputs []float64) Point {
	var grad = p.jacobianf(newInputs) // M(N,M)
	var norm, _ = grad.Norm(2, 1)     // order 2 norms on axis 0, should be in R^N
	var pt = Point{
		newInputs,
		p.f(newInputs),
		grad,
		norm,
		p,
	}
	// fmt.Printf("New point X: F(%.2f) = %.2f (grad norms are now: %.2f)\n", pt.inputs, pt.images, pt.gradNorm)
	// fmt.Println()
	// fmt.Printf("Jacobian in %.2f:\n%.2f", pt.inputs, pt.gradient)
	// fmt.Println()
	return pt
}
