package main

import "fmt"

// Optimizer : A type providing all the necessary methods to iterate
// in an optimization problem.
type Optimizer interface {
	move(current *Point, pas float64) Point
	getCurrent() *Point
	checkConverged(itNumber uint) bool
}

// ##############################################################
// Then follow all the specialized optimizers
// ##############################################################

// MonoGradientDescent : A simple optimizer which only considers the first
// objective function, and moves following its -gradient to valleys.
type MonoGradientDescent struct {
	current   *Point  // Starting point
	function  int     // function of the multiobjective problem to use (use 0 if mono-objective)
	tolerance float64 // min gradient norm to continue iterating
	maxit     uint    // max number of iterations before halt
}

func (o *MonoGradientDescent) move(current *Point, pas float64) Point {
	var x = make([]float64, len(current.inputs))
	copy(x, current.inputs)

	// We loop on the variables (axis)
	for i := range x {
		// the descent direction on axis i is -dF1/dxi
		g, err := current.gradient.At(o.function, i)
		if err != nil {
			panic(err)
		}
		x[i] -= pas * g.(float64)
	}
	// fmt.Println("--------------------------------------------------------------")
	// fmt.Printf("--> Now moving from %.2f to %.2f\n", current.inputs, x)

	var pt = current.Problem.evaluate(x)
	o.current = &pt
	return pt
}

func (o *MonoGradientDescent) getCurrent() *Point {
	return o.current
}

func (o *MonoGradientDescent) checkConverged(itNumber uint) bool {

	// Check if we ran for too long
	if itNumber > o.maxit {
		fmt.Printf("Stopping without convergence after %d iterations. =(\n", o.maxit)
		return true
	}

	// Check if any of the functions have a null gradient vector
	for f := 0; f < p.nDims; f++ {
		gradfNorm, _ := o.current.gradNorm.At(f)
		if gradfNorm.(float64) <= o.tolerance {
			fmt.Printf("Gradient of norm %f is below the tolerance threshold (%f), let's stop, we converged!\n", gradfNorm, o.tolerance)
			return true
		}
		// fmt.Printf("Gradient of norm %.2f passes the tolerance test, continue\n", gradfNorm)
	}

	return false
}
