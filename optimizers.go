package main

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/optimize"
	"gorgonia.org/tensor"
)

// Optimizer : A type providing all the necessary methods to iterate
// in an optimization problem.
type Optimizer interface {
	move(current *Point) Point
	getCurrent() *Point
	checkConverged(itNumber uint) bool
}

// ##############################################################
// Then follow all the specialized optimizers
// ##############################################################

// MonoGradientDescent : A simple optimizer which only considers the first
// objective function, and moves following its -gradient to valleys.
type MonoGradientDescent struct {
	current    *Point  // Starting point
	function   int     // function of the multiobjective problem to use (use 0 if mono-objective)
	tolerance  float64 // min gradient norm to continue iterating
	maxit      uint    // max number of iterations before halt
	stepLength float64 // how much we move at every iteration
}

func (o *MonoGradientDescent) move(current *Point) Point {
	var x = make([]float64, len(current.inputs))
	copy(x, current.inputs)

	// We loop on the variables (axis)
	for i := range x {
		// the descent direction on axis i is -dF1/dxi
		g, err := current.gradient.At(o.function, i)
		if err != nil {
			panic(err)
		}
		x[i] -= o.stepLength * g.(float64)
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
	}

	return false
}

// SteepestDescent : A multiobjective gradient descent chosing the steepest direction
// to decrease the value of an objective.
type SteepestDescent struct {
	current          *Point  // starting point
	tolerance        float64 // min gradient norm to continue iteraring
	maxit            uint    // max number of iterations before halt
	criticalDetected bool    // if we cannot find a descent direction anymore
	stepLength       float64 // how much we move at every iteration
}

func (o *SteepestDescent) move(current *Point) Point {
	var x = make([]float64, len(current.inputs))
	copy(x, current.inputs)

	// We search for a descent direction.
	// Define the optimization problem
	problem := optimize.Problem{
		Func: func(x []float64) float64 {
			var direction = make([]float64, len(x))
			copy(direction, x)

			// transform the input direction into a vector
			var d = tensor.New(tensor.WithShape(len(x)), tensor.WithBacking(direction))
			dNorm := toFloat(d.Norm(2, 1))
			fmt.Printf("\tVector %f has squared norm %f.\n", d, dNorm*dNorm)

			// Get the slope (or "Frechet derivative, look how i'm smart and i know words")
			// Of every function in direction d
			slope, _ := current.gradient.MatVecMul(d)
			fmt.Printf("\tGradient in this direction: %.2f\n", slope)

			// Get the steepest slope
			a := toFloat(slope.Max(1))

			// return the unconstrained optimization problem value
			fmt.Printf("\tValue is %.2f + %.2f = %.2f\n\n", a, 0.5*dNorm*dNorm, a+0.5*dNorm*dNorm)
			return a + 0.5*dNorm*dNorm
		},

		Grad: func(grad, x []float64) {
			copy(grad, x) // dProblem/dd = dd for all indices that are not the steepest descent

			// transform the input direction into a vector
			var d = tensor.New(tensor.WithShape(len(x)), tensor.WithBacking(grad)) // d1, d2, ..., dm
			slope, _ := current.gradient.MatVecMul(d)                              // sum_j(dFk/dj*dj)

			// Get the steepest slope index
			temp, _ := slope.Argmax(0)
			j := temp.Get(0).(int)
			grad[j] += slope.Get(j).(float64)
			fmt.Printf("\tGradient of the problem in %f is %f, its maximum is %f.\n", x, grad, grad[j])
		},
	}

	// Propose a direction descent as init value
	var init = make([]float64, p.nVars)
	for i := range init {
		init[i] = -1
	}

	// Solve the optimization problem
	settings := optimize.Settings{InitValues: nil, GradientThreshold: o.tolerance, Converger: nil, MajorIterations: 100000, Runtime: 0, FuncEvaluations: 0, GradEvaluations: 0, HessEvaluations: 0, Recorder: nil, Concurrent: 2}
	result, _ := optimize.Minimize(problem, init, &settings, &optimize.NelderMead{})
	if result.F > 0 {
		o.criticalDetected = true
	}
	fmt.Printf("Found descent direction: %v (problem value = %f)\n", result.X, result.F)

	// Now, update the point
	// We loop on the variables (axis)
	for i := range x {
		x[i] = o.stepLength * result.X[i]
	}
	// fmt.Println("--------------------------------------------------------------")
	// fmt.Printf("--> Now moving from %.2f to %.2f\n", current.inputs, x)

	var pt = current.Problem.evaluate(x)
	o.current = &pt
	time.Sleep(time.Second)
	return pt
}

func (o *SteepestDescent) getCurrent() *Point {
	return o.current
}

func (o *SteepestDescent) checkConverged(itNumber uint) bool {

	// Check if we can't find descent directions anymore
	if o.criticalDetected {
		fmt.Printf("Let's stop, this point is Pareto-critical.\n")
		return true
	}

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
	}

	return false
}
