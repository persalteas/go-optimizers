package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/optimize"
)

func solve(p *optimize.Problem, current []float64, settings *optimize.Settings, method optimize.Method) {
	result, err := optimize.Minimize(*p, current, settings, method)
	if err != nil {
		log.Fatal(err)
	}
	if err = result.Status.Err(); err != nil {
		log.Fatal(err)
	}

	s := result.Status
	stats := result.Stats

	if s.Early() {
		fmt.Printf("%v: %v\n", s.String(), s.Err())
	}
	fmt.Printf("Reached F(%0.4g) = %0.4g in %v\n", result.X, result.F, stats.Runtime)
	fmt.Printf("Finished in %d iterations\n", stats.MajorIterations)
}
