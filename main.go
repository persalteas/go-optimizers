package main

import (
	"fmt"
	"os"

	// "strings"
	"time"

	"encoding/csv"

	"github.com/Arafatk/glot"
	"gorgonia.org/tensor"
)

func run(o Optimizer, optiIndex int) []Point {
	// Main optimization loop
	var trajectory = []Point{}
	trajectory = append(trajectory, *o.getCurrent()) // Set the first point as the begining of the trajectory
	for !o.checkConverged(uint(len(trajectory) - 1)) {
		next := o.move(o.getCurrent())
		trajectory = append(trajectory, next)
	}
	fmt.Printf("Converged in %d iterations.\n", len(trajectory)-1)
	fmt.Printf("Minimum found at: %.2f\n", o.getCurrent().inputs)
	fmt.Println()
	fmt.Println("Exiting...")
	trajToCSV(&trajectory, optiIndex)
	return trajectory
}

func main() {
	// solveMonoObjectiveProblem()
	solveMultiobjectiveProblem()

}

func solveMultiobjectiveProblem() {
	p = BealeProblem
	var nVars, nDims int = p.getDims() // The problem is defined in the file problem.go
	fmt.Println("Welcome to the IBISC superoptimizer. Time to superoptimize your life.")
	fmt.Printf("Starting with an optimization problem: %d cost functions to minimize, depending on %d variables.\n", nDims, nVars)

	var startingPoint = []float64{1.0, 4.0} // Coordinates in the space of variables
	var first = p.evaluate(startingPoint)

	// Create some Optimizers. Pick your favorite algorithm, see optimizers.go for the list.
	var stopTolerance float64 = 0.000001
	var myOptis = []Optimizer{
		&MonoGradientDescent{&first, 0, stopTolerance, 10000, 0.01},
		// &MonoGradientDescent{&first, 1, stopTolerance, 10000, 0.01},
		// &SteepestDescent{&first, stopTolerance, 10000, false, 1.0},
	}
	// var names = []string{"Gradient Descent on Function 1", "Gradient Descent on Function 2", "SteepestDescent"}
	var names = []string{"SteepestDescent"}

	// Prepare a plot
	persist := true // Keep the Gnuplot window open
	debug := false  // do not print commands to stdout
	plot3d, _ := glot.NewPlot(3, persist, debug)
	plot2d, _ := glot.NewPlot(2, persist, debug)

	fmt.Println("Starting optimization...")
	var trajectories [][]Point = make([][]Point, len(myOptis))

	for a := 0; a < len(myOptis); a++ {
		trajectories[a] = run(myOptis[a], a)
	}

	plot3dTrajectories(plot3d, &names)
	plot2dTrajectories(plot2d, &names)

	time.Sleep(time.Second * 2)
}

func tryGnuplotCmd(plot *glot.Plot, cmd string) {
	err := plot.Cmd(cmd)
	if err != nil {
		panic(err)
	}
}

func trajToCSV(traj *[]Point, optiIndex int) {
	// Create a CSV file containing our data-points
	// First columns are the variables, the the objective values
	var data = [][]string{}
	for l := 0; l < len(*traj); l++ {
		thisPoint := make([]string, 2+p.nDims)
		thisPoint[0] = fmt.Sprintf("%.2f", (*traj)[l].inputs[0])
		thisPoint[1] = fmt.Sprintf("%.2f", (*traj)[l].inputs[1])
		im := (*traj)[l].images
		for j := 0; j < p.nDims; j++ {
			val, _ := im.At(j, 0)
			thisPoint[2+j] = fmt.Sprintf("%.2f", val)
		}
		data = append(data, thisPoint)
	}
	file, err := os.Create(fmt.Sprintf("trajectory%d.csv", optiIndex+1))
	if err != nil {
		panic(err)
	}

	writer := csv.NewWriter(file)
	writer.Comma = ' '
	for _, value := range data {
		writer.Write(value)
	}
	writer.Flush()
	file.Close()
}

func plot3dTrajectories(plot *glot.Plot, names *[]string) {

	// Prepare the plot
	var cmd string = "set style data linespoints; load 'persalpalette.pal'; "
	cmd += "set xyplane at 0; set xlabel 'x';  set ylabel 'y'; "
	cmd += "set xrange [-5:5]; set yrange [-5:5]; "
	cmd += "set isosample 20; set contour surface; set cntrparam levels 30; unset clabel; "

	// Plot the functions
	cmd += "splot "
	for j, eq := range *p.equations {
		cmd += eq + fmt.Sprintf(" ls %d", j+1) + " title '" + eq + "', "
	}

	// Plot the trajectories
	for k, name := range *names {
		cmd += fmt.Sprintf("'trajectory%d.csv'", k+1) + " using 1:2:(0) " + fmt.Sprintf("lc %d pt 3", k+p.nDims+1) + " title '" + name + "'"

		// Plot projections on functions
		for j := 0; j < p.nDims; j++ {
			cmd += fmt.Sprintf(", '' using 1:2:%d ls %d pt 3 notitle", j+3, j+1)
		}
		if k < len(*names)-1 {
			cmd += ", "
		} else {
			cmd += "; "
		}
	}
	cmd += "pause mouse keypress"

	// fmt.Println(cmd)
	tryGnuplotCmd(plot, cmd)
}

func plot2dTrajectories(plot *glot.Plot, names *[]string) {

	// Prepare the plot
	var cmd string = "set xrange [-5:5]; set yrange [-5:5]; set isosample 100; "

	// Save the contours to dat files
	cmd += "set contour base; set cntrparam levels 50; unset surface; "
	for j, eq := range *p.equations {
		cmd += fmt.Sprintf("set table 'Function%d.dat'; splot ", j+1) + eq + fmt.Sprintf(" title \"Function %d\"", j+1) + "; unset table; "
	}

	// Prepare the final plot
	cmd += "reset; set xrange [-5:5]; set yrange [-5:5]; unset clabel; "
	cmd += "set xlabel 'x';  set ylabel 'y'; set key below; load 'persalpalette.pal'; plot "

	// Plot the contours
	for j, eq := range *p.equations {
		cmd += fmt.Sprintf("'Function%d.dat'", j+1) + fmt.Sprintf(" with lines ls %d title '", j+1) + eq + "', "
	}

	// Plot the trajectory in the space of variables
	for k, name := range *names {
		cmd += fmt.Sprintf("'trajectory%d.csv'", k+1) + " using 1:2 with linespoints " + fmt.Sprintf("lc %d pt 3", k+p.nDims+1) + " title '" + name + "'"
		if k < len(*names)-1 {
			cmd += ", "
		}
	}

	// fmt.Println(cmd)
	tryGnuplotCmd(plot, cmd)
}

func toFloat(t *tensor.Dense, err error) float64 {

	if err != nil {
		panic(err)
	}
	temp := t.Get(0)
	return temp.(float64)
}
