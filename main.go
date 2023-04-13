package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// states:
//   0: empty
//   1: agent
//   2: food
//   3: eating food

// 2 output
// N hidden
// 2 inputs

var (
	simulations = 5
	simulation  = 0

	epochs          = 20
	epoch           = 0
	secondsPerEpoch = 0.1
	epochHistory    = make1DMatrix(simulations, -1)
)

var (
	toBeUpdated = -99.0

	moveLowerBound = -1.0
	moveUpperBound = 1.0
)

var (
	width  = 30
	height = 10
	enviro = make2DMatrix(width, height, 0)

	agentX = 0
	agentY = 0
	foodX  = 0
	foodY  = 0
	food   = []int{foodX, foodY}
)

var (
	inputCount  = 2
	hiddenCount = 10
	outputCount = 2

	alpha         = 0.01
	biasDeviation = 0.01
)

var (
	inputToHiddenWeights  = make2DMatrix(inputCount, hiddenCount, 0.5)
	hiddenToOutputWeights = make2DMatrix(hiddenCount, outputCount, 0.5)
)

func main() {

	renderMap := false

	for simulation := 0; simulation < simulations; simulation++ {

		// randomize start positions
		agentX = rand.Intn(width)
		agentY = rand.Intn(height)
		foodX = rand.Intn(width)
		foodY = rand.Intn(height)
		for foodX == agentX {
			foodX = rand.Intn(width)
			foodY = rand.Intn(height)
		}

		if renderMap {
			// initial render of map
			updateMap()
			printMap()
			time.Sleep(time.Duration(secondsPerEpoch * float64(time.Second)))
		}

		// run main functions iteratively
		for epoch := 0; epoch < epochs; epoch++ {

			outputActivations := think()
			move(outputActivations)
			learn(outputActivations)

			if renderMap {
				updateMap()
				printMap()
				time.Sleep(time.Duration(secondsPerEpoch * float64(time.Second)))
			}

			if foundFood() {
				break
			}

		}

		epochHistory[simulation] = float64(epoch)

	}

	// analyze simulations
	predictedStart, predictedEnd, improvement := linearReg(epochHistory)
	fmt.Printf("improvement = %d, epochs at start = %d, epochs at end = %d\n", int(improvement), int(predictedStart), int(predictedEnd))

}

func linearReg(data []float64) (float64, float64, float64) {
	n := float64(len(data))
	var sumX, sumY, sumXY, sumX2 float64
	for i := 0; i < len(data); i++ {
		sumX += float64(i)
		sumY += data[i]
		sumXY += float64(i) * data[i]
		sumX2 += float64(i) * float64(i)
	}
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	intercept := (sumY - slope*sumX) / n
	predictedStart := intercept
	predictedEnd := intercept + slope*float64(len(data)-1)
	return predictedStart, predictedEnd, predictedEnd - predictedStart
}

func think() []float64 {

	inputActivations := []float64{float64(foodX), float64(foodY)}
	hiddenActivations := make1DMatrix(hiddenCount, toBeUpdated)
	outputActivations := make1DMatrix(outputCount, toBeUpdated)

	inputToHiddenAxonPotentials := dot(getAxonPotentials(inputActivations, inputToHiddenWeights), makeBiases(hiddenCount))

	hiddenActivations = getPostsynapticActivations(inputToHiddenAxonPotentials)

	hiddenToOutputAxonPotentials := dot(getAxonPotentials(hiddenActivations, hiddenToOutputWeights), makeBiases(outputCount))

	outputActivations = getPostsynapticActivations(hiddenToOutputAxonPotentials)
	return outputActivations

}

func move(outputActivations []float64) {

	moveX := int(clamp(outputActivations[0], moveLowerBound, moveUpperBound))
	moveY := int(clamp(outputActivations[1], moveLowerBound, moveUpperBound))

	agentX = (agentX + moveX) % width
	agentY = (agentY + moveY) % height

}

func learn(outputActivations []float64) {

	// update target
	target := []float64{toBeUpdated, toBeUpdated}
	if agentX < foodX {
		target[0] = 1
	} else if agentX > foodX {
		target[0] = -1
	} else {
		target[0] = 0
	}
	if agentY < foodY {
		target[1] = 1
	} else if agentY > foodY {
		target[1] = -1
	} else {
		target[1] = 0
	}

	// backpropagate

	outputError := minus(outputActivations, target)

	outputDerivative = outputActivations * minus(make1DMatrix(outputCount, 1), outputActivations)

}

func clamp(v, min, max float64) float64 {
	return math.Min(math.Max(v, min), max)
}

func activationFunction(x float64) float64 { // sigmoid
	return 1.0 / (1.0 + math.Exp(-x))
}

func norm(x, mean, stddev float64) float64 {
	exponent := -(math.Pow(x-mean, 2.0) / (2.0 * math.Pow(stddev, 2.0)))
	denominator := stddev * math.Sqrt(2.0*math.Pi)
	return math.Exp(exponent) / denominator
}

func getAxonPotentials(presynapticActivations []float64, weights [][]float64) []float64 {
	N := len(weights)
	M := len(weights[0])
	axonPotentials := make([]float64, N)
	for n := 0; n < N; n++ {
		for m := 0; m < M; m++ {
			axonPotentials[n] += presynapticActivations[m] * weights[m][n]
		}
	}
	return axonPotentials
}

func getPostsynapticActivations(axonPotentials []float64) []float64 {
	N := len(axonPotentials)
	postsynapticActivations := make([]float64, N)
	for n := 0; n < N; n++ {
		postsynapticActivations[n] = activationFunction(axonPotentials[n])
	}
	return postsynapticActivations
}

func foundFood() bool {
	return agentX == foodX && agentY == foodY
}

func updateMap() {
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			if foundFood() {
				enviro[x][y] = 3
			} else if x == agentX && y == agentY {
				enviro[x][y] = 1
			} else if x == foodX && y == foodY {
				enviro[x][y] = 2
			} else {
				enviro[x][y] = 0
			}
		}
	}
}

func make1DMatrix(M int, v float64) []float64 {
	mat := make([]float64, M)
	for x := 0; x < M; x++ {
		mat[x] = v
	}
	return mat
}

func make2DMatrix(M, N int, v float64) [][]float64 {
	mat := make([][]float64, M)
	for x := 0; x < M; x++ {
		mat[x] = make([]float64, N)
		for y := 0; y < N; y++ {
			mat[x][y] = v
		}
	}
	return mat
}

func makeBiases(N int) []float64 {
	biases := make([]float64, N)
	for n := 0; n < N; n++ {
		biases[n] = norm(biasDeviation, 0, 1)
	}
	return biases
}

func dot(v1, v2 []float64) []float64 {
	N := len(v1)
	v := make([]float64, N)
	for n := 0; n < N; n++ {
		v[n] = v1[n] + v2[n]
	}
	return v
}

func minus(v1, v2 []float64) []float64 {
	N := len(v1)
	v := make([]float64, N)
	for n := 0; n < N; n++ {
		v[n] = v1[n] - v2[n]
	}
	return v
}

func times(v1, v2 []float64) []float64 {
	N := len(v1)
	v := make([]float64, N)
	for n := 0; n < N; n++ {
		v[n] = v1[n] * v2[n]
	}
	return v
}

func printMap() {

	fmt.Printf("simulation %d | epoch %d", simulation, epoch)

	printBound := func() {
		fmt.Printf("|")
		for x := 0; x < width; x++ {
			fmt.Printf("-")
		}
		fmt.Printf("|\n")
	}

	printRow := func(y int) {
		fmt.Printf("|")
		for x := 0; x < width; x++ {
			fmt.Printf("%d", int(enviro[x][y]))
		}
		fmt.Printf("|\n")
	}

	printBound()
	for y := 0; y < height; y++ {
		printRow(y)
	}
	printBound()

}
