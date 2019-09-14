package runners

import (
    "fmt"
    "github.com/buger/goterm"
    "github.com/gokadin/ann-core/core"
    "log"
    "math"
)

const (
    VerboseLevelLow = 0
    VerboseLevelMedium = 1
    VerboseLevelHigh = 2
)

type NetworkRunner struct {
    learningRate float64
    batchSize int
    maxError float64
    validOutputRange float64
    verboseLevel int
}

func NewNetworkRunner() *NetworkRunner {
    return &NetworkRunner{
        batchSize: 1,
        learningRate: 0.01,
        maxError: 0.001,
        validOutputRange: 0.1,
        verboseLevel: VerboseLevelMedium,
    }
}

func (nr *NetworkRunner) Train(network *core.Network, inputs, expected [][]float64) {
    nr.validateTrain(network, inputs, expected)

    fmt.Println("Beginning training of", len(inputs), "associations")

    epochCounter := 0
    batchPerEpoch := len(inputs) / nr.batchSize
    for {
        epochCounter++
        batchCounter := 0
        associationCounter := 0
        err := 0.0

        nr.logEpoch(epochCounter)

        for i, input := range inputs {
            associationCounter++
            network.Activate(input)
            calculateDeltas(network, expected[i])
            accumulateGradients(network)
            err += math.Pow(accumulateError(network, expected[i]), 2)

            if associationCounter == nr.batchSize {
                batchCounter++
                associationCounter = 0
                updateWeights(network, nr.learningRate)

                err /= 2 * float64(nr.batchSize)
                if err < nr.maxError {
                	goterm.Println()
                    return
                }

                nr.logBatchProgress(batchCounter, batchPerEpoch, err)
                err = 0.0
            }
        }
        goterm.Println()
    }
}

func (nr NetworkRunner) logEpoch(epochCounter int) {
    if nr.verboseLevel == VerboseLevelLow {
        return
    }

    fmt.Println("Epoch", epochCounter)
}

func (nr NetworkRunner) logBatchProgress(batchCounter, batchPerEpoch int, err float64) {
    if nr.verboseLevel == VerboseLevelLow {
        return
    }

    if nr.verboseLevel == VerboseLevelMedium {
        fmt.Printf("%d/%d   error: %f", batchCounter, batchPerEpoch, err)
        return
    }

    goterm.MoveCursorBackward(100)
    goterm.Printf("%d/%d   ", batchCounter, batchPerEpoch)
    progressStr := "[";
    progressBars := batchCounter * 30 / batchPerEpoch
    for i := 0; i < progressBars; i++ {
        progressStr += "="
    }
    for i := 0; i < 30 - progressBars; i++ {
        progressStr += "."
    }
    progressStr += "]   "
    goterm.Print(progressStr)
    goterm.Print("error:", err)
    goterm.Flush()
}

func (nr *NetworkRunner) Test(network *core.Network, inputs, expected [][]float64) {
    nr.validateTest(network, inputs, expected)

    err := 0.0
    accuracyCount := 0
	for i, input := range inputs {
		network.Activate(input)

		err += math.Pow(accumulateError(network,  expected[i]), 2)

		isValid := true
		for j, node := range network.OutputLayer().Nodes() {
            if node.Output() < expected[i][j] - nr.validOutputRange ||
                node.Output() > expected[i][j] + nr.validOutputRange {
                isValid = false
                break
            }
        }
		if isValid {
            accuracyCount++
        }
	}

	err /= 2 * float64(len(inputs))
	accuracy := accuracyCount * 100 / len(inputs)
	fmt.Println("Error:", err, "Accuracy:", accuracy, "%")
}

func (nr *NetworkRunner) validateTest(network *core.Network, inputs, expected [][]float64) {

}

func (nr *NetworkRunner) validateTrain(network *core.Network, inputs, expected [][]float64) {
    if nr.batchSize == 0 || nr.batchSize > len(inputs) {
        log.Fatal("batch size is incompatible with the given training set")
    }
}

func (nr *NetworkRunner) SetBatchSize(batchSize int) {
    nr.batchSize = batchSize
}

func (nr *NetworkRunner) SetLearningRate(learningRate float64) {
    nr.learningRate = learningRate
}

func (nr *NetworkRunner) SetMaxError(maxError float64) {
    nr.maxError = maxError
}

func (nr *NetworkRunner) SetValidOutputRange(validOutputRange float64) {
    nr.validOutputRange = validOutputRange
}

func (nr *NetworkRunner) SetVerboseLevel(verboseLevel int) {
    nr.verboseLevel = verboseLevel
}
