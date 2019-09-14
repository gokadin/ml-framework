package runners

import (
    "fmt"
    "github.com/gokadin/ml-framework/core"
    "github.com/gokadin/ml-framework/tensor"
)

type NetworkRunner struct {
    learningRate float64
    batchSize int
    epochs int
    maxError float64
    validOutputRange float64
}

func NewNetworkRunner() *NetworkRunner {
    return &NetworkRunner{
        learningRate: 0.01,
        batchSize: 1,
        epochs: -1,
        maxError: 0.001,
        validOutputRange: 0.1,
    }
}

func (nr *NetworkRunner) Train(network *core.Network, inputs, target *tensor.Tensor) {
    sgd := NewSGD(network)
    criterion := NewCriterion(lossFunctionMeanSquared, target)
    lossMean := 1.0
    iterations := 0
    for lossMean > nr.maxError {
        //for i := 0; i < 10; i++ {
        pred := network.Forward(inputs)
        loss := criterion.Forward(pred)
        loss.Backward()
        sgd.Step(nr.learningRate, len(inputs.Data()))
        lossMean = loss.Data()[0][0] / 4.0
        if iterations % 10000 == 0 {
            fmt.Println("Epoch", iterations, "finished with error", lossMean)
        }
        iterations++
    }

    fmt.Println("Finished in", iterations, "loss:", lossMean)
}

func (nr *NetworkRunner) SetBatchSize(batchSize int) {
    nr.batchSize = batchSize
}

func (nr *NetworkRunner) SetEpochLimit(epochs int) {
    nr.epochs = epochs
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
