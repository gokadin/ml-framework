package runners

import (
    "fmt"
    "github.com/gokadin/ml-framework/core"
    "github.com/gokadin/ml-framework/tensor"
    "time"
)

const (
    defaultLearningRate = 0.01
    defaultBatchSize = 1
    defaultEpochs = 0
    defaultMaxError = 0.001
    defaultValidOutputRange = 0.1
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
        learningRate: defaultLearningRate,
        batchSize: defaultBatchSize,
        epochs: defaultEpochs,
        maxError: defaultMaxError,
        validOutputRange: defaultValidOutputRange,
    }
}

func (nr *NetworkRunner) Train(network *core.Network, inputs, target *tensor.Tensor) {
    sgd := NewSGD(network)
    criterion := NewCriterion(lossFunctionMeanSquared, target)
    var lossMean float64
    coefficient := nr.learningRate / 4.0//float64(nr.batchSize)

    var aveTime int64
    t := time.Now().UnixNano()
	for i := 1; i != nr.epochs; i++ {
        pred := network.Forward(inputs)
        loss := criterion.Forward(pred)
        sgd.Step(loss, coefficient)

        lossMean = loss.Data()[0][0] / float64(nr.batchSize)
        if i % 10000 == 0 {
            fmt.Println("Epoch", i, "finished with error", lossMean)
            t2ms := (time.Now().UnixNano() - t) / int64(time.Millisecond)
            aveTime += t2ms
            t = time.Now().UnixNano()
        }
        if lossMean < nr.maxError {
            fmt.Println("Finished in", i, "loss:", lossMean)
            fmt.Println(aveTime / int64(i / 10000))
            break
        }
    }
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
