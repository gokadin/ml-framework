package runners

import (
    "fmt"
    "github.com/gokadin/ml-framework/core"
    "github.com/gokadin/ml-framework/tensor"
    "log"
    "time"
)

const (
    defaultBatchSize = 1
    defaultEpochs = 0
    defaultMaxError = 0.001
    defaultValidOutputRange = 0.1
    defaultOptimizerType = OptimizerDefault
)

type NetworkRunner struct {
    batchSize int
    epochs int
    maxError float64
    validOutputRange float64
    optimizer string
    optimizerOverrides []float64
}

func NewNetworkRunner() *NetworkRunner {
    return &NetworkRunner{
        batchSize: defaultBatchSize,
        epochs: defaultEpochs,
        maxError: defaultMaxError,
        validOutputRange: defaultValidOutputRange,
        optimizer: defaultOptimizerType,
        optimizerOverrides: make([]float64, 0),
    }
}

func (nr *NetworkRunner) Train(network *core.Network, inputs, target *tensor.Tensor) {
    sgd := NewSGD(network, nr.createOptimizer())
    criterion := NewCriterion(lossFunctionMeanSquared)
    numBatches := len(inputs.Data()) / nr.batchSize

    var aveTime int64 = 1
    t := time.Now().UnixNano()
	for i := 1; i != nr.epochs; i++ {
		lossMean := 0.0
		shuffleDataset(inputs.Data(), target.Data())
		for batchCounter := 0; batchCounter < numBatches; batchCounter++ {
		    batchInputs := tensor.NewTensor(partitionData(inputs.Data(), batchCounter, nr.batchSize))
            batchTarget := tensor.NewTensor(partitionData(target.Data(), batchCounter, nr.batchSize))

            pred := network.Forward(batchInputs)
            loss := criterion.Forward(pred, batchTarget)
            lossMean += loss.Data()[0][0]
            loss.Backward()
            sgd.Step(nr.batchSize, i * nr.batchSize + batchCounter)
        }

        lossMean /= float64(nr.batchSize)
        if i % 10000 == 0 {
            fmt.Println("Epoch", i, "finished with error", lossMean)
            t2ms := (time.Now().UnixNano() - t) / int64(time.Millisecond)
            aveTime += t2ms
            t = time.Now().UnixNano()
        }
        if lossMean < nr.maxError {
            fmt.Println("Finished in", i, "loss:", lossMean)
            div := int64(i / 10000)
            if div == 0 {
                div = 1
            }
            fmt.Println(aveTime / div)
            break
        }
    }
}

func (nr *NetworkRunner) createOptimizer() optimizer {
    switch nr.optimizer {
    case OptimizerDefault:
        return newDefaultOptimizer(nr.optimizerOverrides)
    case OptimizerMomentum:
        return newMomentumOptimizer(nr.optimizerOverrides)
    case OptimizerAdam:
        return newAdamOptimizer(nr.optimizerOverrides)
    }

    log.Fatal("Unknown optimizer selected:", nr.optimizer)
    return nil
}

func (nr *NetworkRunner) SetBatchSize(batchSize int) {
    nr.batchSize = batchSize
}

func (nr *NetworkRunner) SetEpochLimit(epochs int) {
    nr.epochs = epochs
}

func (nr *NetworkRunner) SetMaxError(maxError float64) {
    nr.maxError = maxError
}

func (nr *NetworkRunner) SetValidOutputRange(validOutputRange float64) {
    nr.validOutputRange = validOutputRange
}

func (nr *NetworkRunner) SetOptimizer(optimizerType string, overrides ...float64) {
    nr.optimizer = optimizerType
    nr.optimizerOverrides = overrides
}
