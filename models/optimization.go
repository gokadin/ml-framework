package models

import (
    "github.com/gokadin/ml-framework/mat"
    "github.com/gokadin/ml-framework/tensor"
    "log"
    "math"
)

const (
    OptimizerDefault = "optimizerDefault"
    OptimizerMomentum = "optimizerMomentum"
    OptimizerAdam = "optimizerAdam"

    /* Momentum optimizer */
    defaultLearningRate = 0.01
    defaultMomentum = 0.9

    /* Adam optimizer */
    defaultBeta1 = 0.9
    defaultBeta2 = 0.999
    defaultEpsStable = 1e-08
)

type optimizer interface {
    update(tensor *tensor.Tensor, batchSize, counter int)
}

func newOptimizer(optimizerType string) optimizer {
    switch optimizerType {
    case OptimizerDefault:
        return newDefaultOptimizer([]float64{})
    case OptimizerMomentum:
        return newMomentumOptimizer([]float64{})
    case OptimizerAdam:
        return newAdamOptimizer([]float64{})
    }

    log.Fatal("Unknown optimizer selected:", optimizerType)
    return nil
}

type defaultOptimizer struct {
    learningRate float64
}

func newDefaultOptimizer(overrides []float64) *defaultOptimizer {
    o := &defaultOptimizer{
        learningRate: defaultLearningRate,
    }
    if len(overrides) == 1 {
        o.learningRate = overrides[0]
    }
    return o
}

func (do defaultOptimizer) update(tensor *tensor.Tensor, batchSize, counter int) {
    tensor.Reduce(mat.MulScalar(tensor.Gradient(), do.learningRate / float64(batchSize)))
}

type momentumOptimizer struct {
    learningRate float64
    momentum float64
    velocityMap map[string][][]float64
}

func newMomentumOptimizer(overrides []float64) *momentumOptimizer {
    o := &momentumOptimizer{
        learningRate: defaultLearningRate,
        momentum: defaultMomentum,
        velocityMap: make(map[string][][]float64),
    }
    if len(overrides) >= 1 {
        o.learningRate = overrides[0]
    }
    if len(overrides) == 2 {
        o.momentum = overrides[1]
    }
    return o
}

func (mo momentumOptimizer) update(tensor *tensor.Tensor, batchSize, counter int) {
    if _, ok := mo.velocityMap[tensor.Id()]; !ok {
        mo.velocityMap[tensor.Id()] = mat.CreateMatrix(len(tensor.Data()), len(tensor.Data()[0]))
    }

    mo.velocityMap[tensor.Id()] = mat.Add(mat.MulScalar(mo.velocityMap[tensor.Id()], mo.momentum), mat.MulScalar(tensor.Gradient(), mo.learningRate / float64(batchSize)))
    tensor.Reduce(mo.velocityMap[tensor.Id()])
}

type adamOptimizer struct {
    learningRate float64
    beta1 float64
    beta2 float64
    epsStable float64
    meanMap map[string][][]float64
    velocityMap map[string][][]float64
}

func newAdamOptimizer(overrides []float64) *adamOptimizer {
    o := &adamOptimizer{
        learningRate: defaultLearningRate,
        beta1: defaultBeta1,
        beta2: defaultBeta2,
        epsStable: defaultEpsStable,
        meanMap: make(map[string][][]float64),
        velocityMap: make(map[string][][]float64),
    }
    if len(overrides) >= 1 {
        o.learningRate = overrides[0]
    }
    if len(overrides) >= 2 {
        o.beta1 = overrides[1]
    }
    if len(overrides) >= 3 {
        o.beta2 = overrides[2]
    }
    if len(overrides) >= 4 {
        o.epsStable = overrides[3]
    }
    return o
}

func (ao adamOptimizer) update(tensor *tensor.Tensor, batchSize, counter int) {
    if _, ok := ao.velocityMap[tensor.Id()]; !ok {
        ao.meanMap[tensor.Id()] = mat.CreateMatrix(len(tensor.Data()), len(tensor.Data()[0]))
        ao.velocityMap[tensor.Id()] = mat.CreateMatrix(len(tensor.Data()), len(tensor.Data()[0]))
    }

    g := mat.DivScalar(tensor.Gradient(), float64(batchSize))

    ao.meanMap[tensor.Id()] = mat.Add(mat.MulScalar(ao.meanMap[tensor.Id()], ao.beta1), mat.MulScalar(g, 1 - ao.beta1))
    ao.velocityMap[tensor.Id()] = mat.Add(mat.MulScalar(ao.velocityMap[tensor.Id()], ao.beta2), mat.MulScalar(mat.Pow(g, 2), 1 - ao.beta2))

    biasCorr := mat.DivScalar(ao.meanMap[tensor.Id()], 1 - math.Pow(ao.beta1, float64(counter)))
    sqrtBiasCorr := mat.DivScalar(ao.velocityMap[tensor.Id()], 1 - math.Pow(ao.beta2, float64(counter)))

    tensor.Reduce(mat.Div(mat.MulScalar(biasCorr, ao.learningRate), mat.AddScalar(mat.Sqrt(sqrtBiasCorr), ao.epsStable)))
}
