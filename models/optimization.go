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
    defaultLearningRate float32 = 0.01
    defaultMomentum float32 = 0.9

    /* Adam optimizer */
    defaultBeta1 float32 = 0.9
    defaultBeta2 float32 = 0.999
    defaultEpsStable float32 = 1e-08
)

type optimizer interface {
    update(tensor *tensor.Tensor, batchSize, counter int)
}

func newOptimizer(optimizerType string) optimizer {
    switch optimizerType {
    case OptimizerDefault:
        return newDefaultOptimizer([]float32{})
    case OptimizerMomentum:
        return newMomentumOptimizer([]float32{})
    case OptimizerAdam:
        return newAdamOptimizer([]float32{})
    }

    log.Fatal("Unknown optimizer selected:", optimizerType)
    return nil
}

type defaultOptimizer struct {
    learningRate float32
}

func newDefaultOptimizer(overrides []float32) *defaultOptimizer {
    o := &defaultOptimizer{
        learningRate: defaultLearningRate,
    }
    if len(overrides) == 1 {
        o.learningRate = overrides[0]
    }
    return o
}

func (do defaultOptimizer) update(tensor *tensor.Tensor, batchSize, counter int) {
    tensor.Reduce(mat.MulScalar(tensor.Gradient(), do.learningRate / float32(batchSize)))
}

type momentumOptimizer struct {
    learningRate float32
    momentum float32
    velocityMap map[string]*mat.Mat32f
}

func newMomentumOptimizer(overrides []float32) *momentumOptimizer {
    o := &momentumOptimizer{
        learningRate: defaultLearningRate,
        momentum: defaultMomentum,
        velocityMap: make(map[string]*mat.Mat32f),
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
        mo.velocityMap[tensor.Id()] = mat.NewMat32fZeros(tensor.Shape())
    }

    mo.velocityMap[tensor.Id()] = mat.Add(mat.MulScalar(mo.velocityMap[tensor.Id()], mo.momentum), mat.MulScalar(tensor.Gradient(), mo.learningRate / float32(batchSize)))
    tensor.Reduce(mo.velocityMap[tensor.Id()])
}

type adamOptimizer struct {
    learningRate float32
    beta1 float32
    beta2 float32
    epsStable float32
    meanMap map[string]*mat.Mat32f
    velocityMap map[string]*mat.Mat32f
}

func newAdamOptimizer(overrides []float32) *adamOptimizer {
    o := &adamOptimizer{
        learningRate: defaultLearningRate,
        beta1: defaultBeta1,
        beta2: defaultBeta2,
        epsStable: defaultEpsStable,
        meanMap: make(map[string]*mat.Mat32f),
        velocityMap: make(map[string]*mat.Mat32f),
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
        ao.meanMap[tensor.Id()] = mat.NewMat32fZeros(tensor.Shape())
        ao.velocityMap[tensor.Id()] = mat.NewMat32fZeros(tensor.Shape())
    }

    g := mat.DivScalar(tensor.Gradient(), float32(batchSize))

    ao.meanMap[tensor.Id()] = mat.Add(mat.MulScalar(ao.meanMap[tensor.Id()], ao.beta1), mat.MulScalar(g, 1 - ao.beta1))
    ao.velocityMap[tensor.Id()] = mat.Add(mat.MulScalar(ao.velocityMap[tensor.Id()], ao.beta2), mat.MulScalar(mat.Pow(g, 2), 1 - ao.beta2))

    biasCorr := mat.DivScalar(ao.meanMap[tensor.Id()], 1 - float32(math.Pow(float64(ao.beta1), float64(counter))))
    sqrtBiasCorr := mat.DivScalar(ao.velocityMap[tensor.Id()], 1 - float32(math.Pow(float64(ao.beta2), float64(counter))))

    tensor.Reduce(mat.Div(mat.MulScalar(biasCorr, ao.learningRate), mat.AddScalar(mat.Sqrt(sqrtBiasCorr), ao.epsStable)))
}
