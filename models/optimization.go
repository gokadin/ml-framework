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
    defaultLearningRate float32 = 0.001
    defaultMomentum float32 = 0.9

    /* Adam optimizer */
    defaultBeta1 float32 = 0.9
    defaultBeta2 float32 = 0.999
    defaultEpsStable float32 = 1e-08
)

type Optimizer interface {
    Update(parameters ...*tensor.Tensor)
}

func NewOptimizer(optimizerType string) Optimizer {
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

func (do defaultOptimizer) Update(parameters ...*tensor.Tensor) {
    for _, parameter := range parameters {
        parameter.Reduce(mat.MulScalar(parameter.GradientToMat32(), do.learningRate).Data())
    }
}

type momentumOptimizer struct {
    learningRate float32
    momentum float32
    velocityMap map[int]*mat.Mat32f
}

func newMomentumOptimizer(overrides []float32) *momentumOptimizer {
    o := &momentumOptimizer{
        learningRate: defaultLearningRate,
        momentum: defaultMomentum,
        velocityMap: make(map[int]*mat.Mat32f),
    }
    if len(overrides) >= 1 {
        o.learningRate = overrides[0]
    }
    if len(overrides) == 2 {
        o.momentum = overrides[1]
    }
    return o
}

func (mo momentumOptimizer) Update(parameters ...*tensor.Tensor) {
    for _, parameter := range parameters {
        if _, ok := mo.velocityMap[parameter.Id()]; !ok {
            mo.velocityMap[parameter.Id()] = mat.NewMat32fZeros(mat.WithShape(parameter.Shape().X, parameter.Shape().Y))
        }

        mo.velocityMap[parameter.Id()] = mat.Add(mat.MulScalar(mo.velocityMap[parameter.Id()], mo.momentum), mat.MulScalar(parameter.GradientToMat32(), mo.learningRate))
        parameter.Reduce(mo.velocityMap[parameter.Id()].Data())
	}
}

type adamOptimizer struct {
    learningRate float32
    beta1 float32
    beta2 float32
    epsStable float32
    workerMap map[int]chan bool
    out chan bool
}

func newAdamOptimizer(overrides []float32) *adamOptimizer {
    o := &adamOptimizer{
        learningRate: defaultLearningRate,
        beta1: defaultBeta1,
        beta2: defaultBeta2,
        epsStable: defaultEpsStable,
        workerMap: make(map[int]chan bool),
        out: make(chan bool),
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

func (ao *adamOptimizer) Update(parameters ...*tensor.Tensor) {
    for _, parameter := range parameters {
        in, ok := ao.workerMap[parameter.Id()]
        if !ok {
            in = make(chan bool)
            ao.workerMap[parameter.Id()] = in
            go adamUpdate(parameter, in, ao.out, ao.beta1, ao.beta2, ao.epsStable, ao.learningRate)
        }

        in <- true
    }

    doneCounter := 0
    for range ao.out {
        doneCounter++
        if doneCounter == len(parameters) {
            break
        }
    }
}

func adamUpdate(parameter *tensor.Tensor, in, out chan bool, beta1, beta2, epsStable, learningRate float32) {
    mean := mat.NewMat32fZeros(mat.WithShape(parameter.Shape().X, parameter.Shape().Y))
    velocity := mat.NewMat32fZeros(mat.WithShape(parameter.Shape().X, parameter.Shape().Y))
    var count float64 = 0

    for range in {
    	count++
        g := parameter.GradientToMat32()

        mean = mat.Add(mat.MulScalar(mean, beta1), mat.MulScalar(g, 1 - beta1))
        velocity = mat.Add(mat.MulScalar(velocity, beta2), mat.MulScalar(mat.Pow(g, 2), 1 - beta2))

        biasCorr := mat.DivScalar(mean, 1 - float32(math.Pow(float64(beta1), count)))
        sqrtBiasCorr := mat.DivScalar(velocity, 1 - float32(math.Pow(float64(beta2), count)))

        parameter.Reduce(mat.Div(mat.MulScalar(biasCorr, learningRate), mat.AddScalar(mat.Sqrt(sqrtBiasCorr), epsStable)).Data())

        out <- true
    }
}
