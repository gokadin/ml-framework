package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
)

func main() {
	mnist()
	//xor()
	//persistenseTest()
}

func persistenseTest() {
	model := models.Build(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	model.Configure(models.ModelConfig{
		Epochs: 3,
		Loss: models.LossSoftmaxCrossEntropy,
	})
	model.Initialize(784)
	model.Save("mnist")

	// ...

	m := models.Restore("mnist")

	x := 3

	_ = x
	_ = m
}

func mnist() {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	//model := models.Restore("mnist")
	model := models.Build(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	model.Configure(models.ModelConfig{
		Epochs: 1,
		Loss: models.LossSoftmaxCrossEntropy,
	})

	model.Fit(dataset)
	model.Save("mnist")
	model.Run(dataset)
}

func xor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	model := models.Restore("xor")
	//model := models.Build(
	//	modules.Dense(2, modules.ActivationSigmoid),
	//	modules.Dense(1, modules.ActivationIdentity))
	//
	//model.Fit(dataset)
	model.Run(dataset)
	//model.Save("xor")
}
