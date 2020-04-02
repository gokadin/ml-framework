package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/rl"
)

func main() {
	//mnist()
	//xor()
	rltest()
}

func rltest() {
	ws := rl.NewW4()
	ws.Run()
	//ws.RunSaved()
}

func mnist() {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	//models.Restore("mnist")
	model := models.Build(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	model.Configure(models.ModelConfig{
		Epochs: 3,
		Loss: models.LossSoftmaxCrossEntropy,
	})

	model.Fit(dataset)
	model.Run(dataset)
	//model.Save("mnist")
}

func xor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	//models.Restore("xor")
	model := models.Build(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Optimizer:          models.OptimizerAdam,
		Loss:               models.LossMeanSquared,
	})

	model.Fit(dataset)
	model.Run(dataset)
	//model.Save("xor")
}