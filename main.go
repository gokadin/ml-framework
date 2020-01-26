package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
)

func main() {
	mnist()
}

func mnist() {
	dataset := datasets.From("mnist")

	model := models.Build(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Epochs: 3,
		Loss: models.LossMeanSquared,
	})

	model.Fit(dataset)
	model.Run(dataset)
}

func xor() {
	dataset := datasets.From("xor")

	model := models.Build(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))

	model.Fit(dataset)
	model.Run(dataset)
}