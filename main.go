package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
)

func main() {
	//dataset := datasets.From("mnist")
	dataset := datasets.From("xor")

	model := models.Build(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))
		//modules.Dense(128, modules.ActivationRelu),
		//modules.Dense(10, modules.ActivationSoftmax))

	//model.Configure(models.ModelConfig{
	//	Epochs: 3,
	//})

	model.Fit(dataset)
	model.Evaluate(dataset)
}
