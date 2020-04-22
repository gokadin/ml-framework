package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"testing"
)

func Test_mnist(t *testing.T) {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	//models.Restore("mnist")
	model := models.Build(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	model.Configure(models.ModelConfig{
		Epochs: 20,
		Loss: models.LossSoftmaxCrossEntropy,
	})

	model.Fit(dataset)
	model.Run(dataset)
}
