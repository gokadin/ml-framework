package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/runners"
	"testing"
)

func Test_mnist(t *testing.T) {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	//models.Restore("mnist")
	runner := runners.BuildModelRunner(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	runner.Configure(runners.ModelConfig{
		Epochs: 20,
		Loss:   modules.LossSoftmaxCrossEntropy,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
