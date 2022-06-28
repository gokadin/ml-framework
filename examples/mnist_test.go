package examples

import (
	"ml-framework/datasets"
	"ml-framework/modules"
	"ml-framework/runners"
	"testing"
)

func Test_mnist(t *testing.T) {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	runner := runners.BuildFromModules(
		modules.Linear(128),
		modules.Relu(),
		modules.Linear(10))

	runner.Configure(runners.ModelConfig{
		Epochs:    1,
		Loss:      modules.LossSoftmaxCrossEntropy,
		LogToFile: true,
	})

	runner.Fit(dataset)
	runner.Validate(dataset)
}
