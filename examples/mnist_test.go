package examples

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/runners"
	"testing"
)

func Test_mnist(t *testing.T) {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	runner := runners.BuildModelRunner(
		modules.Linear(128),
		modules.Relu(),
		modules.Linear(10))

	runner.Configure(runners.ModelConfig{
		Epochs: 5,
		Loss:   modules.LossSoftmaxCrossEntropy,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
