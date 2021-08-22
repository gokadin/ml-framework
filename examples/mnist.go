package examples

import (
	"ml-framework/datasets"
	"ml-framework/modules"
	"ml-framework/runners"
)

func RunMnist() {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	runner := runners.BuildFromModules(
		modules.Linear(128),
		modules.Relu(),
		modules.Linear(10))

	runner.Configure(runners.ModelConfig{
		Epochs: 3,
		Loss:   modules.LossSoftmaxCrossEntropy,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
