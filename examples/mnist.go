package examples

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/runners"
)

func RunMnist() {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	runner := runners.BuildModelRunner(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	runner.Configure(runners.ModelConfig{
		Epochs: 50,
		Loss:   modules.LossSoftmaxCrossEntropy,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
