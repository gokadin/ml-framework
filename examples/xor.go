package examples

import (
	"ml-framework/datasets"
	"ml-framework/modules"
	"ml-framework/runners"
)

func RunXor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	runner := runners.BuildFromModules(
		modules.Linear(2),
		modules.Sigmoid(),
		modules.Linear(1))

	runner.Configure(runners.ModelConfig{
		Loss: modules.LossMeanSquared,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
