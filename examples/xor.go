package examples

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/runners"
)

func RunXor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	runner := runners.BuildModelRunner(
		modules.Linear(2),
		modules.Sigmoid(),
		modules.Linear(1))

	runner.Configure(runners.ModelConfig{
		Loss:      modules.LossMeanSquared,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
