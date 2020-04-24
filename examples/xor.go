package examples

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/runners"
)

func RunXor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	runner := runners.BuildModelRunner(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))

	runner.Configure(runners.ModelConfig{
		Optimizer: models.OptimizerAdam,
		Loss:      modules.LossMeanSquared,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
}
