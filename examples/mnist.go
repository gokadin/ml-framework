package examples

import (
	"ml-framework/datasets"
	"ml-framework/models"
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
		Epochs: 5,
		Loss:   modules.LossSoftmaxCrossEntropy,
	})

	runner.Fit(dataset)
	runner.Validate(dataset)

	//runner.GetModel().Save("mnist")
}

func RunSaved() {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	runner := runners.BuildFromModel(models.Restore("mnist"))

	runner.Validate(dataset)
}
