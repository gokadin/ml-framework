package main

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
)

func main() {
	//perftest()
	mnist()
	//xor()
	//random()
}

func mnist() {
	dataset := datasets.From("mnist").SetBatchSize(60000)

	model := models.Build(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Epochs: 3,
		Loss: models.LossMeanSquared,
	})

	model.Fit(dataset)
	model.Run(dataset)
}

func xor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	model := models.Build(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Optimizer:          models.OptimizerAdam,
		Loss:               models.LossMeanSquared,
	})

	model.Fit(dataset)
	model.Run(dataset)
}

func random() {
	//x := [][]float64{{1, 0}, {1, 1}, {0, 1}, {0, 0}}
	//y := [][]float64{{1}, {0}, {1}, {0}}
	x := [][]float64{{1, 0}, {1, 1}, {0, 1}}
	y := [][]float64{{0.5}, {1}, {0.6}}

	dataset := datasets.NewDataset()
	dataset.AddData(datasets.TrainingSetX, x)
	dataset.AddData(datasets.TrainingSetY, y)
	dataset.SetBatchSize(3).DisableShuffle()

	model := models.Build(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Optimizer: models.OptimizerMomentum,
	})

	model.Fit(dataset)
}