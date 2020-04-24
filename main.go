package main

import (
	"github.com/gokadin/ml-framework/api"
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/runners"
)


func main() {
	mnist()
	//xor()
	//rltest()

	//startServer()
}

func startServer() {
	server := api.NewMLServer()
	server.Start()
}

func rltest() {
	//ws := rl.NewReinforce()
	//ws.Run()
	//ws.TestSingle()
	//ws.TestPercentage()
}

func mnist() {
	dataset := datasets.From("mnist").SetBatchSize(1000)

	//models.Restore("mnist")
	runner := runners.BuildModelRunner(
		modules.Dense(128, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationSoftmax))

	runner.Configure(runners.ModelConfig{
		Epochs: 50,
		Loss:   modules.LossSoftmaxCrossEntropy,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
	//model.Save("mnist")
}

func xor() {
	dataset := datasets.From("xor").SetBatchSize(4)

	//models.Restore("xor")
	runner := runners.BuildModelRunner(
		modules.Dense(2, modules.ActivationSigmoid),
		modules.Dense(1, modules.ActivationIdentity))

	runner.Configure(runners.ModelConfig{
		Optimizer: models.OptimizerAdam,
		Loss:      modules.LossMeanSquared,
	})

	runner.Fit(dataset)
	runner.Run(dataset)
	//model.Save("xor")
}