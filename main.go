package main

import (
	"github.com/gokadin/ml-framework/core"
	"github.com/gokadin/ml-framework/runners"
	"github.com/gokadin/ml-framework/tensor"
)

func main() {
	net := core.NewNetwork().
		AddInputLayer(2).
		AddHiddenLayer(2, tensor.ActivationFunctionIdentity).
		AddOutputLayer(1, tensor.ActivationFunctionIdentity)

	data := tensor.NewTensor([][]float64{{1, 1}})
	//data := tensor.NewTensor([][]float64{{1, 0}, {1, 1}, {0, 1}, {0, 0}})
	target := tensor.NewTensor([][]float64{{0.5}})
	//target := tensor.NewTensor([][]float64{{1}, {0}, {1}, {0}})

	runner := runners.NewNetworkRunner()
	runner.Train(net, data, target)
}
