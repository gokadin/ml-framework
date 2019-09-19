package runners

import (
	"github.com/gokadin/ml-framework/core"
	"github.com/gokadin/ml-framework/tensor"
)

type SGD struct {
	network *core.Network
	autograd *tensor.Autograd
}

func NewSGD(network *core.Network) *SGD {
	return &SGD{
        network: network,
        autograd: tensor.NewAutograd(),
	}
}

func (sgd *SGD) Step(loss *tensor.Tensor, learningRate float64) {
	for _, layer := range sgd.network.GetLayers() {
		if layer.IsOutputLayer() {
			continue
		}
		for _, parameter := range layer.GetParameters() {
			grad := sgd.autograd.Gradient(parameter, loss)
            parameter.Reduce(grad, learningRate)
		}
	}
}
