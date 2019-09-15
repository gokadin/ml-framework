package runners

import "github.com/gokadin/ml-framework/core"

type SGD struct {
	network *core.Network
}

func NewSGD(network *core.Network) *SGD {
	return &SGD{
        network: network,
	}
}

func (sgd *SGD) Step(learningRate float64) {
	for _, layer := range sgd.network.GetLayers() {
		if layer.IsOutputLayer() {
			continue
		}
		for _, parameter := range layer.GetParameters() {
			parameter.Reduce(learningRate)
		}
	}
}
