package runners

import (
	"github.com/gokadin/ml-framework/models"
)

type SGD struct {
	network *models.Model
	optimizer optimizer
}

func NewSGD(network *models.Model, optimizer optimizer) *SGD {
	return &SGD{
        network,
        optimizer,
	}
}

func (sgd *SGD) Step(batchSize, counter int) {
	for i, layer := range sgd.network.GetLayers() {
		if layer.IsOutputLayer() {
			continue
		}
		for j, p := range layer.GetParameters() {
			sgd.optimizer.update(p, string(i) + string(j), batchSize, counter)
		}
	}
}
