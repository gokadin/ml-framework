package models

type SGD struct {
	model   *Model
	optimizer optimizer
}

func NewSGD(network *Model, optimizer optimizer) *SGD {
	return &SGD{
        network,
        optimizer,
	}
}

func (sgd *SGD) Step(batchSize, counter int) {
	for i, module := range sgd.model.modules {
		for j, p := range module.GetParameters() {
			sgd.optimizer.update(p, string(i) + string(j), batchSize, counter)
		}
	}
}
