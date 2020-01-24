package models

import (
	"fmt"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"log"
	"time"
)

func forward(modules []modules.Module, x *tensor.Tensor) *tensor.Tensor {
	for _, module := range modules {
		x = module.Forward(x)
	}

	return x
}

func fit(model *Model, x, target *tensor.Tensor) {
	sgd := NewSGD(model, createOptimizer(model.configuration.Optimizer))
	criterion := NewCriterion(model.configuration.Loss)
	numBatches := len(x.Data()) / model.configuration.BatchSize

	var aveTime int64 = 1
	t := time.Now().UnixNano()
	for i := 1; i != model.configuration.Epochs; i++ {
		lossMean := 0.0
		shuffleDataset(x.Data(), target.Data())
		for batchCounter := 0; batchCounter < numBatches; batchCounter++ {
			batchInputs := tensor.NewTensor(partitionData(x.Data(), batchCounter, model.configuration.BatchSize))
			batchTarget := tensor.NewTensor(partitionData(target.Data(), batchCounter, model.configuration.BatchSize))

			pred := forward(model.modules, batchInputs)
			loss := criterion.Forward(pred, batchTarget)
			lossMean += loss.Data()[0][0]
			loss.Backward()
			sgd.Step(model.configuration.BatchSize, i * model.configuration.BatchSize + batchCounter)
		}

		lossMean /= float64(model.configuration.BatchSize)
		if i % 10000 == 0 {
			fmt.Println("Epoch", i, "finished with error", lossMean)
			t2ms := (time.Now().UnixNano() - t) / int64(time.Millisecond)
			aveTime += t2ms
			t = time.Now().UnixNano()
		}
		if lossMean < model.configuration.MaxError {
			fmt.Println("Finished in", i, "loss:", lossMean)
			div := int64(i / 10000)
			if div == 0 {
				div = 1
			}
			fmt.Println(aveTime / div)
			break
		}
	}
}

func createOptimizer(optimizerType string) optimizer {
	switch optimizerType {
	case OptimizerDefault:
		return newDefaultOptimizer([]float64{})
	case OptimizerMomentum:
		return newMomentumOptimizer([]float64{})
	case OptimizerAdam:
		return newAdamOptimizer([]float64{})
	}

	log.Fatal("Unknown optimizer selected:", optimizerType)
	return nil
}
