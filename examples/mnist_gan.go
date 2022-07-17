package examples

import (
	"fmt"
	"ml-framework/datasets"
	"ml-framework/models"
	"ml-framework/modules"
	"ml-framework/tensor"
)

func RunMnistGan() {
	//train()
	t := tensor.Ones(2, 3, 3)
	fmt.Println(t.ToFloat32())
}

func generatorModel() *models.Model {
	return models.Build(
		modules.Linear(128),
		modules.LeakyRelu(),
		modules.Linear(784),
		modules.Tanh())
}

func discriminatorModel() *models.Model {
	return models.Build(
		modules.Linear(128),
		modules.LeakyRelu(),
		modules.Sigmoid())
}

func train() {
	batchSize := 1000
	epochs := 3
	dataset := datasets.From("mnist").SetBatchSize(batchSize)

	//generator := generatorModel()
	//discriminator := discriminatorModel()

	//*******

	//criterion := modules.NewCriterion(modules.LossMeanSquared)
	//optimizer := models.NewOptimizer(models.OptimizerAdam)

	graph := tensor.NewGraph()
	batchX := tensor.OfShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).Data().Shape().D[1]).SetName("batch x")
	//z := tensor.From(tensor.InitNormalized, batchSize, 100)
	//batchY := tensor.OfShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).Data().Shape().Y).SetName("batch y")
	//generatorPred := generator.Build(z).SetName("z")
	//real := tensor.Ones(batchSize, 1)
	//fake := tensor.Ones(batchSize, 0)

	for epoch := 1; epoch != epochs+1; epoch++ {
		for dataset.HasNextBatch() {
			batchDataX, _ := dataset.NextBatch()
			batchX.SetData(batchDataX.Data())

			//graph.Forward(generatorPred)
		}
	}

	graph.Close()
	//r.optimizer.Close()
}
