package runners

import (
	"fmt"
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
)

type runner struct {
	model         *models.Model
	configuration ModelConfig
	criterion     modules.Criterion
	optimizer     models.Optimizer
	metric        *metric
}

func BuildModelRunner(modules ...modules.Module) *runner {
	runner := &runner{
		model: models.Build(modules...),
		configuration:      ModelConfig{},
	}

	runner.metric = newMetric(&runner.configuration)
	runner.configuration.populateDefaults()
	runner.metric.start()

	return runner
}

func (r *runner) Configure(configuration ModelConfig) {
	r.configuration = configuration
	r.configuration.populateDefaults()
}

func (r *runner) Configuration() *ModelConfig {
	return &r.configuration
}

func (r *runner) Loss(pred, batchY *tensor.Tensor) *tensor.Tensor {
	return r.criterion.Forward(pred, batchY)
}

func (r *runner) Optimizer() models.Optimizer {
	return r.optimizer
}

func (r *runner) Initialize() {
	r.criterion = modules.NewCriterion(r.configuration.Loss)
	r.optimizer = models.NewOptimizer(r.configuration.Optimizer)
}

func (r *runner) Fit(dataset *datasets.Dataset) {
	r.Initialize()
	graph := tensor.NewGraph()
	batchX := tensor.OfShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).Data().Shape().Y).SetName("batch x")
	batchY := tensor.OfShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).Data().Shape().Y).SetName("batch y")
	pred := r.model.Build(batchX).SetName("prediction")
	loss := r.criterion.Forward(pred, batchY).SetName("loss")

	r.metric.events.trainingStarted <- true

	for epoch := 1; epoch != r.configuration.Epochs + 1; epoch++ {
		r.metric.events.epochStarted <- epoch

		var epochLoss float32
		for dataset.HasNextBatch() {
			r.metric.events.batchStarted <- dataset.BatchCounter()

			batchDataX, batchDataY := dataset.NextBatch()
			batchX.SetData(batchDataX.Data())
			batchY.SetData(batchDataY.Data())

			r.metric.events.forwardStarted <- true
			graph.Forward(loss)
			r.metric.events.forwardFinished <- true
			r.metric.events.backwardStarted <- true
			graph.Backward(loss, r.model.TrainableVariables()...)
			r.metric.events.backwardFinished <- true

			batchLoss := averageLoss(loss)
			epochLoss += batchLoss

			r.metric.events.optimizerStarted <- true
			r.optimizer.Update(r.model.TrainableVariables()...)
			r.metric.events.optimizerFinished <- true

			r.metric.events.batchFinished <- true
		}

		epochLoss /= float32(dataset.NumBatches())

		r.metric.events.epochFinished <- epochLoss

		if epochLoss < r.configuration.MaxError {
			break
		}
	}

	r.metric.events.trainingFinished <- true
}

func (r *runner) Run(dataset *datasets.Dataset) {
	graph := tensor.NewGraph()
	r.Initialize()

	x := tensor.OfShape(dataset.Get(datasets.ValidationSetX).Data().Shape().X, dataset.Get(datasets.ValidationSetX).Data().Shape().Y).
		SetData(dataset.Get(datasets.ValidationSetX).Data().Data())
	target := tensor.OfShape(dataset.Get(datasets.ValidationSetY).Data().Shape().X, dataset.Get(datasets.ValidationSetY).Data().Shape().Y).
		SetData(dataset.Get(datasets.ValidationSetY).Data().Data())

	y := r.model.Build(x)
	loss := r.criterion.Forward(y, target)

	graph.Forward(loss)

	fmt.Printf("Error: %f Accuracy: %.2f\n", averageLoss(loss), accuracyOneHot(y, target))
}
