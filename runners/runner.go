package runners

import (
	"fmt"
	"github.com/jaypipes/ghw"
	"gorgonia.org/cu"
	"ml-framework/datasets"
	"ml-framework/models"
	"ml-framework/modules"
	"ml-framework/telemetry"
	"ml-framework/tensor"
)

type runner struct {
	model         *models.Model
	configuration ModelConfig
	criterion     modules.Criterion
	optimizer     models.Optimizer
	metric        *metric
	logger        *telemetry.Logger
}

func BuildFromModules(modules ...modules.Module) *runner {
	return BuildFromModel(models.Build(modules...))
}

func BuildFromModel(model *models.Model) *runner {
	runner := &runner{
		model:         model,
		configuration: ModelConfig{},
		logger:        telemetry.NewLogger(),
	}

	runner.metric = newMetric(&runner.configuration, runner.logger)
	runner.configuration.populateDefaults()
	runner.metric.start()

	gpu, err := ghw.GPU()
	if err != nil {
		runner.logger.Info(fmt.Sprintf("Error getting GPU info: %v", err))
	}

	runner.logger.Info(fmt.Sprintf("%v", gpu))

	for _, card := range gpu.GraphicsCards {
		runner.logger.Info(fmt.Sprintf("%v", card))
	}

	runner.logger.Info(fmt.Sprintf("CUDA version: %v", cu.Version()))

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

	r.logger.LogToFile = r.configuration.LogToFile
	r.logger.LogFolder = r.configuration.LogFolder
	r.logger.Initialize()
}

func (r *runner) Fit(dataset *datasets.Dataset) {
	r.Initialize()
	graph := tensor.NewGraph()
	batchX := tensor.OfShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).Data().Shape().Y).SetName("batch x")
	batchY := tensor.OfShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).Data().Shape().Y).SetName("batch y")
	pred := r.model.Build(batchX).SetName("prediction")
	loss := r.criterion.Forward(pred, batchY).SetName("loss")

	r.metric.events.trainingStarted <- true

	for epoch := 1; epoch != r.configuration.Epochs+1; epoch++ {
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
			r.metric.events.batchLoss <- batchLoss
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

	graph.Close()
	r.optimizer.Close()

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

	graph.Close()

	r.logger.Info(fmt.Sprintf("Error: %f Accuracy: %.2f\n", averageLoss(loss), accuracyOneHot(y, target)))
}
