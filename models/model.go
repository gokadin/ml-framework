package models

import (
	"fmt"
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"math/rand"
	"runtime"
	"time"
)

type Model struct {
	modules []modules.Module
	configuration ModelConfig
	criterion criterion
	optimizer optimizer
	trainableVariables []*tensor.Tensor
	graph *tensor.Graph
	metric *metric
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	model := &Model{
		configuration: ModelConfig{},
		graph: tensor.NewGraph(),
		trainableVariables: make([]*tensor.Tensor, 0),
	}
	model.metric = newMetric(&model.configuration)
	model.configuration.populateDefaults()
	model.initialize()
	model.metric.start()

	return model
}

func Build(modules ...modules.Module) *Model {
	model := NewModel()
	model.modules = modules
	return model
}

func (m *Model) Add(module modules.Module) *Model {
	m.modules = append(m.modules, module)
	return m
}

func (m *Model) Configure(configuration ModelConfig) {
	m.configuration = configuration
	m.configuration.populateDefaults()
	m.initialize()
}

func (m *Model) TrainableVariables() []*tensor.Tensor {
	return m.trainableVariables
}

func (m *Model) Fit(dataset *datasets.Dataset) {
	batchX := tensor.Variable(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).ShapeY()).SetName("batch x")
	batchY := tensor.Variable(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).ShapeY()).SetName("batch y")
	pred := m.buildModules(batchX).SetName("prediction")
	loss := m.criterion.forward(pred, batchY).SetName("loss")

	m.metric.events.trainingStarted <- true

	for epoch := 1; epoch != m.configuration.Epochs + 1; epoch++ {
		m.metric.events.epochStarted <- epoch

		epochLoss := 0.0
		for {
			if !dataset.HasNextBatch() {
				dataset.ResetBatchCounter()
				break
			}

			m.metric.events.batchStarted <- dataset.BatchCounter()

			batchDataX, batchDataY := dataset.NextBatch()
			batchX.SetData(batchDataX)
			batchY.SetData(batchDataY)

			m.metric.events.forwardStarted <- true
			m.graph.Forward(loss)
			m.metric.events.forwardFinished <- true
			m.metric.events.backwardStarted <- true
			m.graph.Backward(loss, m.trainableVariables...)
			m.metric.events.backwardFinished <- true

			batchLoss := averageLoss(loss)
			epochLoss += batchLoss

			for _, parameter := range m.TrainableVariables() {
				m.optimizer.update(parameter, dataset.BatchSize(), epoch * dataset.BatchSize() + dataset.BatchCounter())
			}

			m.metric.events.batchFinished <- true
		}

		epochLoss /= float64(dataset.NumBatches())

		m.metric.events.epochFinished <- epochLoss

		if epochLoss < m.configuration.MaxError {
			break
		}
	}

	m.metric.events.trainingFinished <- true
}

func (m *Model) Run(dataset *datasets.Dataset) {
	x := tensor.Constant(dataset.Get(datasets.ValidationSetX).Data())
	target := tensor.Constant(dataset.Get(datasets.ValidationSetY).Data())

	y := m.buildModules(x)
	loss := m.criterion.forward(y, target)

	m.graph.Forward(loss)

	fmt.Printf("Error: %f Accuracy: %.2f", averageLoss(loss), accuracy(y, target, m.configuration.ValidOutputRange))
}

func (m *Model) initialize() {
	m.criterion = newCriterion(m.configuration.Loss)
	m.optimizer = newOptimizer(m.configuration.Optimizer)
}

func (m *Model) buildModules(x *tensor.Tensor) *tensor.Tensor {
	pred := x
	for _, module := range m.modules {
		pred = module.Forward(pred)
		m.trainableVariables = append(m.trainableVariables, module.GetParameters()...)
	}
	return pred
}