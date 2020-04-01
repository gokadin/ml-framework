package models

import (
	"fmt"
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/mat"
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
	isInitialized bool
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

func (m *Model) Initialize(inputSize int) {
	if m.isInitialized {
		return
	}

	for _, module := range m.modules {
		module.Initialize(inputSize)
		inputSize = module.GetParameters()[0].Shape().Y
	}

	m.criterion = newCriterion(m.configuration.Loss)
	m.optimizer = newOptimizer(m.configuration.Optimizer)

	m.isInitialized = true
}

func (m *Model) Configure(configuration ModelConfig) {
	m.configuration = configuration
	m.configuration.populateDefaults()
}

func (m *Model) Configuration() *ModelConfig {
	return &m.configuration
}

func (m *Model) TrainableVariables() []*tensor.Tensor {
	return m.trainableVariables
}

func (m *Model) Fit(dataset *datasets.Dataset) {
	m.Initialize(dataset.Shape().Y)
	batchX := tensor.Variable(mat.WithShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).Data().Shape().Y)).SetName("batch x")
	batchY := tensor.Variable(mat.WithShape(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).Data().Shape().Y)).SetName("batch y")
	pred := m.Predict(batchX).SetName("prediction")
	loss := m.criterion.forward(pred, batchY).SetName("loss")

	m.metric.events.trainingStarted <- true

	for epoch := 1; epoch != m.configuration.Epochs + 1; epoch++ {
		m.metric.events.epochStarted <- epoch

		var epochLoss float32
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
				m.optimizer.Update(parameter, dataset.BatchSize(), epoch * dataset.BatchSize() + dataset.BatchCounter())
			}

			m.metric.events.batchFinished <- true
		}

		epochLoss /= float32(dataset.NumBatches())

		m.metric.events.epochFinished <- epochLoss

		if epochLoss < m.configuration.MaxError {
			break
		}
	}

	m.metric.events.trainingFinished <- true
}

func (m *Model) Run(dataset *datasets.Dataset) {
	m.Initialize(dataset.Shape().Y)
	x := tensor.Constant(dataset.Get(datasets.ValidationSetX).Data())
	target := tensor.Constant(dataset.Get(datasets.ValidationSetY).Data())

	y := m.Predict(x)
	loss := m.criterion.forward(y, target)

	m.graph.Forward(loss)

	fmt.Printf("Error: %f Accuracy: %.2f\n", averageLoss(loss), accuracyOneHot(y, target))
}

func (m *Model) Predict(x *tensor.Tensor) *tensor.Tensor {
	m.trainableVariables = make([]*tensor.Tensor, 0)
	pred := x
	for _, module := range m.modules {
		pred = module.Forward(pred)
		m.trainableVariables = append(m.trainableVariables, module.GetParameters()...)
	}
	return pred
}

func (m *Model) Save(name string) {
	saveModel(m, name)
}

func (m *Model) PredictNoGrad(x *tensor.Tensor) *tensor.Tensor {
	pred := x
	for _, module := range m.modules {
		pred = module.Forward(pred)
	}
	return pred
}

func (m *Model) Loss(pred, batchY *tensor.Tensor) *tensor.Tensor {
	return m.criterion.forward(pred, batchY)
}

func (m *Model) Forward(tensor *tensor.Tensor) {
	m.graph.Forward(tensor)
}

func (m *Model) Backward(of *tensor.Tensor, derivatives ...*tensor.Tensor) {
	m.graph.Backward(of, derivatives...)
}

func (m *Model) Optimizer() optimizer {
	return m.optimizer
}
