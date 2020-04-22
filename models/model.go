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
	metric *metric
	isInitialized bool
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	model := &Model{
		configuration: ModelConfig{},
		trainableVariables: make([]*tensor.Tensor, 0),
		modules: make([]modules.Module, 0),
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
	graph := tensor.NewGraph()
	batchX := tensor.Variable(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).Data().Shape().Y).SetName("batch x")
	batchY := tensor.Variable(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).Data().Shape().Y).SetName("batch y")
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
			batchX.SetData(batchDataX.Data())
			batchY.SetData(batchDataY.Data())

			m.metric.events.forwardStarted <- true
			graph.Forward(loss)
			m.metric.events.forwardFinished <- true
			m.metric.events.backwardStarted <- true
			graph.Backward(loss, m.trainableVariables...)
			m.metric.events.backwardFinished <- true

			batchLoss := averageLoss(loss)
			epochLoss += batchLoss

			m.optimizer.Update(m.TrainableVariables()...)

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
	graph := tensor.NewGraph()
	m.Initialize(dataset.Shape().Y)
	x := tensor.Variable(dataset.Get(datasets.ValidationSetX).Data().Shape().X, dataset.Get(datasets.ValidationSetX).Data().Shape().Y).
		SetData(dataset.Get(datasets.ValidationSetX).Data().Data())
	target := tensor.Variable(dataset.Get(datasets.ValidationSetY).Data().Shape().X, dataset.Get(datasets.ValidationSetY).Data().Shape().Y).
		SetData(dataset.Get(datasets.ValidationSetY).Data().Data())

	y := m.Predict(x)
	loss := m.criterion.forward(y, target)

	graph.Forward(loss)

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

func (m *Model) Optimizer() optimizer {
	return m.optimizer
}

func (m *Model) Modules() []modules.Module {
	return m.modules
}

func (m *Model) Copy() *Model {
	duplicate := NewModel()

	duplicate.configuration.LearningRate = m.configuration.LearningRate
	duplicate.configuration.Epochs = m.configuration.Epochs
	duplicate.configuration.Optimizer = m.configuration.Optimizer
	duplicate.configuration.Loss = m.configuration.Loss
	duplicate.configuration.ValidOutputRange = m.configuration.ValidOutputRange
	duplicate.configuration.MaxError = m.configuration.MaxError

	for _, module := range m.modules {
		var duplicateModule modules.Module
		switch module.Type() {
		case "dense":
			duplicateModule = modules.Dense(module.GetParameters()[0].Shape().Y, module.GetActivation())
			break
		}
		duplicateModule.InitializeWith(
			tensor.Variable(module.GetParameters()[0].Shape().X, module.GetParameters()[0].Shape().Y).SetData(module.GetParameters()[0].ToFloat32()),
			tensor.Variable(module.GetParameters()[1].Shape().X, module.GetParameters()[1].Shape().Y).SetData(module.GetParameters()[1].ToFloat32()))
		duplicate.Add(duplicateModule)
	}

	duplicate.Initialize(m.modules[0].GetParameters()[0].Shape().Y)

	return duplicate
}

func (m *Model) SyncFrom(target *Model) {
	for i, module := range target.modules {
		for j, parameter := range module.GetParameters() {
			m.modules[i].GetParameters()[j].SetData(parameter.ToFloat32())
		}
	}
}
