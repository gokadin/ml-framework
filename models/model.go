package models

import (
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"math/rand"
	"runtime"
	"time"
)

type Model struct {
	modules            []modules.Module
	trainableVariables []*tensor.Tensor
	isInitialized      bool
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	model := &Model{
		trainableVariables: make([]*tensor.Tensor, 0),
		modules:            make([]modules.Module, 0),
	}

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

	m.isInitialized = true
}

func (m *Model) TrainableVariables() []*tensor.Tensor {
	return m.trainableVariables
}

func (m *Model) Predict(input *tensor.Tensor) *tensor.Tensor {
	m.trainableVariables = make([]*tensor.Tensor, 0)
	pred := input
	for _, module := range m.modules {
		pred = module.Forward(pred)
		m.trainableVariables = append(m.trainableVariables, module.GetParameters()...)
	}
	return pred
}

func (m *Model) PredictNoGrad(input *tensor.Tensor) *tensor.Tensor {
	pred := input
	for _, module := range m.modules {
		pred = module.Forward(pred)
	}
	return pred
}

func (m *Model) Save(name string) {
	saveModel(m, name)
}

func (m *Model) Modules() []modules.Module {
	return m.modules
}

func (m *Model) Copy() *Model {
	duplicate := NewModel()

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
