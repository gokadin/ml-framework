package models

import (
	"math/rand"
	"ml-framework/modules"
	"ml-framework/tensor"
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

func (m *Model) TrainableVariables() []*tensor.Tensor {
	return m.trainableVariables
}

func (m *Model) Build(input *tensor.Tensor) *tensor.Tensor {
	pred := input
	for _, module := range m.modules {
		pred = module.Build(pred)
		m.trainableVariables = append(m.trainableVariables, module.GetParameters()...)
	}
	return pred
}

func (m *Model) BuildNoGrad(input *tensor.Tensor) *tensor.Tensor {
	pred := input
	for _, module := range m.modules {
		pred = module.Build(pred)
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
		duplicate.Add(module.Copy())
	}

	return duplicate
}

func (m *Model) SyncFrom(target *Model) {
	for i, module := range target.modules {
		for j, parameter := range module.GetParameters() {
			m.modules[i].GetParameters()[j].SetData(parameter.ToFloat32())
		}
	}
}
