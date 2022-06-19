package models

import (
	"encoding/json"
	"math/rand"
	"ml-framework/modules"
	"ml-framework/tensor"
	"runtime"
	"time"
)

type Model struct {
	Modules            []modules.Module `json:"modules"`
	trainableVariables []*tensor.Tensor
	isInitialized      bool
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	model := &Model{
		trainableVariables: make([]*tensor.Tensor, 0),
		Modules:            make([]modules.Module, 0),
	}

	return model
}

func Build(modules ...modules.Module) *Model {
	model := NewModel()
	model.Modules = modules
	return model
}

func (m *Model) Add(module modules.Module) *Model {
	m.Modules = append(m.Modules, module)
	return m
}

func (m *Model) TrainableVariables() []*tensor.Tensor {
	return m.trainableVariables
}

func (m *Model) Build(input *tensor.Tensor) *tensor.Tensor {
	pred := input
	for _, module := range m.Modules {
		pred = module.Build(pred)
		m.trainableVariables = append(m.trainableVariables, module.GetParameters()...)
	}
	return pred
}

func (m *Model) BuildNoGrad(input *tensor.Tensor) *tensor.Tensor {
	pred := input
	for _, module := range m.Modules {
		pred = module.Build(pred)
	}
	return pred
}

func (m *Model) Save(name string) {
	saveModel(m, name)
}

func (m *Model) Copy() *Model {
	duplicate := NewModel()

	for _, module := range m.Modules {
		duplicate.Add(module.Copy())
	}

	return duplicate
}

func (m *Model) SyncFrom(target *Model) {
	for i, module := range target.Modules {
		for j, parameter := range module.GetParameters() {
			m.Modules[i].GetParameters()[j].SetData(parameter.ToFloat32())
		}
	}
}

func (m *Model) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.Modules)
}

func (m *Model) UnmarshalJSON(data []byte) error {
	layers := []interface{}{}
	json.Unmarshal(data, &layers)

	for _, layer := range layers {
		switch layer.(map[string]interface{})["type"].(string) {
		case "linear":
			//l := modules.L
			//a := l
			//_ = a
		}
	}
	return nil
}
