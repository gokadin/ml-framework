package models

import (
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"math/rand"
	"runtime"
	"time"
)

type Model struct {
	modules []modules.Module
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	return &Model{}
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

func (m *Model) Fit(dataset *datasets.Dataset) {

}

func (m *Model) Evaluate(dataset *datasets.Dataset) {
	y := tensor.NewTensor(dataset.Get(datasets.ValidationSetX).Data())
	yHat := tensor.NewTensor(dataset.Get(datasets.ValidationSetY).Data())
	_ = yHat
	for _, module := range m.modules {
		y = module.Forward(y)
	}

	x := y
	_ = x
}
