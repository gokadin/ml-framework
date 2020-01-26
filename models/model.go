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
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	model := &Model{
		configuration: ModelConfig{},
	}
	model.configuration.populateDefaults()
	model.initialize()

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

func (m *Model) Fit(dataset *datasets.Dataset) {
	x := tensor.NewTensor(dataset.Get(datasets.TrainingSetX).Data())
	target := tensor.NewTensor(dataset.Get(datasets.TrainingSetY).Data())

	fit(m, x, target)
}

func (m *Model) Run(dataset *datasets.Dataset) {
	x := tensor.NewTensor(dataset.Get(datasets.ValidationSetX).Data())
	target := tensor.NewTensor(dataset.Get(datasets.ValidationSetY).Data())

	y := forward(m.modules, x)
	loss := m.criterion.forward(y, target)

	fmt.Println(loss.Data()[0][0])
}

func (m *Model) initialize() {
	m.criterion = newCriterion(m.configuration.Loss)
	m.optimizer = newOptimizer(m.configuration.Optimizer)
}