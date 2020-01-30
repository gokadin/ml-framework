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
}

func NewModel() *Model {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	model := &Model{
		configuration: ModelConfig{},
		graph: tensor.NewGraph(),
		trainableVariables: make([]*tensor.Tensor, 0),
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

func (m *Model) TrainableVariables() []*tensor.Tensor {
	return m.trainableVariables
}

func (m *Model) Fit(dataset *datasets.Dataset) {
	batchX := tensor.Variable(dataset.BatchSize(), dataset.Get(datasets.TrainingSetX).ShapeY()).SetName("batch x")
	batchY := tensor.Variable(dataset.BatchSize(), dataset.Get(datasets.TrainingSetY).ShapeY()).SetName("batch y")

	pred := m.buildModules(batchX).SetName("prediction")
	loss := m.criterion.forward(pred, batchY).SetName("loss")

	var aveTime int64 = 1
	t := time.Now().UnixNano()
	for epoch := 1; epoch != m.configuration.Epochs; epoch++ {
		lossMean := 0.0
		dataset.Shuffle()

		var ttAve int64
		ttAve = 0
		tt := time.Now().UnixNano()
		for {
			if !dataset.HasNextBatch() {
				dataset.ResetBatchCounter()
				break
			}

			batchDataX, batchDataY := dataset.NextBatch()
			batchX.SetData(batchDataX)
			batchY.SetData(batchDataY)

			m.graph.Forward(loss)
			m.graph.Backward(loss, m.trainableVariables...)

			batchLossMean := 0.0
			for _, yLoss := range loss.Data()[0] {
				batchLossMean += yLoss
			}
			lossMean += batchLossMean / float64(len(loss.Data()[0]))

			for _, parameter := range m.TrainableVariables() {
				m.optimizer.update(parameter, dataset.BatchSize(), epoch * dataset.BatchSize() + dataset.BatchCounter())
			}

			if dataset.BatchCounter() > 0 && dataset.BatchCounter() % 100 == 0 {
				tt2 := time.Now().UnixNano()
				diff := tt2 - tt
				diffms := diff / int64(time.Millisecond)
				ttAve += diffms
				fmt.Println("current:", diffms, "ave:", ttAve / int64(dataset.BatchCounter() / 100))
				tt = time.Now().UnixNano()
			}
		}

		lossMean /= float64(dataset.BatchSize())
		if epoch % 10000 == 0 {
			fmt.Println("Epoch", epoch, "finished with error", lossMean)
			t2ms := (time.Now().UnixNano() - t) / int64(time.Millisecond)
			aveTime += t2ms
			t = time.Now().UnixNano()
		}
		if lossMean < m.configuration.MaxError {
			fmt.Println("Finished in", epoch, "loss:", lossMean)
			div := int64(epoch / 10000)
			if div == 0 {
				div = 1
			}
			fmt.Println(aveTime / div)
			break
		}
	}
}

func (m *Model) Run(dataset *datasets.Dataset) {
	x := tensor.Constant(dataset.Get(datasets.ValidationSetX).Data())
	target := tensor.Constant(dataset.Get(datasets.ValidationSetY).Data())

	y := m.buildModules(x)
	loss := m.criterion.forward(y, target)

	m.graph.Forward(loss)

	fmt.Println(loss.Data()[0][0])
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