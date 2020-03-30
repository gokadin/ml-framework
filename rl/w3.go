package rl

import (
	"fmt"
	"github.com/faiface/pixel/pixelgl"
	"github.com/gokadin/ml-framework/graphics2D"
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"math/rand"
	"runtime"
	"time"
)

type W3 struct {
	gridWorld *graphics2D.GridWorld
	model *models.Model
	epsilon float32
	gamma float32
}

func NewW3() *W3 {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	return &W3{
		epsilon: 1.0,
		gamma: 0.9,
	}
}

func (w *W3) Run() {
	//pixelgl.Run(w.gridWorld.Run)
	w.model = w.buildModel()
	state := tensor.Variable(mat.WithShape(1, 64))
	state2 := tensor.Variable(mat.WithShape(1, 64))
	y := tensor.Variable(mat.WithShape(1, 4))
	qval := w.model.Predict(state)
	newQ := w.model.PredictNoGrad(state2)
	var lossSum float32
	p, _ := plot.New()
	p.X.Label.Text = "Epochs"
	p.Y.Label.Text = "Loss"
	line, _ := plotter.NewLine(plotter.XYs{})
	gameCounter := 0
	for i := 0; i < w.model.Configuration().Epochs; i++ {
		w.createGame()
		stateMat := w.gridWorld.GetState()
		w.addNoise(stateMat)
		state.SetData(stateMat)
		gameInProgress := true
		for gameInProgress {
			gameCounter++
			w.model.Graph().Forward(qval)

			// choose action
			var action int
			if rand.Float32() < w.epsilon {
				action = rand.Intn(4)
			} else {
				action = maxIndex(qval.Data().Data())
			}
			w.gridWorld.MakeMove(action)
			nextStateMat := w.gridWorld.GetState()
			w.addNoise(nextStateMat)
			state2.SetData(nextStateMat)
			reward := w.gridWorld.GetReward()

			w.model.Graph().Forward(newQ)
			maxQValue := maxValue(newQ.Data().Data())

			var yValue float32
			if reward == -1 {
				yValue = float32(reward) + w.gamma * maxQValue
			} else {
				yValue = float32(reward)
			}
			y.SetData(mat.NewMat32f(mat.WithShape(1, 4), qval.Data().Copy()))
			y.Data().Set(action, yValue)

			loss := w.model.Loss(qval, y)
			w.model.Graph().Forward(loss)
			w.model.Graph().Backward(loss, w.model.TrainableVariables()...)
			lossSum += averageLoss(loss)
			line.XYs = append(line.XYs, plotter.XY{Y: float64(loss.Data().At(action)), X: float64(i)})

			for _, parameter := range w.model.TrainableVariables() {
				w.model.Optimizer().Update(parameter, 1, 1)
			}

			state.SetData(mat.NewMat32f(mat.WithShape(1, 64), state2.Data().Copy()))

			if reward != -1 {
				gameInProgress = false
			}
		}

		if i != 0 && i % 100 == 0 {
			fmt.Println(fmt.Sprintf("loss %f", lossSum / float32(i + gameCounter)))
		}

		if w.epsilon > 0.1 {
			w.epsilon -= 1.0 / float32(w.model.Configuration().Epochs)
		}
	}

	p.Add(line)
	if err := p.Save(10*vg.Inch, 10*vg.Inch, "points.png"); err != nil {
		panic(err)
	}

}

func (w *W3) test() {
	pixelgl.Run(w.gridWorld.Run)
	w.createGame()
	state := tensor.Variable(mat.WithShape(1, 64))
	isGameRunning := true
	qval := w.model.Predict(state)
	for isGameRunning {
		state.SetData(w.gridWorld.GetState())
		w.model.Graph().Forward(qval)
		action := maxIndex(qval.Data().Data())
		w.gridWorld.MakeMove(action)
	}
}

func (w *W3) buildModel() *models.Model {
	model := models.Build(
		modules.Dense(150, modules.ActivationRelu),
		modules.Dense(100, modules.ActivationRelu),
		modules.Dense(4, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Epochs: 1000,
		Loss: models.LossMeanSquared,
	})

	return model
}

func (w *W3) createGame() {
	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
	w.gridWorld.PlaceAgent(0, 0)
	w.gridWorld.PlaceWall(1, 1)
	w.gridWorld.PlaceTarget(2, 3)
	w.gridWorld.PlaceDanger(0, 2)
}

func (w *W3) addNoise(mat *mat.Mat32f) {
	mat.Apply(func(f float32) float32 {
		return f + rand.Float32() / 10
	})
}

func maxIndex(values []float32) int {
	var max float32 = values[0]
	index := 0
	for i := 1; i < len(values); i++ {
		if values[i] > max {
			max = values[i]
			index = i
		}
	}
	return index
}

func maxValue(values []float32) float32 {
	var max float32 = values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > max {
			max = values[i]
		}
	}
	return max
}