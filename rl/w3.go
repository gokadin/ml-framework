package rl

import (
	"fmt"
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
	epsilon float64
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
	w.model = w.buildModel()
	state := tensor.Variable(1, 64)
	state2 := tensor.Variable(1, 64)
	y := tensor.Variable(1, 4)
	qval := w.model.Predict(state)
	newQ := w.model.PredictNoGrad(state2)
	loss := w.model.Loss(qval, y)
	var lossSum float32
	p, _ := plot.New()
	p.X.Label.Text = "Epochs"
	p.Y.Label.Text = "Loss"
	line, _ := plotter.NewLine(plotter.XYs{})
	gameCounter := 0
	totalCounter := 0
	successCounter := 0
	gameAveMoves := 0
	for i := 0; i < w.model.Configuration().Epochs; i++ {
		w.createGame()
		stateMat := w.gridWorld.GetState()
		w.addNoise(stateMat)
		state.SetData(stateMat.Data())
		gameInProgress := true
		for gameInProgress {
			gameCounter++
			w.model.Forward(qval)

			// choose action
			var action int
			r := rand.Float64()
			if r < w.epsilon {
				action = rand.Intn(4)
			} else {
				action = maxIndex(qval.ToFloat32())
			}

			w.gridWorld.MakeMove(action)
			reward := w.gridWorld.GetReward()
			nextStateMat := w.gridWorld.GetState()
			w.addNoise(nextStateMat)
			state2.SetData(nextStateMat.Data())
			w.model.Forward(newQ)
			maxQValue := maxValue(newQ.ToFloat32())

			var yValue float32
			if reward == -1 {
				yValue = float32(reward) + w.gamma * maxQValue
			} else {
				yValue = float32(reward)
			}
			y.SetData(qval.ToFloat32())
			y.Set(action, yValue)

			w.model.Forward(loss)
			w.model.Backward(loss, w.model.TrainableVariables()...)

			lossSum += loss.ToFloat32()[action]
			line.XYs = append(line.XYs, plotter.XY{Y: float64(loss.ToFloat32()[action]), X: float64(i)})

			for _, parameter := range w.model.TrainableVariables() {
				w.model.Optimizer().Update(parameter, 1, i + 2)
			}

			state.SetData(nextStateMat.Data())

			if reward != -1 {
				gameInProgress = false
				if reward == 10 {
					successCounter++
				}
			}
		}

		gameAveMoves += gameCounter
		totalCounter += gameCounter
		if i != 0 && i % 100 == 0 {
			fmt.Println(fmt.Sprintf("epoch %d loss %f moves %d success %d%%", i, lossSum / float32(i + totalCounter), gameAveMoves / i, successCounter * 100 / i))
		}
		gameCounter = 0

		if w.epsilon > 0.1 {
			w.epsilon -= 1.0 / float64(w.model.Configuration().Epochs)
		}
	}

	p.Add(line)
	if err := p.Save(10*vg.Inch, 10*vg.Inch, "points.png"); err != nil {
		panic(err)
	}

	w.model.Save("experimental-rl")
}

func (w *W3) RunSaved() {
	w.model = models.Restore("experimental-rl")

	w.test()
}

func (w *W3) test() {
	w.createGame()
	//pixelgl.Run(w.gridWorld.Run)
	w.gridWorld.Print()
	state := tensor.Variable(1, 64)
	qval := w.model.Predict(state)
	counter := 0
	isGameRunning := true
	for isGameRunning {
		stateMat := w.gridWorld.GetState()
		w.addNoise(stateMat)
		state.SetData(stateMat.Data())

		w.model.Forward(qval)
		action := maxIndex(qval.ToFloat32())
		w.gridWorld.MakeMove(action)
		fmt.Println(fmt.Sprintf("taking action %d", action))

		//nextStateMat := w.gridWorld.GetState()
		//w.addNoise(nextStateMat)
		reward := w.gridWorld.GetReward()

		if reward != -1 {
			isGameRunning = false
			if reward > 0 {
				fmt.Println("game won")
			} else {
				fmt.Println("game lost")
			}
		}

		counter++
		if counter > 15 {
			fmt.Println("game lost... too many moves")
			isGameRunning = false
		}

		w.gridWorld.Print()
		time.Sleep(500 * time.Millisecond)
	}
}

func (w *W3) buildModel() *models.Model {
	model := models.Build(
		modules.Dense(164, modules.ActivationRelu),
		modules.Dense(150, modules.ActivationRelu),
		modules.Dense(4, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Epochs: 5000,
		Loss: models.LossMeanSquared,
		LearningRate: 0.0001,
	})

	model.Initialize(64)

	return model
}

func (w *W3) createGame() {
	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
	w.gridWorld.PlaceAgent(0, 3)
	w.gridWorld.PlaceWall(1, 2)
	w.gridWorld.PlaceTarget(2, 0)
	w.gridWorld.PlaceDanger(0, 1)
}

func (w *W3) addNoise(mat *mat.Mat32f) {
	mat.Apply(func(f float32) float32 {
		return f + rand.Float32() / 10
	})
}

func maxIndex(values []float32) int {
	max := values[0]
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
	max := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > max {
			max = values[i]
		}
	}
	return max
}