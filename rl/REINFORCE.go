package rl

import (
	"fmt"
	"github.com/encryptio/alias"
	"github.com/gokadin/ml-framework/graphics2D"
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	gym "github.com/openai/gym-http-api/binding-go"
	"math/rand"
	"runtime"
	"time"
)

/*
	Implements the REINFORCE algorithm
 */
type Reinforce struct {
	gridWorld *graphics2D.GridWorld
	model *models.Model
	targetModel *models.Model
	metric *metric
	epsilon float64
	gamma float32
	targetNetworkSyncFrequency int
	targetNetworkSyncCounter int
	maxMoves int
	batchSize int
	memSize int
	stateSize int
	numActions int
}

func NewReinforce() *Reinforce {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	return &Reinforce{
		epsilon: 1.0,
		gamma: 0.9,
		maxMoves: 50,
		memSize: 1000,
		batchSize: 200,
		targetNetworkSyncFrequency: 500,
		stateSize: 64,
		numActions: 4,
		metric: newMetric(),
	}
}

const BaseURL = "http://localhost:5000"

func (w *Reinforce) Run() {
	w.metric.start()
	w.model = w.buildModel()

	client, err := gym.NewClient(BaseURL)
	if err != nil {
		panic(err)
	}

	insts, err := client.ListAll()
	if err != nil {
		panic(err)
	}
	fmt.Println("Started with instances:", insts)

	id, err := client.Create("CartPole-v0")
	if err != nil {
		panic(err)
	}
	defer client.Close(id)

	actSpace, err := client.ActionSpace(id)
	must(err)
	fmt.Printf("Action space: %+v\n", actSpace)

	must(client.StartMonitor(id, "/tmp/cartpole-monitor", false, false, false))

	stateMat := make([]float32, 4)
	state := tensor.Variable(mat.WithShape(1, 4))
	pred := w.model.Predict(state)
	loss := tensor.Neg()

	fmt.Println()
	fmt.Println("Starting new episode...")
	obs, err := client.Reset(id)
	must(err)
	fmt.Println("First observation:", obs)
	for {
		act, err := client.SampleAction(id)
		must(err)

		// Take the action, getting a new observation, a reward,
		// and a flag indicating if the episode is done.
		newObs, rew, done, _, err := client.Step(id, act, false)
		newState := w.obsToState(newObs.([]float64))
		must(err)
		obs = newObs
		fmt.Println("reward:", rew, " -- observation:", obs)
		if done {
			break
		}
	}

	must(client.CloseMonitor(id))
}

func (w *Reinforce) selectAction(probabilities []float64) int {
	x, _ := alias.New(probabilities)
	index := x.Gen(rand.New(rand.NewSource(time.Now().UTC().UnixNano())))
	return int(index)
}

func (w *Reinforce) obsToState(obs []float64) []float32 {
	state := make([]float32, len(obs))
	for i := 0; i < len(obs); i++ {
		state[i] = float32(obs[i])
	}
	return state
}

func (w *Reinforce) calculateYValue(maxQValue float32, reward int) float32 {
	if reward == -1 {
		return float32(reward) + w.gamma * maxQValue
	}

	return float32(reward)
}

func (w *Reinforce) TestPercentage() {
	w.model = models.Restore("reinforce")

	runs := 1000
	numWins := 0
	for i := 0; i < runs; i++ {
		if w.test(false) {
			numWins++
		}
	}

	fmt.Println(fmt.Sprintf("gridworld performance %2.f%%", float32(numWins) * 100 / float32(runs)))
}

func (w *Reinforce) TestSingle() {
	w.model = models.Restore("reinforce")
	w.test(true)
}

func (w *Reinforce) test(visualize bool) bool {
	w.createRandomGame()
	if visualize {
		w.gridWorld.Print()
	}
	state := tensor.Variable(mat.WithShape(1, 64))
	qval := w.model.Predict(state)
	counter := 0
	isGameRunning := true
	for isGameRunning {
		stateMat := w.gridWorld.GetState()
		w.addNoise(stateMat)
		state.SetData(stateMat)

		w.model.Forward(qval)
		action := maxIndex(qval.Data().Data())
		w.gridWorld.MakeMove(action)
		reward := w.gridWorld.GetReward()
		if visualize {
			fmt.Println(fmt.Sprintf("action %d reward %d", action, reward))
		}

		if reward != -1 {
			isGameRunning = false
			if reward > 0 {
				if visualize {
					fmt.Println("game won")
				}
				return true
			} else {
				if visualize {
					fmt.Println("game lost")
				}
			}
		}

		counter++
		if counter > 15 {
			if visualize {
				fmt.Println("game lost... too many moves")
			}
			isGameRunning = false
		}

		if visualize {
			w.gridWorld.Print()
			time.Sleep(500 * time.Millisecond)
		}
	}

	return false
}

func (w *Reinforce) buildModel() *models.Model {
	model := models.Build(
		modules.Dense(150, modules.ActivationLeakyRelu),
		modules.Dense(4, modules.ActivationSoftmax))

	model.Configure(models.ModelConfig{
		Epochs: 1000,
		Loss: models.LossMeanSquared,
		LearningRate: 0.0009,
	})

	model.Initialize(64)

	return model
}

func (w *Reinforce) createGame() {
	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
	w.gridWorld.PlaceAgent(0, 3)
	w.gridWorld.PlaceWall(1, 2)
	w.gridWorld.PlaceTarget(2, 0)
	w.gridWorld.PlaceDanger(0, 1)
}

func (w *Reinforce) createRandomGame() {
	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
	usedPositions := make([]struct{i int; j int}, 0)
	agentI := rand.Intn(4)
	agentJ := rand.Intn(4)
	usedPositions = append(usedPositions, struct{i int; j int}{agentI, agentJ})
	w.gridWorld.PlaceAgent(agentI, agentJ)
	for {
		i := rand.Intn(4)
		j := rand.Intn(4)
		if w.positionsContains(usedPositions, i, j) {
			continue
		}
		w.gridWorld.PlaceWall(i, j)
		usedPositions = append(usedPositions, struct{i int; j int}{i, j})
		break
	}
	for {
		i := rand.Intn(4)
		j := rand.Intn(4)
		if w.positionsContains(usedPositions, i, j) {
			continue
		}
		w.gridWorld.PlaceTarget(i, j)
		usedPositions = append(usedPositions, struct{i int; j int}{i, j})
		break
	}
	for {
		i := rand.Intn(4)
		j := rand.Intn(4)
		if w.positionsContains(usedPositions, i, j) {
			continue
		}
		w.gridWorld.PlaceDanger(i, j)
		usedPositions = append(usedPositions, struct{i int; j int}{i, j})
		break
	}
}

func (w *Reinforce) positionsContains(arr []struct{i int; j int}, i, j int) bool {
	for k := 0; k < len(arr); k++ {
		if arr[k].i == i && arr[k].j == j {
			return true
		}
	}
	return false
}

func (w *Reinforce) addNoise(mat *mat.Mat32f) *mat.Mat32f {
	mat.Apply(func(f float32) float32 {
		return f + rand.Float32() / 10
	})
	return mat
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
