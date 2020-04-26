package rl

import (
	"fmt"
	gym "github.com/openai/gym-http-api/binding-go"
)

func gymClient() (*gym.Client, gym.InstanceID) {
	env, err := gym.NewClient("http://localhost:5000")
	must(err)

	insts, err := env.ListAll()
	must(err)
	fmt.Println("Started with instances:", insts)

	id, err := env.Create("CartPole-v0")
	must(err)

	actSpace, err := env.ActionSpace(id)
	must(err)
	fmt.Printf("Action space: %+v\n", actSpace)

	return env, id
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func obsToState(obs []float64) []float32 {
	state := make([]float32, len(obs))
	for i := 0; i < len(obs); i++ {
		state[i] = float32(obs[i])
	}
	return state
}
