package rl

import (
	"github.com/encryptio/alias"
	"math/rand"
	"time"
)

type agent struct {
	actions []int
}

func newAgent() *agent {
	return &agent{
		actions: make([]int, 10),
	}
}

func (a *agent) choose(probabilities []float64) int {
	x, _ := alias.New(probabilities)
	index := x.Gen(rand.New(rand.NewSource(time.Now().UTC().UnixNano())))
	return int(index)
}