package rl

import "math/rand"

type state struct {
	websites []float32
	currentWebsite int
}

func newState() *state {
	s := &state{
		websites: make([]float32, 10),
		currentWebsite: 2,
	}

	for i := 0; i < len(s.websites); i++ {
		s.websites[i] = rand.Float32()
	}

	return s
}

func (s *state) currentState() int {
	return rand.Intn(10)
}

func (s *state) takeAction(action int) float32 {
	var reward float32
	for i := 0; i < 10; i++ {
		if rand.Float32() < s.websites[action] {
			reward++
		}
	}
	return reward
}
