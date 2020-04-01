package rl

import "github.com/gokadin/ml-framework/mat"

type replayExperience struct {
	oldState *mat.Mat32f
	newState *mat.Mat32f
	action int
	reward int
}

func newReplayExperience(oldState, newState *mat.Mat32f, action, reward int) *replayExperience {
	return &replayExperience{oldState, newState, action, reward}
}
