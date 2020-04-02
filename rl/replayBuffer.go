package rl

import (
	"github.com/gokadin/ml-framework/mat"
	"math/rand"
)

type ReplayBuffer struct {
	bufferSize int
	batchSize int
	experiences []*replayExperience
	pointer int
	isFull bool
}

func NewReplayBuffer(bufferSize, batchSize int) *ReplayBuffer {
	return &ReplayBuffer{
		bufferSize: bufferSize,
		batchSize: batchSize,
		experiences: make([]*replayExperience, bufferSize),
	}
}

func (rb *ReplayBuffer) Append(oldState, newState *mat.Mat32f, action, reward int) {
	rb.experiences[rb.pointer] = newReplayExperience(oldState, newState, action, reward)

	if rb.pointer == rb.bufferSize - 1 {
		rb.pointer = 0
		if !rb.isFull {
			rb.isFull = true
		}

		return
	}

	rb.pointer++
}

func (rb *ReplayBuffer) IsFull() bool {
	return rb.isFull
}

func (rb *ReplayBuffer) NextBatch() []*replayExperience {
	batch := make([]*replayExperience, rb.batchSize)
	for i, randomIndex := range rand.Perm(rb.bufferSize) {
		batch[i] = rb.experiences[randomIndex]
		if i == rb.batchSize - 1 {
			break
		}
	}
	return batch
}
