package datasets

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
)

const cacheRootDir = ".cache"

func createCache() {
	_ = os.Mkdir(cacheRootDir, os.ModePerm)
}

func isCached(filename string) bool {
	_, err := os.Stat(fmt.Sprintf("%s/%s", cacheRootDir, filename))
	return !os.IsNotExist(err)
}

func addToCache(filename string, reader io.Reader) {
	out, err := os.Create(fmt.Sprintf("%s/%s", cacheRootDir, filename))
	if err != nil {
		log.Fatal("Could not write to cache")
	}
	defer out.Close()

	_, err = io.Copy(out, reader)
}

func getCachedFile(filename string) []byte {
	file, err := os.Open(fmt.Sprintf("%s/%s", cacheRootDir, filename))
	if err != nil {
		log.Fatal("Could not retrieve file from cache")
	}
	defer file.Close()

	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatal("Could not decode file from cache")
	}
	return bytes
}