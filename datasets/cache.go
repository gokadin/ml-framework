package datasets

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
)

const cacheRootDir = ".ml-framework-cache"

func createCache() {
	_ = os.Mkdir(getCacheRoot(), os.ModePerm)
}

func isCached(filename string) bool {
	_, err := os.Stat(fmt.Sprintf("%s/%s", getCacheRoot(), filename))
	return !os.IsNotExist(err)
}

func addToCache(filename string, content []byte) {
	out, err := os.Create(fmt.Sprintf("%s/%s", getCacheRoot(), filename))
	if err != nil {
		log.Fatal("Could not write to cache")
	}
	defer out.Close()

	bytesWritten, err := out.Write(content)
	_ = bytesWritten
	if err != nil {
		log.Fatal("Could not write to cache file")
	}
}

func getCachedFile(filename string) []byte {
	file, err := os.Open(fmt.Sprintf("%s/%s", getCacheRoot(), filename))
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

func getCacheRoot() string {
	return os.TempDir() + "/" + cacheRootDir
}
