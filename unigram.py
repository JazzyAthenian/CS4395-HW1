def compute_unigram(fileName):
    f = open(fileName, "r")
    tokenized = f.read().split() # Default tokenizer is a space
    tokenDict = dict()
    wordCount = 0

    # Filling out the dictionary
    for token in tokenized:
        if token in tokenDict:
            tokenDict.update({token: tokenDict[token]+1})
        else:
            tokenDict.update({token: 1})
        wordCount = wordCount + 1

    # Now, calculating weights
    total = 0
    for key in tokenDict:
        tokenDict.update({key: tokenDict[key]/wordCount})
        total += tokenDict[key]
    print(tokenDict)

if __name__ == '__main__':
    compute_unigram("train.txt")