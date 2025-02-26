def compute_unigram_laplace(fileName):
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
        tokenDict.update({key: (tokenDict[key]+ 1)/(wordCount + len(tokenDict))})
        total += tokenDict[key]
    print(tokenDict)
    return tokenDict
            

if __name__ == '__main__':
    unigram = compute_unigram_laplace("train.txt")