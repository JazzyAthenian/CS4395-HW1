import re
import math
from nltk.stem import WordNetLemmatizer

def preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\.\-]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = '. ' + text  

    wnl = WordNetLemmatizer()
    tokens = text.split()
    tokens = [wnl.lemmatize(token) for token in tokens]
    return tokens

def compute_unigram(fileName):
    """Compute unigram probabilities without smoothing."""
    with open(fileName, "r") as f:
        tokenized = f.read().split()
    
    tokenDict = {}

    tokenDict = {"<UNK>" : 1}

    wordCount = 1

    for token in tokenized:
        tokenDict[token] = tokenDict.get(token, 0) + 1
        wordCount += 1

    for key in tokenDict:
        tokenDict[key] /= wordCount
    
    return tokenDict

def compute_unigram_laplace(fileName):
    f = open(fileName, "r")
    tokenized = f.read().split() # Default tokenizer is a space
    tokenDict = {"<UNK>" : 1}

    wordCount = 1

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
    
    return tokenDict

def compute_unigram_add_k(fileName, k):
    """Compute unigram probabilities with add-k smoothing."""
    with open(fileName, "r") as f:
        tokenized = f.read().split()
    
    tokenDict = {"<UNK>" : 1}

    wordCount = 1

    for token in tokenized:
        tokenDict[token] = tokenDict.get(token, 0) + 1
        wordCount += 1

    vocab_size = len(tokenDict)

    for key in tokenDict:
        tokenDict[key] = (tokenDict[key] + k) / (wordCount + (k * vocab_size))
    
    return tokenDict

def compute_bigram(fileName, smoothing=None, k=None):
    """Compute bigram probabilities with optional smoothing."""
    bigrams = {}
    word_count = {"<UNK>":1 }
    with open(fileName, "r") as f:
        for line in f:
            tokens = preprocessing(line)
            tokens = ["<UNK>"] + tokens
            for i in range(len(tokens) - 1):
                word1, word2 = tokens[i], tokens[i+1]

                if word1 not in bigrams:
                    bigrams[word1] = {}

                bigrams[word1][word2] = bigrams[word1].get(word2, 0) + 1
                word_count[word1] = bigrams[word1].get(word1, 0) + 1
    for word in bigrams:
        total = sum(bigrams[word].values())
        vocab_size = len(word_count)

        if smoothing == "laplace":
            bigrams[word] = {word2: (v + 1) / (total + vocab_size) for word2, v in bigrams[word].items()}
        elif smoothing == "add_k":
            bigrams[word] = {word2: (v + k) / (total + (k * vocab_size)) for word2, v in bigrams[word].items()}
        else:
            bigrams[word] = {word2: v / total for word2, v in bigrams[word].items()}
    
    return bigrams

def processing_test(bigram, fileName):
    f = open(fileName, "r")
    test_data = []
    
    for line in f:
        tokens = preprocessing(line)

        for i in range(len(tokens)):
            word = tokens[i]
            
            if word not in bigram:
                test_data.append("<UNK>")
            else: 
                test_data.append(word)
            
    return test_data

def compute_perplexity(model, test_file, ngram="unigram", smoothing="none", k=None):
    """Compute perplexity for unigram or bigram models."""
    test_tokens = processing_test(model, test_file)

    log_prob_sum = 0
    count = 0

    if ngram == "unigram":
        vocab_size = len(model)
        for word in test_tokens:
            if word in model:
                prob = model[word]
            else:
                if smoothing == "add_k":
                    prob = k / (sum(model.values()) + k * vocab_size)
                else:
                    prob = 1 / (sum(model.values()) + vocab_size)  # Laplace smoothing

            log_prob_sum += math.log(prob)
            count += 1

    elif ngram == "bigram":
        vocab_size = len(model)
        for i in range(len(test_tokens) - 1):
            word1, word2 = test_tokens[i], test_tokens[i+1]
            if word1 in model and word2 in model[word1]:
                prob = model[word1][word2]
            else:
                if smoothing == "add_k":
                    prob = k / (sum(model[word1].values()) + k * vocab_size) if word1 in model else k / vocab_size
                else:
                    prob = 1 / (sum(model[word1].values()) + vocab_size) if word1 in model else 1 / vocab_size

            log_prob_sum += math.log(prob)
            count += 1

    perplexity = math.exp(-log_prob_sum / count) if count > 0 else float('inf')
    return perplexity

if __name__ == '__main__':
    train_file = "train.txt"
    test_file = "val.txt"

    # Compute unigram models
    unigram = compute_unigram(train_file)
    unigram_add_k = compute_unigram_add_k(train_file, k=0.2)
    unigram_add_k1 = compute_unigram_add_k(train_file, k=0.5)
    unigram_laplace = compute_unigram_laplace(train_file)

    # Compute bigram models
    bigram = compute_bigram(train_file)
    bigram_laplace = compute_bigram(train_file, smoothing="laplace")
    bigram_add_k = compute_bigram(train_file, smoothing="add_k", k=0.2)
    bigram_add_k1 = compute_bigram(train_file, smoothing="add_k", k=0.5)


    # Compute perplexity
    print("Perplexity Scores:")
    print(f"Unigram (No Smoothing): {compute_perplexity(unigram, test_file, ngram='unigram')}")
    print(f"Unigram (Laplace): {compute_perplexity(unigram_laplace, test_file, ngram="unigram", smoothing='laplace')}")
    print(f"Unigram (Add-k, k=0.2): {compute_perplexity(unigram_add_k, test_file, ngram='unigram', smoothing='add_k', k=0.2)}")
    print(f"Unigram (Add-k, k=0.5): {compute_perplexity(unigram_add_k1, test_file, ngram='unigram', smoothing='add_k', k=0.5)}")

    print()

    print(f"Bigram (No Smoothing): {compute_perplexity(bigram, test_file, ngram='bigram')}")
    print(f"Bigram (Laplace): {compute_perplexity(bigram_laplace, test_file, ngram='bigram', smoothing='laplace')}")
    print(f"Bigram (Add-k, k=0.2): {compute_perplexity(bigram_add_k, test_file, ngram='bigram', smoothing='add_k', k=0.2)}")
    print(f"Bigram (Add-k, k=0.5): {compute_perplexity(bigram_add_k1, test_file, ngram='bigram', smoothing='add_k', k=0.5)}")

