# Kaggle Competition: Test your analytics skills by predicting which 
# New York Times blog articles will be the most popular
# The Analytics Edge, MITx, edX, April 2015 - May 2015

### Load the packages
# RWeka (R interface to Weka, a collection of data mining algorithms)
library(RWeka)
# tm (Text Mining Package)
library(tm)
# randomForest (Breiman and Cutler's random forests for classification and regression)
library(randomForest)

# Load the datasets
newsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
newsTest = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

# Format date/time using using strptime function
newsTrain$PubDate = strptime(newsTrain$PubDate, "%Y-%m-%d %H:%M:%S")
newsTest$PubDate = strptime(newsTest$PubDate, "%Y-%m-%d %H:%M:%S")

# Create a corpus from the Headline, Abstract and Snippet variables
train = paste(newsTrain$Headline, newsTrain$Abstract, newsTrain$Snippet, paste = " ")
test = paste(newsTest$Headline, newsTest$Abstract, newsTest$Snippet, paste = " ")
corpus = Corpus(VectorSource(c(train, test)))

# Convert to lower-case
corpus = tm_map(corpus, tolower)
# Convert to a Plain Text Document
corpus = tm_map(corpus, PlainTextDocument)
# Remove punctuation
corpus = tm_map(corpus, removePunctuation)
# Remove stopwords
corpus = tm_map(corpus, removeWords, stopwords("english"))
# Stem document
corpus = tm_map(corpus, stemDocument)

### Text analytics (first prediction)
# the frequency of 1-grams (create matrix, remove sparse terms, create data frame)
tokenizer1 <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
dtm1 <- DocumentTermMatrix(corpus, control = list(tokenize = tokenizer1))
sparse1 = removeSparseTerms(dtm1, 0.965)
words1 = as.data.frame(as.matrix(sparse1))
# the frequency of 2-grams (create matrix, remove sparse terms, create data frame)
tokenizer2 <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
dtm2 <- DocumentTermMatrix(corpus, control = list(tokenize = tokenizer2))
sparse2 = removeSparseTerms(dtm2, 0.9905)
words2 = as.data.frame(as.matrix(sparse2))
# the frequency of 3-grams (create matrix, remove sparse terms, create data frame)
tokenizer3 <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
dtm3 <- DocumentTermMatrix(corpus, control = list(tokenize = tokenizer3))
sparse3 = removeSparseTerms(dtm3, 0.9925)
words3 = as.data.frame(as.matrix(sparse3))

# wordsTrain and wordsTest data frames (first prediction)
words = cbind(words1, words2, words3)
colnames(words) = make.names(colnames(words))
wordsTrain = head(words, nrow(newsTrain))
wordsTest = tail(words, nrow(newsTest))

# Factor analysis (first prediction)
popular0 = subset(wordsTrain, newsTrain$Popular == 0)
popular1 = subset(wordsTrain, newsTrain$Popular == 1)
means0 = colMeans(popular0)
means1 = colMeans(popular1)
values0 = means0[means1 > 0]
values1 = means1[means1 > 0]
variability = sort(abs(log10(values0/values1)), decreasing = T)[1:12]
wordsTrain = wordsTrain[,names(variability)]
wordsTest = wordsTest[,names(variability)]

# Combine all variables (first prediction)
wordsTrain$Popular = newsTrain$Popular
wordsTrain$NewsDesk = as.factor(newsTrain$NewsDesk)
wordsTest$NewsDesk = as.factor(newsTest$NewsDesk)
wordsTrain$SectionName = as.factor(newsTrain$SectionName)
wordsTest$SectionName = as.factor(newsTest$SectionName)
wordsTrain$SubsectionName = as.factor(newsTrain$SubsectionName)
wordsTest$SubsectionName = as.factor(newsTest$SubsectionName)
wordsTrain$Weekday = as.factor(newsTrain$PubDate$wday)
wordsTest$Weekday = as.factor(newsTest$PubDate$wday)
wordsTrain$Hour = as.factor(newsTrain$PubDate$hour)
wordsTest$Hour = as.factor(newsTest$PubDate$hour)
wordsTrain$WordCount = newsTrain$WordCount
wordsTest$WordCount = newsTest$WordCount

### Text analytics (second prediction)
# the frequency of n-grams (create matrix, remove sparse terms, create data frame)
tokenizer_ <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 6))
dtm_ <- DocumentTermMatrix(corpus, control = list(tokenize = tokenizer_))
sparse_ = removeSparseTerms(dtm_, 0.992)

# wordsTrain_ and wordsTest_ data frames (second prediction)
words_ = as.data.frame(as.matrix(sparse_))
colnames(words_) = make.names(colnames(words_))
wordsTrain_ = head(words_, nrow(newsTrain))
wordsTest_ = tail(words_, nrow(newsTest))

# Combine all variables (second prediction)
wordsTrain_$Popular = newsTrain$Popular
wordsTrain_$NewsDesk = as.factor(newsTrain$NewsDesk)
wordsTest_$NewsDesk = as.factor(newsTest$NewsDesk)
wordsTrain_$SectionName = as.factor(newsTrain$SectionName)
wordsTest_$SectionName = as.factor(newsTest$SectionName)
wordsTrain_$SubsectionName = as.factor(newsTrain$SubsectionName)
wordsTest_$SubsectionName = as.factor(newsTest$SubsectionName)
wordsTrain_$WordCount = newsTrain$WordCount
wordsTest_$WordCount = newsTest$WordCount

# Make the first Random Forest model
model1 = randomForest(Popular ~ ., data=wordsTrain)
# Inherit the factor levels from the bigger (train) data set to the smaller (test) set
wordsTest$NewsDesk = factor(wordsTest$NewsDesk, levels=levels(wordsTrain$NewsDesk))
wordsTest$SectionName = factor(wordsTest$SectionName, levels=levels(wordsTrain$SectionName))
wordsTest$SubsectionName = factor(wordsTest$SubsectionName, levels=levels(wordsTrain$SubsectionName))
# Make the first prediction
prediction1 = predict(model1, newdata=wordsTest, type="response")

# Make the second Random Forest model
model2 = randomForest(Popular ~ ., data=wordsTrain_)
# Inherit the factor levels from the bigger (train) data set to the smaller (test) set
wordsTest_$NewsDesk = factor(wordsTest_$NewsDesk, levels=levels(wordsTrain_$NewsDesk))
wordsTest_$SectionName = factor(wordsTest_$SectionName, levels=levels(wordsTrain_$SectionName))
wordsTest_$SubsectionName = factor(wordsTest_$SubsectionName, levels=levels(wordsTrain_$SubsectionName))
# Make the second prediction
prediction2 = predict(model2, newdata=wordsTest_, type="response")

# Final prediction as the weighted combination of both predictions
finalPrediction = prediction1 * 0.667 + prediction2 * 0.333
# Prepare the submission file for Kaggle
MySubmission = data.frame(UniqueID = newsTest$UniqueID, Probability1 = finalPrediction)
write.csv(MySubmission, "predictNYT.csv", row.names=FALSE)
