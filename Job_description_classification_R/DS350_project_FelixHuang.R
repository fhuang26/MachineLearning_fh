# --------------------------------------------------------------------------------------------------------
#
#  UW DS350  	 	    Project			                Felix Huang
#
#
# (0) Data pre-processing : to prepare job description data
#
# --------------------------------------------------------------------------------------------------------

rm( list = ls())  # Clear environment
# setwd('C:/Users/fhuan/Documents/Felix/UW Data Science/DS 350 Methods for Data Analysis Nick McClure/week 9 NLP')
jd = read.csv("train.tsv", sep="\t")
# 1580 x 2; two fields: tags, description
str(jd)
jd$tags = as.character(jd$tags)
jd$d = as.character(jd$description)

jd$description = NULL
str(jd)

jd$d = tolower(jd$d)
head(jd)
jd$d = sapply(jd$d, function(x) paste2(strsplit(x, "[ \t\n]b[.]s[.][ \t\n]")[[1]], sep=' bs '))
jd$d = sapply(jd$d, function(x) gsub("'", "", x))
# Now the rest of the punctuation
jd$d = sapply(jd$d, function(x) gsub("[[:punct:]]", " ", x))
head(jd)

# Remove extra white space, so we can split words by spaces
jd$d = sapply(jd$d, function(x) gsub("[ ]+"," ",x))
head(jd)

# Remove non-ascii
jd$d = iconv(jd$d, from="latin1", to="ASCII", sub="")
jd$d = sapply(jd$d, function(x) paste2(strsplit(x, "[ \t\n]bs")[[1]], sep=' bs '))
head(jd)

# remove stopwords; be careful doing this in the right order
# stopwords()
my_stops = as.character(sapply(stopwords(), function(x) gsub("'","",x)))
jd$d = sapply(jd$d, function(x){
  paste(setdiff(strsplit(x," ")[[1]], my_stops),collapse=" ")
})
# Remove extra white space, tab, or new line
jd$d = sapply(jd$d, function(x) gsub("[ \t\n]+"," ",x))
write.csv(jd, file="jd.csv", row.names=FALSE)

# --------------------------------------------------------------------------------------------------------
#
# (1) Naive-Bayes for 'associate-needed' label
#
# --------------------------------------------------------------------------------------------------------

# rm( list = ls())  # Clear environment
jd = read.csv("jd.csv")
nrow(jd)
jd$tags = as.character(jd$tags)
jd$d = as.character(jd$d)

jd_as = jd
jd_as$d = sapply(jd_as$d, function(x) paste("$", x, "$"))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associates bachelors degree")[[1]], sep=' associatesBachelorsDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associates degree")[[1]], sep=' associatesDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associate degree")[[1]], sep=' associateDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "university graduate")[[1]], sep=' universityGraduate '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college degree")[[1]], sep=' collegeDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "two year college")[[1]], sep=' twoYearCollege '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college courses")[[1]], sep=' collegeCourses '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college certificate")[[1]], sep=' collegeCertificate '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college certifications")[[1]], sep=' collegeCertifications '))

# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
jd$tags = sapply(jd$tags, function(x) strsplit(x, " ")[[1]])
jd$as = sapply(jd$tags, function(x) ifelse(("associate-needed" %in% x),TRUE,FALSE))
for (k in 1:1580) {
  if (jd$as[k] == TRUE) {
    jd_as$tags[k] = "associate-needed"
  } else {
    jd_as$tags[k] = ""
  }
}
as_key_words = c('associatesBachelorsDegree', 'associatesDegree', 'associateDegree', 'universityGraduate', 'collegeCourses',
                 'degree', 'twoYearCollege', 'college', 'grads', 'graduates', 'graduate', 'collegeCertificate',
                 'collegeCertifications', 'collegeDegree');

expand_context = function(b,pad) {
  b2 = c(b[1])
  blen = length(b)
  for (j in 2:blen) {
    if (!is.null(b[j])) {
      f = b[j]
      if (f == FALSE) {
        for (t in 1:pad) {
          if (j > t && b[j-t] == TRUE) f = TRUE
          if (j <= blen-t && b[j+t] == TRUE) f = TRUE
        }
      }
      b2 = c(b2,f)
    } else {
      b2 = c(b2,FALSE)
    }
  }
  return(b2)
}
# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
for (k in 1:nrow(jd_as)) {
  da = strsplit(jd_as$d[k], " ")[[1]]
  if (da[1] %in% as_key_words) {
    b = c(TRUE)
  } else {
    b = c(FALSE)
  }
  for (i in 2:length(da)) {
    x = da[i]
    if (x %in% as_key_words) {
      b = c(b,TRUE)
    } else {
      b = c(b,FALSE)
    }
  }
  if (sum(b) == 0) {
    jd_as$d[k] = "-"
  } else {
    b2 = expand_context(b, 7)
    jd_as$d[k] = paste2(da[b2], sep=" ")
  }
}
# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
write.csv(jd_as, file="jd_as.csv", row.names=FALSE)

# Create a Corpus (matrix of frequent terms)
##-----Text Corpus-----
# We have to tell R that our collection of reviews is really a corpus.
text_corpus = Corpus(VectorSource(jd_as$d))
head(text_corpus)
text_term_matrix = DocumentTermMatrix(text_corpus)
dim(text_term_matrix)
# 1580 x 2495

# NOTE BE CAREFUL DOING THIS WITH LARGER DATA SETS!!!!!!
text_corpus_mat = as.matrix(text_term_matrix)
dim(text_corpus_mat)

# Convert to Data Frame
text_frame = as.data.frame(text_corpus_mat)
text_frame$tags = jd_as$tags
head(text_frame)

# Convert to factors:
tf_fac = as.data.frame(lapply(text_frame, as.factor))

# Split into train/test set
train_ind = sample(1:nrow(tf_fac), round(0.8*nrow(tf_fac)))
train_set = tf_fac[train_ind,]
test_set  = tf_fac[-train_ind,]
# test_idx = test_set$idx
# train_set$idx = NULL
# test_set$idx  = NULL
# Compute Naive Bayes Model
text_nb_as = naiveBayes(tags ~ ., data=train_set)
test_pred_as = predict(text_nb_as, newdata=test_set)
result = (test_pred_as == test_set$tags)
accuracy_nb_as = sum(result)/nrow(test_set)
accuracy_nb_as
# 0.9208861

# --------------------------------------------------------------------------------------------------------
#
# (2) bs-degree-needed : Naive Bayes
#
# --------------------------------------------------------------------------------------------------------

rm( list = ls())  # Clear environment
jd = read.csv("jd.csv")
nrow(jd)
jd$tags = as.character(jd$tags)
jd$d = as.character(jd$d)

jd_bs = jd
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "bachelor degree")[[1]], sep=' bachelorDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "bachelors degree")[[1]], sep=' bachelorsDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "four year college")[[1]], sep=' fourYearCollege '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "college degree")[[1]], sep=' collegeDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "college graduate")[[1]], sep=' collegeGraduate '))

# Remove extra white space, tab, or new line
jd_bs$d = sapply(jd_bs$d, function(x) gsub("[ \t\n]+"," ",x))
jd$tags = sapply(jd$tags, function(x) strsplit(x, " ")[[1]])
jd$as = sapply(jd$tags, function(x) ("bs-degree-needed" %in% x))
for (k in 1:1580) {
  if (jd$as[k] == TRUE) {
    jd_bs$tags[k] = "bs-degree-needed"
  } else {
    jd_bs$tags[k] = ""
  }
}
bs_key_words = c('bachelorDegree', 'bachelorsDegree', 'fourYearCollege', 'collegeGraduate', 'collegeDegree', 'college', 'bs', 'ba');

expand_context = function(b,pad) {
  b2 = c(b[1])
  blen = length(b)
  for (j in 2:blen) {
    if (!is.null(b[j])) {
      f = b[j]
      if (f == FALSE) {
        for (t in 1:pad) {
          if (j > t && b[j-t] == TRUE) f = TRUE
          if (j <= blen-t && b[j+t] == TRUE) f = TRUE
        }
      }
      b2 = c(b2,f)
    } else {
      b2 = c(b2,FALSE)
    }
  }
  return(b2)
}
# Remove extra white space, tab, or new line
jd_bs$d = sapply(jd_bs$d, function(x) gsub("[ \t\n]+"," ",x))
for (k in 1:nrow(jd_bs)) {
  da = strsplit(jd_bs$d[k], " ")[[1]]
  if (da[1] %in% bs_key_words) {
    b = c(TRUE)
  } else {
    b = c(FALSE)
  }
  for (i in 2:length(da)) {
    x = da[i]
    if (x %in% bs_key_words) {
      b = c(b,TRUE)
    } else {
      b = c(b,FALSE)
    }
  }
  if (sum(b) == 0) {
    jd_bs$d[k] = "-"
  } else {
    b2 = expand_context(b, 3)
    jd_bs$d[k] = paste2(da[b2], sep=" ")
  }
}
# Remove extra white space, tab, or new line
jd_bs$d = sapply(jd_bs$d, function(x) gsub("[ \t\n]+"," ",x))
write.csv(jd_bs, file="jd_bs.csv", row.names=TRUE)

# Create a Corpus (matrix of frequent terms)
##-----Text Corpus-----
# We have to tell R that our collection of reviews is really a corpus.
text_corpus = Corpus(VectorSource(jd_bs$d))
head(text_corpus)
text_term_matrix = DocumentTermMatrix(text_corpus)
dim(text_term_matrix)
# 1580 x 968
# 1580 x 25233

# NOTE BE CAREFUL DOING THIS WITH LARGER DATA SETS!!!!!!
text_corpus_mat = as.matrix(text_term_matrix)
dim(text_corpus_mat)

# Convert to Data Frame
text_frame = as.data.frame(text_corpus_mat)
text_frame$tags = jd_bs$tags
# index_v = c(1)
# for (k in 2:nrow(text_frame)) index_v = c(index_v, k)
# text_frame$idx = index_v
head(text_frame)
text_frame$idx[1580]
# Convert to factors:
tf_fac = as.data.frame(lapply(text_frame, as.factor))

# Split into train/test set
train_ind = sample(1:nrow(tf_fac), round(0.8*nrow(tf_fac)))
train_set = tf_fac[train_ind,]
test_set  = tf_fac[-train_ind,]
# test_idx = test_set$idx
# train_set$idx = NULL
# test_set$idx  = NULL
# Compute Naive Bayes Model
text_nb = naiveBayes(tags ~ ., data=train_set)
test_pred_bs = predict(text_nb, newdata=test_set)
for (k in 1:length(test_pred_bs)) {
  if (test_pred_bs[k] == TRUE && test_pred_as[k] == TRUE) {
    test_pred_bs[k] = FALSE
  }
}
result = (test_pred_bs == test_set$tags)
accuracy = sum(result)/nrow(test_set)
accuracy
# 0.8924051 for bs Naive-Bayes

# --------------------------------------------------------------------------------------------------------
#
# (3) Neural Network for 'associate-needed' label
#
# --------------------------------------------------------------------------------------------------------

jd = read.csv("jd.csv")
nrow(jd)
jd$tags = as.character(jd$tags)
jd$d = as.character(jd$d)

jd_as = jd
jd_as$d = sapply(jd_as$d, function(x) paste("$", x, "$"))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associates bachelors degree")[[1]], sep=' associatesBachelorsDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associates degree")[[1]], sep=' associatesDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associate degree")[[1]], sep=' associateDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "university graduate")[[1]], sep=' universityGraduate '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college degree")[[1]], sep=' collegeDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "two year college")[[1]], sep=' twoYearCollege '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college courses")[[1]], sep=' collegeCourses '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college certificate")[[1]], sep=' collegeCertificate '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college certifications")[[1]], sep=' collegeCertifications '))

# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
jd$tags = sapply(jd$tags, function(x) strsplit(x, " ")[[1]])
jd$as = sapply(jd$tags, function(x) ifelse(("associate-needed" %in% x),TRUE,FALSE))
for (k in 1:1580) {
  if (jd$as[k] == TRUE) {
    jd_as$tags[k] = "associate-needed"
  } else {
    jd_as$tags[k] = ""
  }
}
as_key_words = c('associatesBachelorsDegree', 'associatesDegree', 'associateDegree', 'universityGraduate', 'collegeCourses',
                 'degree', 'twoYearCollege', 'college', 'grads', 'graduates', 'graduate', 'collegeCertificate',
                 'collegeCertifications', 'collegeDegree');

expand_context = function(b,pad) {
  b2 = c(b[1])
  blen = length(b)
  for (j in 2:blen) {
    if (!is.null(b[j])) {
      f = b[j]
      if (f == FALSE) {
        for (t in 1:pad) {
          if (j > t && b[j-t] == TRUE) f = TRUE
          if (j <= blen-t && b[j+t] == TRUE) f = TRUE
        }
      }
      b2 = c(b2,f)
    } else {
      b2 = c(b2,FALSE)
    }
  }
  return(b2)
}
# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
for (k in 1:nrow(jd_as)) {
  da = strsplit(jd_as$d[k], " ")[[1]]
  if (da[1] %in% as_key_words) {
    b = c(TRUE)
  } else {
    b = c(FALSE)
  }
  for (i in 2:length(da)) {
    x = da[i]
    if (x %in% as_key_words) {
      b = c(b,TRUE)
    } else {
      b = c(b,FALSE)
    }
  }
  if (sum(b) == 0) {
    jd_as$d[k] = "-"
  } else {
    b2 = expand_context(b, 7)
    jd_as$d[k] = paste2(da[b2], sep=" ")
  }
}
# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
write.csv(jd_as, file="jd_as.csv", row.names=FALSE)

jd_as = read.csv("jd_as.csv")
jd_as$tags = as.character(jd_as$tags)
jd_as$d = as.character(jd_as$d)

text_corpus = Corpus(VectorSource(jd_as$d))
head(text_corpus)
text_term_matrix = DocumentTermMatrix(text_corpus)
dim(text_term_matrix)
# 1580 x 2330

# NOTE BE CAREFUL DOING THIS WITH LARGER DATA SETS!!!!!!
text_corpus_mat = as.matrix(text_term_matrix)
dim(text_corpus_mat)
# 1580 x 2330

# Convert to Data Frame
tf_as = as.data.frame(text_corpus_mat)
colNames = colnames(tf_as)
N_col = length(colNames)
t = which(colNames == 'associatesdegree')
# column 28

as_key_words_little = tolower(as_key_words)
init_coeff = as.numeric(colNames %in% as_key_words_little)
length(init_coeff)
sum(init_coeff)
init_coeff[t]
tf_as$as_label = rep(0,nrow(tf_as))
for (k in 1:nrow(tf_as)) {
  if (jd$as[k] == TRUE) {
    tf_as$as_label[k] = 1
  } else {
    tf_as$as_label[k] = 0
  }
}
str(tf_as$as_label)
sum(tf_as$as_label)

# Sigmoid
sigmoid = function(x){
  return(1/(1 + exp(-x)))
}

# sigmoid is special because its derivative is related to itself
d_sigmoid = function(x){
  return( sigmoid(x) * (1 - sigmoid(x)) )
}

accuracy_nn_test_v = c()
for (h in 1:5) {
# Split into train/test set
train_ind = sample(1:nrow(tf_as), round(0.8*nrow(tf_as)))
train_nn = tf_as[train_ind,]
test_nn  = tf_as[-train_ind,]
nrow(train_nn)

coeffi_v = init_coeff

b = 0 # Intercept
step_size = 0.1 # starts from larger step and gradually decreases
n_loops = 8000
rem_n_loops = n_loops/2 # for every half loops, divide step_size by 2
c28 = c()  # coefficient for an important feature, associatesDegree
for (j in 1:n_loops) {
  # Select a random point
  idx_arr = 1:nrow(train_nn)
  rem = j %% 3
  rnd_idx = idx_arr[idx_arr %% 3 == rem]
  k = sample(rnd_idx, 1)  # random row index from alternating 1/3 of train_nn
  rand_row = as.matrix(train_nn[k,])
  
  # calculate network output for this random row
  inner_product = rand_row[1:N_col] %*% coeffi_v
  network_out = sigmoid(inner_product + b)  # b is intercept.
  
  # Determine if we need to make it greater or less
  actual = train_nn$as_label[k]
  
  if (((actual==1) & (network_out>0.5)) | ((actual==0) & (network_out<=0.5))){
    pull = 0 # Correctly identified, no need to change
  } else if ((actual==1) & (network_out<=0.5)){
    pull = +1 # False negative, pull in positive direction
  } else if ((actual==0) & (network_out>0.5)){
    pull = -1 # False positive, pull in negative direction
  }
  
  # to modify coefficients
  d_sig = d_sigmoid(inner_product)
  for (i in 1:N_col) {
    c_gradient = d_sig * rand_row[i]  # this is chain rule. df/dx = (df/dg) * (dg/dx), df/dg = d_sig and dg/dx = y = rand_row[i]
    coeffi_v[i] = coeffi_v[i] + (pull * step_size * c_gradient)
  }
  b_gradient = d_sig  # d_sig * 1: 1 for add_gradient
  b = b + (pull * step_size * b_gradient)  # Intercept
  
  a = coeffi_v[28]
  c28 = c(c28, a)
  if (j %% 20 == 0) {
    cat("j=",j,"  c28=",a,"  b=",b,"\n")
  }
  
  if (j == n_loops - rem_n_loops && rem_n_loops > 30) {
    step_size = step_size / 2
    rem_n_loops = rem_n_loops / 2
    cat("j=",j,"  step_size=",step_size,"\n")
  }
}

plot(c28)
coeffi_v
mt3 = as.matrix(train_nn)
f = sigmoid((mt3[,1:N_col] %*% coeffi_v) + rep(b, nrow(mt3)))
y_pred = ifelse((f > 0.5),1,0)
sum(y_pred)
accuracy_nn_train = sum(y_pred == train_nn$as_label)/nrow(mt3)
accuracy_nn_train
# 0.9912975

mt1 = as.matrix(test_nn)
f = sigmoid((mt1[,1:N_col] %*% coeffi_v) + rep(b, nrow(mt1)))
y_pred = ifelse((f > 0.5),1,0)
sum(y_pred)
accuracy_nn_test = sum(y_pred == test_nn$as_label)/nrow(mt1)
accuracy_nn_test
# 0.9493671
cat("h=",h,"  accuracy_nn_test=",accuracy_nn_test,"\n")
accuracy_nn_test_v = c(accuracy_nn_test_v, accuracy_nn_test)
}
accuracy_nn_test_v
# 0.9493671 0.9462025 0.9335443 0.9335443 0.9462025 0.9493671
mean(accuracy_nn_test_v)
# 0.943038
# 0.9455696 6-8-2017 8:59pm

# --------------------------------------------------------------------------------------------------------
#
# (4) Neural Network for 'bs-degree-needed' label
#
# --------------------------------------------------------------------------------------------------------

rm( list = ls())  # Clear environment
jd = read.csv("jd.csv")
nrow(jd)
jd$tags = as.character(jd$tags)
jd$d = as.character(jd$d)

jd_bs = jd
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "bachelor degree")[[1]], sep=' bachelorDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "bachelors degree")[[1]], sep=' bachelorsDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "four year college")[[1]], sep=' fourYearCollege '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "college degree")[[1]], sep=' collegeDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "college graduate")[[1]], sep=' collegeGraduate '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "associates bachelors degree")[[1]], sep=' associatesBachelorsDegree '))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "associates degree")[[1]], sep=' associatesDegree'))
jd_bs$d = sapply(jd_bs$d, function(x) paste2(strsplit(x, "associate degree")[[1]], sep=' associateDegree'))

# Remove extra white space, tab, or new line
jd_bs$d = sapply(jd_bs$d, function(x) gsub("[ \t\n]+"," ",x))
jd$tags = sapply(jd$tags, function(x) strsplit(x, " ")[[1]])
jd$bs = sapply(jd$tags, function(x) ("bs-degree-needed" %in% x))
for (k in 1:1580) {
  if (jd$bs[k] == TRUE) {
    jd_bs$tags[k] = "bs-degree-needed"
  } else {
    jd_bs$tags[k] = ""
  }
}
# for expanding context
bs_key_words = c('bachelorDegree', 'bachelorsDegree', 'fourYearCollege', 'collegeGraduate', 'collegeDegree', 'college', 'bs', 'ba',
                 'degree', 'preferred', 'associatesBachelorsDegree', 'associatesDegree', 'associateDegree');

bs_key_words_bs = c('bachelorDegree', 'bachelorsDegree', 'fourYearCollege', 'collegeGraduate', 'collegeDegree', 'college', 'bs', 'ba');
# for neural network train

expand_context = function(b,pad) {
  b2 = c(b[1])
  blen = length(b)
  for (j in 2:blen) {
    if (!is.null(b[j])) {
      f = b[j]
      if (f == FALSE) {
        for (t in 1:pad) {
          if (j > t && b[j-t] == TRUE) f = TRUE
          if (j <= blen-t && b[j+t] == TRUE) f = TRUE
        }
      }
      b2 = c(b2,f)
    } else {
      b2 = c(b2,FALSE)
    }
  }
  return(b2)
}
# Remove extra white space, tab, or new line
jd_bs$d = sapply(jd_bs$d, function(x) gsub("[ \t\n]+"," ",x))
for (k in 1:nrow(jd_bs)) {
  da = strsplit(jd_bs$d[k], " ")[[1]]
  if (da[1] %in% bs_key_words) {
    b = c(TRUE)
  } else {
    b = c(FALSE)
  }
  for (i in 2:length(da)) {
    x = da[i]
    if (x %in% bs_key_words) {
      b = c(b,TRUE)
    } else {
      b = c(b,FALSE)
    }
  }
  if (sum(b) == 0) {
    jd_bs$d[k] = "-"
  } else {
    b2 = expand_context(b, 6)
    jd_bs$d[k] = paste2(da[b2], sep=" ")
  }
}
# Remove extra white space, tab, or new line
jd_bs$d = sapply(jd_bs$d, function(x) gsub("[ \t\n]+"," ",x))
write.csv(jd_bs, file="jd_bs.csv", row.names=TRUE)

jd_bs = read.csv("jd_bs.csv")
jd_bs$tags = as.character(jd_bs$tags)
jd_bs$d = as.character(jd_bs$d)

text_corpus = Corpus(VectorSource(jd_bs$d))
head(text_corpus)
text_term_matrix = DocumentTermMatrix(text_corpus)
dim(text_term_matrix)
# 1580 x 2862

# NOTE BE CAREFUL DOING THIS WITH LARGER DATA SETS!!!!!!
text_corpus_mat = as.matrix(text_term_matrix)
dim(text_corpus_mat)
# 1580 x 2862

# Convert to Data Frame
tf_bs = as.data.frame(text_corpus_mat)
colNames = colnames(tf_bs)
N_col = length(colNames)
t = which(colNames == 'bachelorsdegree')
# column 51

bs_key_words_little = tolower(bs_key_words_bs)
init_coeff = as.numeric(colNames %in% bs_key_words_little)
length(init_coeff)
sum(init_coeff)
init_coeff[t]
tf_bs$bs_label = rep(0,nrow(tf_bs))
for (k in 1:nrow(tf_bs)) {
  if (jd$bs[k] == TRUE) {
    tf_bs$bs_label[k] = 1
  } else {
    tf_bs$bs_label[k] = 0
  }
}
str(tf_bs$bs_label)
sum(tf_bs$bs_label)
# 327

# Sigmoid
sigmoid = function(x){
  return(1/(1 + exp(-x)))
}

# sigmoid is special because its derivative is related to itself
d_sigmoid = function(x){
  return( sigmoid(x) * (1 - sigmoid(x)) )
}

accuracy_nn_test_v = c()
for (h in 1:3) {
  # Split into train/test set
  train_ind = sample(1:nrow(tf_bs), round(0.8*nrow(tf_bs)))
  train_nn = tf_bs[train_ind,]
  test_nn  = tf_bs[-train_ind,]
  nrow(train_nn)
  # 1264
  
  coeffi_v = init_coeff
  
  b = 0 # Intercept
  step_size = 0.1 # starts from larger step and gradually decreases
  n_loops = 8000
  rem_n_loops = n_loops/2 # for every half loops, divide step_size by 2
  c43 = c()  # coefficient for an important feature, bachelorsDegree
  sa_d = 14  # simulated annealing, probability to increase positive more, and this probablity upper bound decreases to 0.
             # This may help climb hills to find closer to global optimal solution.
  for (j in 1:n_loops) {
    # Select a random point
    idx_arr = 1:nrow(train_nn)
    rem = j %% 3
    rnd_idx = idx_arr[idx_arr %% 3 == rem]
    k = sample(rnd_idx, 1)  # random row index from alternating 1/3 of train_nn
    rand_row = as.matrix(train_nn[k,])
    
    # calculate network output for this random row
    inner_product = rand_row[1:N_col] %*% coeffi_v
    network_out = sigmoid(inner_product + b)  # b is intercept.
    
    # Determine if we need to make it greater or less
    actual = train_nn$bs_label[k]
    
    if (((actual==1) & (network_out>0.5)) | ((actual==0) & (network_out<=0.5))){
      pull = 0 # Correctly identified, no need to change
    } else if ((actual==1) & (network_out<=0.5)){
      pull = +1 # False negative, pull in positive direction
    } else if ((actual==0) & (network_out>0.5)){
      pull = -1 # False positive, pull in negative direction
    }
    if (pull == 0 && sa_d > 0) {
      w = sample(1:100, 1)
      if (w <= sa_d) {
        if (w %% 2 == 0) {
          pull = 1
        } else {
          pull = -1
        }
      }
    }
    
    # to modify coefficients
    if (pull != 0) {
      d_sig = d_sigmoid(inner_product)
      for (i in 1:N_col) {
        c_gradient = d_sig * rand_row[i]  # this is chain rule. df/dx = (df/dg) * (dg/dx), df/dg = d_sig and dg/dx = y = rand_row[i]
        coeffi_v[i] = coeffi_v[i] + (pull * step_size * c_gradient)
      }
      b_gradient = d_sig  # d_sig * 1: 1 for add_gradient
      b = b + (pull * step_size * b_gradient)  # Intercept
    }
    
    a = coeffi_v[t]
    c43 = c(c43, a)
    if (j %% 20 == 0) {
      cat("j=",j,"  c43=",a,"  b=",b,"\n")
    }
    
    if (j == n_loops - rem_n_loops && rem_n_loops > 30) {
      step_size = step_size / 2
      sa_d = sa_d / 2
      if (sa_d <= 3) {
        sa_d = 0  # no simulated annealing in late stage to help convergence
      }
      rem_n_loops = rem_n_loops / 2
      cat("j=",j,"  step_size=",step_size,"  sa_d=",sa_d,"\n")
    }
  }
  
  plot(c43)
  coeffi_v
  mt3 = as.matrix(train_nn)
  f = sigmoid((mt3[,1:N_col] %*% coeffi_v) + rep(b, nrow(mt3)))
  y_pred = ifelse((f > 0.5),1,0)
  sum(y_pred)
  accuracy_nn_train = sum(y_pred == train_nn$bs_label)/nrow(mt3)
  accuracy_nn_train
  # 0.9912975
  
  mt1 = as.matrix(test_nn)
  f = sigmoid((mt1[,1:N_col] %*% coeffi_v) + rep(b, nrow(mt1)))
  y_pred = ifelse((f > 0.5),1,0)
  sum(y_pred)
  accuracy_nn_test = sum(y_pred == test_nn$bs_label)/nrow(mt1)
  accuracy_nn_test
  # 0.924
  cat("h=",h,"  accuracy_nn_test=",accuracy_nn_test,"\n")
  accuracy_nn_test_v = c(accuracy_nn_test_v, accuracy_nn_test)
}
accuracy_nn_test_v
# 0.8987342 0.9240506 0.9208861
mean(accuracy_nn_test_v)
# 0.914557
# 0.9113924 6-8-2017 9:14pm

# --------------------------------------------------------------------------------------------------------
#
#  (5) Logistic regression for 'associate-needed' label
#
# --------------------------------------------------------------------------------------------------------
jd = read.csv("jd.csv")
nrow(jd)
jd$tags = as.character(jd$tags)
jd$d = as.character(jd$d)
jd_as = jd
jd_as$d = sapply(jd_as$d, function(x) paste("$", x, "$"))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associates bachelors degree")[[1]], sep=' associatesBachelorsDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associates degree")[[1]], sep=' associatesDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "associate degree")[[1]], sep=' associateDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "university graduate")[[1]], sep=' universityGraduate '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college degree")[[1]], sep=' collegeDegree '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "two year college")[[1]], sep=' twoYearCollege '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college courses")[[1]], sep=' collegeCourses '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college certificate")[[1]], sep=' collegeCertificate '))
jd_as$d = sapply(jd_as$d, function(x) paste2(strsplit(x, "college certifications")[[1]], sep=' collegeCertifications '))

# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
jd$tags = sapply(jd$tags, function(x) strsplit(x, " ")[[1]])
jd$as = sapply(jd$tags, function(x) ifelse(("associate-needed" %in% x),TRUE,FALSE))
for (k in 1:1580) {
  if (jd$as[k] == TRUE) {
    jd_as$tags[k] = "associate-needed"
  } else {
    jd_as$tags[k] = ""
  }
}
as_key_words = c('associatesBachelorsDegree', 'associatesDegree', 'associateDegree', 'universityGraduate', 'collegeCourses',
                 'twoYearCollege', 'collegeCertificate', 'collegeCertifications', 'collegeDegree');

expand_context = function(b,pad) {
  b2 = c(b[1])
  blen = length(b)
  for (j in 2:blen) {
    if (!is.null(b[j])) {
      f = b[j]
      if (f == FALSE) {
        for (t in 1:pad) {
          if (j > t && b[j-t] == TRUE) f = TRUE
          if (j <= blen-t && b[j+t] == TRUE) f = TRUE
        }
      }
      b2 = c(b2,f)
    } else {
      b2 = c(b2,FALSE)
    }
  }
  return(b2)
}
# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
for (k in 1:nrow(jd_as)) {
  da = strsplit(jd_as$d[k], " ")[[1]]
  if (da[1] %in% as_key_words) {
    b = c(TRUE)
  } else {
    b = c(FALSE)
  }
  for (i in 2:length(da)) {
    x = da[i]
    if (x %in% as_key_words) {
      b = c(b,TRUE)
    } else {
      b = c(b,FALSE)
    }
  }
  if (sum(b) == 0) {
    jd_as$d[k] = "-"
  } else {
    b2 = expand_context(b, 2)
    jd_as$d[k] = paste2(da[b2], sep=" ")
  }
}
# Remove extra white space, tab, or new line
jd_as$d = sapply(jd_as$d, function(x) gsub("[ \t\n]+"," ",x))
write.csv(jd_as, file="jd_as.csv", row.names=FALSE)

jd_as = read.csv("jd_as.csv")
jd_as$tags = as.character(jd_as$tags)
jd_as$d = as.character(jd_as$d)
text_corpus = Corpus(VectorSource(jd_as$d))
head(text_corpus)
text_term_matrix = DocumentTermMatrix(text_corpus)
dim(text_term_matrix)
# 1580 x 161

# NOTE BE CAREFUL DOING THIS WITH LARGER DATA SETS!!!!!!
text_corpus_mat = as.matrix(text_term_matrix)
dim(text_corpus_mat)
# 1580 x 161

# Convert to Data Frame
text_frame = as.data.frame(text_corpus_mat)
tf_num = text_frame
tf_num$tags = jd$as
str(tf_num$tags)
data_matrix = model.matrix(tags ~ .,data = tf_num)
# Calculate the principle components:
pc_data_matrix = prcomp(data_matrix)

# Look at magnitude of the variances explained (These are the eigenvalues!)
plot(pc_data_matrix$sdev)
dim(pc_data_matrix$x)

# How many components?  Let's see the accuracy by # of components
aic_by_num_pc = sapply(2:ncol(pc_data_matrix$x), function(x){
  formula_rhs_temp = paste(paste0('pc_data_matrix$x[,',1:x,']'), collapse = ' + ')
  formula_temp = paste('tf_num$tags ~',formula_rhs_temp)
  pc_x_components_temp = glm(eval(parse(text=formula_temp)), family="binomial")
  return(AIC(pc_x_components_temp))
})
plot(aic_by_num_pc, type='l', lwd=2,
     main='AIC of of P.C. of glm with X components',
     xlab="# of components", ylab='AIC')
num_pc_lead_to_min_aic = which.min(aic_by_num_pc)
min_aic_by_num_pc = min(aic_by_num_pc)
num_pc_lead_to_min_aic
# [1] 28
min_aic_by_num_pc
# [1] 409.6226

train_ind = sample(1:nrow(tf_num), round(0.8*nrow(tf_num)))
train_num = tf_num[train_ind,]
test_num  = tf_num[-train_ind,]

glm_model = glm(tags ~., data=train_num, family="binomial")
test_glm_pred = predict(glm_model, newdata=test_num)
test_glm_pred = ifelse((test_glm_pred > 0.5),1,0)
result_glm = (test_glm_pred == test_num$tags)
accuracy_glm = sum(result_glm)/nrow(test_num)
accuracy_glm
# 0.962
# 0.9556962  6-8-2017 9:19pm
# 0.9462025  6-9-2017 1:04pm
# 0.9588608  6-9-2017 1:27pm