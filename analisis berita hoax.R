library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Sarri mulai melakukan negosiasi transfer ke Chelsea.', 'Sarri disebut akan gantikan antonio conte.', 'Wolverhampton Wanderers aktif pada bursa transfer musim panas ini.',
          'Wolverhampton Wanderers dikabarkan akan mendatangkan john terry.', 'AS Roma dikabarkan akan kehilangan pemain kunci mereka.', 'AS Roma sedang dikaitkan dengan daniie  peroti.',
          'Ada beberapa pelatih yang sedang di incar everton.', 'Dikabarkan Alegri akan menjadi manajer everton.', 'RB Leipzig tim yang digadang akan menjadi kambing hitam.',
          'Rumor kepindahan bastian swens ke RB Leipzig masih ramai.', 'Giorgio Chiellini diberitakan sedang ada masalah dengan sang manajer.', 'Rumor tentang masa depan Giorgio Chiellini semakin ramai.',
          'Takashi Inui sedang dikaitkan dengan beberapa tim elite eropa.', 'Takashi Inui yang kemungkinan akan bergabung dengan real sociedad.', 'Jam main yang minim membuat Alaba membuka peluang tuk hengkang.',
          'Real madrid yang tertarik dengan jasa Alaba.', 'Liverpol sedang mengincar klien mbape.', 'Santer tentang rumor transfer neymar ke liverpol.', 'Messi mulai tidak betah berada di Barcelona.',
          'Barcelona akan membandrol Messi dengan harga 3,1 triliun.')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Sarri telah menginformasikan bahwa negosiasi transfer ke Chelsea tidak berlanjut lagi.', 'Wolverhampton Wanderers merogoh kocek hingga £12,5 juta untuk mempermanenkan penyerang Benik Afobe di musim panas ini.',
           'AS Roma meresmikan pembelian kedua mereka pada bursa musim panas ini dalam diri Ivan Marcano.', 'Marco Silva bakal menjadi manajer everton untuk musim kompetisi 2018/19 esok.',
           'RB Leipzig mengumumkan nama Nordi Mukiele sebagai rekrutan pertamanya di musim panas kali ini.', 'Giorgio Chiellini sudah sepakat dan meneken kontrak baru yang membuatnya bertahan di Juventus hingga Juni 2020.',
           'Real Betis resmi mengumumkan perekrutan bintang internasional Jepang, Takashi Inui dari Eibar dengan ikatan kontrak berdurasi tiga tahun.', 'Alaba berani menjamin bahwa ia masih akan membela Bayern pada musim depan.',
           'Liverpool mengumumkan perekrutan Fabinho dari Monaco dengan banderol sebesar €45 juta.', 'Lionel Messi menegaskan loyalitasnya dengan menegaskan Barcelona akan menjadi satu-satunya klub yang dibelanya di Eropa.')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)