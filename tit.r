library(dplyr)
library(ggplot2)
library(stringr)

setwd("./desktop/titanic.dataset")
tit <- read.csv("train.csv")

tit$Title <-gsub('(.*, )|(\\..*)',"",x = tit$Name) 


tit$Title[tit$Title=="Mlle"] <- "Miss"
tit$Title[tit$Title=="Mme"] <- "Miss"
tit$Title[tit$Title=="Ms"] <- "Mrs"
tit$Title[tit$Title %in% rare_title] <- "Rare Title"
table(tit$Sex,tit$Title)

tit$Fsize <- tit$SibSp+tit$Parch+1

ggplot(tit,aes(x=Fsize,fill=factor(Survived)))+geom_histogram(position = "dodge")+
    scale_x_continuous(breaks = c(1:13))+
    xlab("family size")

tit$Fsize2 [tit$Fsize==1] <- "singleton"
tit$Fsize2 [tit$Fsize<5 & tit$Fsize>1] <- "small"
tit$Fsize2 [tit$Fsize>=5] <- "large"

ggplot(tit,aes(x=Fsize2,fill=factor(Survived)))+geom_bar(position = "dodge")+
    xlab("family size")


tit$Deck <- as.factor(sapply(as.character(tit$Cabin),function(x) strsplit(x,NULL)[[1]][1]))


embark_fare <- tit %>%
    filter(PassengerId != 62 & PassengerId != 830)
ggplot(embark_fare,aes(x=Embarked,y=Fare,fill=factor(Pclass)))+
    geom_boxplot()+
    geom_hline(yintercept =80,color="red",lwd=1.5)
    

tit$Embarked [c(62,830)] <- "C"


names(tit)
factor_var <- c('PassengerId','Pclass','Sex','Embarked','Title','Fsize','Fsize2')

tit[factor_var] <-  tit[factor_var]%>% lapply(function(x) as.factor(x))
set.seed(129)
require(mice)
require(randomForest)
mice_mod <- mice(tit[, !names(tit) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)

tit$Age <- mice_output$Age

ggplot(tit,aes(Age,fill=factor(Survived)))+
    geom_histogram()+
    facet_grid(.~Sex)

tit$Child[tit$Age>18] <- "child"
tit$Child[tit$Age<=18] <- "adult"

table(tit$Survived,tit$Child)


# Set a random seed
set.seed(754)


tit1 <- tit [,-16]
tit1$Child <-as.factor(tit1$Child) 

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + Title + Fsize2 + Child ,data = tit1)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

importance <- importance(rf_model)
var_importance <- data.frame(Variables=row.names(importance(rf_model)),Importance=round(importance[ ,'MeanDecreaseGini'],2) )

ggplot(var_importance,aes(x=reorder(Variables,Importance),y=Importance,fill=Importance))+
           geom_bar(stat = "identity")+
           coord_flip()


