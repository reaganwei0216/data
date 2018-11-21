nb <- 10
nodes <- data.frame(id = 1:nb, label = paste("Label", 1:nb),
                    group = sample(LETTERS[1:3], nb, replace = TRUE), value = 1:nb,
                    title = paste0("<p>", 1:nb,"<br>Tooltip !</p>"), stringsAsFactors = FALSE)

edges <- data.frame(from = trunc(runif(nb)*(nb-1))+1,
                    to = trunc(runif(nb)*(nb-1))+1,
                    value = rnorm(nb, 10), label = paste("Edge", 1:nb),
                    title = paste0("<p>", 1:nb,"<br>Edge Tooltip !</p>"))



load("D:/0大數據申請資料/GSCNF0WB.rda")
load("D:/0大數據申請資料/GSCNF2WB.rda")
load("D:/0大數據申請資料/GSCNF6WB.rda")
load("D:/0大數據申請資料/GAPLF0.rda")
dim(GAPLF0)
# https://stackoverflow.com/questions/25925556/gather-multiple-sets-of-columns
# GSCNF6WBx <- subset(GSCNF6WB, select = c(QueryNo, ShopBan1, ShopBan2,ShopBan3,ShopPcent1,ShopPcent2,ShopPcent3))
# test <- head(GSCNF6WBx)
# test

# library(tidyr)
# test %>%
#   gather(key, value, -QueryNo) %>%
#   extract(key, c("question", "loop_number"), "(Shop[A-Za-z]+)([0-9]+)") %>%
#   spread(question, value)

# library(tidyr)
# test2 <- gather(test, condition, value, ShopBan1:ShopBan3)
# test2[order(test2$QueryNo),]
# ?gather

# head(GSCNF6WB)
# 
# subset(GSCNF6WB, QueryNo=="100455843",select = c(QueryNo, ShopBan1, ShopBan2,ShopBan3,ShopPcent1,ShopPcent2,ShopPcent3,ShopCompany1,ShopCompany2,ShopCompany3))
# 

# table(GSCNF0WB$IdNo=="79988141")
# head(GSCNF2WB)
# 
# GSCNF6WB$QueryNo=="79988141"
# table(GSCNF6WB$QueryNo=="79988141")

GSCNF6WBy <- subset(GSCNF6WB, select = c(QueryNo, ShopBan1, ShopBan2,ShopBan3,ShopPcent1,ShopPcent2,ShopPcent3,ShopCompany1,ShopCompany2,ShopCompany3))

library(dplyr)
library(tidyr)
# ??gather

testz <- GSCNF6WBy %>%
  gather(key, value, -QueryNo) %>%
  extract(key, c("question", "loop_number"), "(Shop[A-Za-z]+)([0-9]+)") %>%
  spread(question, value)

dim(testz)
head(testz,10)

# testz <- testz[complete.cases(testz), ]
# head(testz[complete.cases(testz), ],10)


#wait-----
#GSCNF0WB 間接保證額度申請書主檔
# GSCNF0WB.r1 <- subset(GSCNF0WB,DocNo > 0,select =c(QueryNo,IdNo))
GSCNF0WB.r1 <- subset(GSCNF0WB,DocNo > 0,select =c(QueryNo,IdNo))
head(GSCNF0WB.r1)
dim(GSCNF0WB.r1)
# str(GSCNF0WB$DocNo)
str(GSCNF0WB.r1)

dim(testz)
dim(tab)
table(tab$QueryNo)
#跟間接保證額度申請書主檔GSCNF0WB  串
tab <- merge(x = GSCNF0WB.r1, y = testz, by = "QueryNo", all.y = TRUE)
dim(GSCNF0WB.r1)
dim(GSCNF6WB.r1)
dim(tab)
head(tab)
dim(tab[complete.cases(tab), ])
tab <- tab[complete.cases(tab), ]
# 
# tab <- na.omit(tab)
# str(tab)

# tab[tab$IdNo==54040887,]

library(stringr)
#trim----
trim <- function( x ) {
  gsub("(^[[:space:]]+|[[:space:]]+$)","", x)
}
tab$ShopBan <- trim(tab$ShopBan)
head(tab)

# str(tab)
# table(tab$ShopBan=='')

index <- tab$ShopBan!=''
tab <- tab[index,]
dim(tab) #48793 
str(tab)
head(tab)
class(tab$ShopBan)
tail(tab$ShopBan)
tab <- tab[grepl(pattern = "^[0-9]", tab$ShopBan),]
tab$ShopBan <- as.numeric(tab$ShopBan)
summary(tab$ShopBan)

head(tab)
is.na(tab[ , 4])
!is.na(tab[ , 4])
table(is.na(tab[ , 4]))
index9 <-is.na(tab[ , 4])
index8 <-!is.na(tab[ , 4])

tab[index9,] 
dim(tab[index9,] )
dim(tab)
tab <- tab[complete.cases(tab), ]
head(tab)


#GAPLF0受保戶檔
GAPLF0.r1 <- subset(GAPLF0,select =c(Ban,Company))
str(GAPLF0.r1)
tail(GAPLF0.r1)
GAPLF0.r1$Ban <- as.character(GAPLF0.r1$Ban)
dim(GAPLF0.r1)
GAPLF0.r1 <- GAPLF0.r1[grepl(pattern = "^[0-9]", GAPLF0.r1$Ban),]
GAPLF0.r1$Ban <- as.character(GAPLF0.r1$Ban)
GAPLF0.r1$Ban <- as.numeric(GAPLF0.r1$Ban)
tail(GAPLF0.r1)
str(GAPLF0.r1)



# nodes88<- subset(nodes99, id %in% edges98$form | id %in% edges98$to ,select =c(id))  
# tab <- subset(tab,ShopBan %in% GAPLF0.r1$Ban )
# dim(tab)
# head(tab)
str(tab)
# tab2$ShopBan <- as.numeric(tab2$ShopBan)

#select----
# head(tab)
# head(tab)
# tab[tab$IdNo==89650890,]

tab2 <- tab
head(tab)
tab2 <- tab2[tab2$IdNo==16501819,]
tab2
head(tab2)
edge <- subset(tab2,select = c(IdNo,ShopBan,ShopPcent))
edge
dim(edge)
colnames(edge) <- c('from','to','value')
head(edge)
# str(edge)
# length(edge$from)
# length(edge$to)

dim(edge)
node_all <- c(edge$from,edge$to)
node_all2 <- unique(node_all)%>% as.data.frame()
colnames(node_all2) <- 'id'
node_all2$label <-node_all2$id 
# str(node_all2)
library(visNetwork)

visNetwork(node_all2, edge, main = "Title", submain = "Subtitle") %>%
  visExport() %>%
  visOptions(highlightNearest = TRUE,nodesIdSelection = list(enabled = TRUE)) %>%
  visEdges(arrows = "to") %>%
  visLegend()


visNetwork(node_all2, edge, width = "100%") %>% 
  visEdges(arrows = "from") %>% 
  visHierarchicalLayout() 

tab$IdNo
str(tab$IdNo)
tab[tab$IdNo==81483738,]
GSCNF6WB[GSCNF6WB$QueryNo==105388184,]
dim(GSCNF6WB[GSCNF6WB$QueryNo==105388184,])
str(GSCNF6WB$QueryNo)

#old------

rm(list=ls())
library("visNetwork")
load("D:/0大數據申請資料/10612044-1mdb.RData")
load("D:/0大數據申請資料/10612044-2mdb.RData")
load("D:/0大數據申請資料/10612044-3mdb.RData")
load("D:/0大數據申請資料/10612044-4mdb.RData")

load("D:/0大數據申請資料/GSCNF0WB.rda")
load("D:/0大數據申請資料/GSCNF2WB.rda")
load("D:/0大數據申請資料/GSCNF6WB.rda")
load("D:/0大數據申請資料/GAPLF0.rda")


#GSCNF0WB間接保證額度申請書主檔-----

GSCNF0WB.r1 <- subset(GSCNF0WB,DocNo > 0,select =c(QueryNo,IdNo))
dim(GSCNF0WB)
dim(GSCNF0WB.r1)
# str(GSCNF0WB$DocNo)
str(GSCNF0WB.r1)

#GSCNF2WB.revx企業基本資料----
GSCNF2WB.r1<- subset(GSCNF2WB,select =c(QueryNo,Company))
head(GSCNF2WB.r1)


#GAPLF0受保戶檔----
head(GAPLF0)
GAPLF0.r1 <- subset(GAPLF0,select =c(Ban,Company))
dim(GAPLF0.r1)
tail(GAPLF0.r1)
GAPLF0.r1$Ban <- as.character(GAPLF0.r1$Ban)
dim(GAPLF0.r1)
GAPLF0.r1 <- GAPLF0.r1[grepl(pattern = "^[0-9]", GAPLF0.r1$Ban),]
tail(GAPLF0.r1)
str(GAPLF0.r1)
# GAPLF0.r1$Ban <- as.numeric(GAPLF0.r1$Ban)

#GSCNF6WB.revx財務、進銷貨----
# GSCNF6WB.revx <- subset(GSCNF6WB,select =c(QueryNo,ShopBan1,ShopBan2,ShopBan3))
GSCNF6WB.r1 <- subset(GSCNF6WB,select =c(QueryNo,ShopBan1))
str(GSCNF6WB.r1)
head(GSCNF6WB.r1,10)
str(GSCNF0WB.r1)

tab <- merge(x = GSCNF0WB.r1, y = GSCNF6WB.r1, by = "QueryNo", all.x = TRUE)
dim(GSCNF0WB.r1)
dim(GSCNF6WB.r1)
dim(tab)
tail(tab)
tab <- na.omit(tab)
tab$ShopBan1 <- as.character(tab$ShopBan1)

str(tab)
library(stringr)
#trim----
trim <- function( x ) {
  gsub("(^[[:space:]]+|[[:space:]]+$)","", x)
}
tab$ShopBan1 <- trim(tab$ShopBan1)
tab$ShopBan1 <- as.character(tab$ShopBan1)

str(tab)
# tab$ShopBan1==''
index <- tab$ShopBan1!=''
table(index)
tab <- tab[index,]
dim(tab) #48793 
str(tab)
class(GAPLF0.r1$Ban)
class(tab$ShopBan1)
class(GAPLF0.r1$Ban)
# nodes88<- subset(nodes99, id %in% edges98$form | id %in% edges98$to ,select =c(id))  
tab <- subset(tab,ShopBan1 %in% GAPLF0.r1$Ban )
str(tab)
tail(tab)
dim(tab)
tab$ShopBan1 <- as.numeric(tab$ShopBan1)




#edge------
head(tab)
edge <- subset(tab,select = c(IdNo,ShopBan1))
str(edge)
colnames(edge) <- c('from','to')
head(edge)


#tab2-----
tab2 <- as.data.frame(table(edge$to))
dim(tab2)
tab2 <- tab2[tab2$Freq %in% c(10), ]
dim(tab2[tab2$Freq %in% c(10), ])
# table(tab2$Freq %in% c(10))
tab2[tab2$Freq %in% c(10),1]
head(tab2)
edge2 <- subset(edge,to %in% tab2$Var1)
dim(edge2)
head(edge2)



#node2------

head(tab)
node1 <- subset(tab,select = c(IdNo))
class(node1)
node2 <- node1[!duplicated(node1),] %>% as.data.frame()
head(node2)
colnames(node2) <- 'id'
dim(node2)

#test----
str(node2)
node.list <- as.numeric(c(edge2$from,edge2$to))
node3 <- subset(node2,id %in% node.list)
node3$label <-node3$id 
# str(node3)
class(c(edge2$from,edge2$to)) 
dim(node3)
# visNetwork(node2, edge, main = "Title", submain = "Subtitle") %>%
#   visExport() %>%
#   visOptions(highlightNearest = TRUE,nodesIdSelection = list(enabled = TRUE)) %>%
#   visEdges(arrows = "to") %>%
#   visLegend()

#run-----

visNetwork(node2, edge2, main = "Title", submain = "Subtitle") %>%
  visExport() %>%
  visOptions(highlightNearest = TRUE,nodesIdSelection = list(enabled = TRUE)) %>%
  visEdges(arrows = "to") %>%
  visLegend()

network10 <-visNetwork(node3, edge2, main = "GSCNF6WB", submain = "ShopCompany1") %>%
  visExport() %>%
  visOptions(highlightNearest = TRUE,nodesIdSelection = list(enabled = TRUE)) %>%
  visEdges(arrows = "to") %>%
  visInteraction(navigationButtons = TRUE) %>%
  visLegend()

network <- visNetwork(node2, edge, main = "Title", submain = "Subtitle") %>%
  visExport() %>%
  visOptions(highlightNearest = TRUE,nodesIdSelection = list(enabled = TRUE)) %>%
  visEdges(arrows = "to") %>%
  visLegend()

visSave(network, file = "network.html")
visSave(network10, file = "network10.html")


# test <- data.frame(
#   x1 = c(1,2,3,4,5,1,3,5),
#   x2 = c("a","b","c","d","e","a","b","e"),
#   x3 = c("a","b","c","d","e","a","c","e"))
# test
# 
# test[!duplicated(test),]
