
# source("../common_profile.R", local = TRUE)
# source("/Users/yutasuzuki/Desktop/Github/common_profile.R", local = TRUE)
source("/Volumes/Burger/project/AttentinalBlink/common_profile.R", local = TRUE)

# #### Package installation #### 
# library(easypackages)
# 
# liblist <- c("rjson","ggplot2","ggpubr","Cairo","gridExtra","effsize",
#              "BayesFactor","reshape",
#              # "lme4",
#              "lavaan","semPlot","semptools",
#              "RColorBrewer","ggbeeswarm",
#              "permutes","quickpsy",
#              "psycho","colorspace","knitr","kableExtra","openxlsx",
#              "cowplot","fitdistrplus","reshape2","tidyr","tidyverse","ggsignif","ggnewscale",
#              "lmerTest","modelsummary","effectsize","BayesFactor","purrr")
# # 
# # pkg_list <- read.csv("loaded_packages_versions.csv", stringsAsFactors = FALSE)
# # 
# # for (i in 1:nrow(pkg_list)) {
# #   if (!pkg_list[i,"Package"] %in% installed.packages()[,"Package"]){
# #     install.packages(pkg_list[i,"Package"])
# #   }
# # }
# 
# libraries(liblist)
# 
# if(exists(".anovakun.env")){
#   # sys.source("./data/anovakun_485.R", envir = .anovakun.env)
#   sys.source("./data/anovakun_487.R", envir = .anovakun.env)
#   # sys.source("./data/anovakun_486.R", envir = .anovakun.env)
# }else{
#   .anovakun.env <- new.env()
#   # sys.source("./data/anovakun_485.R", envir = .anovakun.env)
#   sys.source("./data/anovakun_487.R", envir = .anovakun.env)
#   # sys.source("./data/anovakun_486.R", envir = .anovakun.env)
#   attach(.anovakun.env)
# }
# 
# subName = NULL
# for( i in seq(100)){ 
#   if(i<10){subName = rbind(subName,paste0("s0", i, sep = ""))} 
#   else{subName = rbind(subName,paste0("s", i, sep = ""))}
# }
# 
# figNumbering = c("A","B","C","D","E","F","G","H","I","J","K","L")
# 
# #fontFamily = "Source Han Sans JP Light"
# fontFamily = "Times"
# 
# CairoFonts(regular = "Times New Roman",bold="Times New Roman Bold")
# SIZE_FONT = 14
# 
# date = today <- Sys.Date()
# date = gsub("-", "", date)
# 
# pairedttest <- function(x,y=NULL){
#   
#   if(is.null(y)){
#     if(length(x) == 1){
#       bf <- t.test(c(0,0), mu = 0)
#     }else{
#       bf = t.test(x, mu = 0)
#     }    
#   } else{
#     if(length(x) == 1){
#       bf <- t.test(c(0,0),c(0,0),var.equal=T,paired=T)
#     }else{
#       bf = t.test(x,y,var.equal=T,paired=T)
#     }
#   }
#   return(bf)
# }
# 
# 
# load_json_dataframe <- function(folder, pattern) {
#   # file_path <- list.files(folder, pattern = pattern, full.names = TRUE)[1]
#   # json_data <- fromJSON(file = file_path)
#   # ,nullValue=NA)
#   # json_data <- lapply(json_data, unlist)
#   # json_data <- rapply(
#   #   json_data,
#   #   function(x) if (is.null(x)) NA else x,
#   #   how = "replace"
#   # )
#   # as.data.frame(json_data)
# 
#   file_path <- list.files(folder, pattern = pattern, full.names = TRUE)[1]
#   txt <- readLines(file_path, warn = FALSE)
#   txt <- gsub("\\bNaN\\b", "null", txt)
#   df_list <- fromJSON(paste(txt, collapse = "\n"))
#   
#   all_names <- unique(unlist(lapply(df_list, names)))
#   
#   rows <- lapply(df_list, function(rec) {
#     vals <- lapply(all_names, function(nm) {
#       v <- rec[[nm]]
#       if (is.null(v) || length(v) == 0) {NA} else {v}
#     })
#     names(vals) <- all_names
#     t(as.data.frame(as.list(vals), stringsAsFactors = FALSE))
#   })
#   
#   df = as.data.frame(rows)
# 
# }
# 
# setEmptyStyle <- function(gData,config,size_font=25){
#   gData <- gData +theme(
#     panel.border = element_blank(), 
#     axis.ticks.length=unit(1, "cm"),
#     axis.ticks.x = element_line(colour = "black",size = 0.5),
#     # axis.line = element_line(),
#     axis.text.x = element_text(colour="black"),
#     axis.text.y = element_text(colour="black"),
#     panel.background = element_rect(fill = "transparent",size = 0.5),
#     panel.grid.major = element_line(colour = NA),
#     panel.grid.major.y = element_line(colour = "gray", size = 0.05),
#     panel.grid.major.x = element_line(colour = NA),
#     panel.grid.minor = element_line(colour = NA),
#     axis.ticks = element_line(colour = "black",size = 0.5),
#     # text = element_text(size = size_font,family = "Times"),
#     text = element_text(size = size_font,family = fontFamily),
#     legend.title = element_text(size=(size_font)),
#     legend.text = element_text(size=(size_font)),
#     legend.key=element_rect(colour="transparent", fill=NA),
#     plot.background=element_rect(fill="transparent", colour=NA),
#     legend.background=element_rect(fill="transparent", colour=NA),
#     plot.title = element_text(size = size_font,hjust=-0.2)
#   )
#   
#   
#   config$xlim_stride = (config$xlim[2] - config$xlim[1]) / 2
#   config$ylim_stride = (config$ylim[2] - config$ylim[1]) / 2
#   
#   # print(config$ylim_stride)
#   # print(config$xlim_stride)
#   
#   if(length(config$ylim) > 1){
#     gData = gData +
#       scale_y_continuous(breaks=config$ylim)+
#       coord_cartesian(ylim=c(config$ylim[1]-config$ylim_stride,rev(config$ylim)[1]+config$ylim_stride),expand=FALSE)+
#       annotate(x=config$xlim[1]-config$xlim_stride, xend=config$xlim[1]-config$xlim_stride,
#                y=config$ylim[1], yend=rev(config$ylim)[1],
#                colour="black", lwd=0.5, geom="segment")+
#       theme(
#         axis.ticks = element_line(colour = "black",size = 0.5)
#       )
#   }
#   
#   if(length(config$xlim) > 1){
#     gData = gData +
#       # scale_x_continuous(breaks=config$xlim)+
#       coord_cartesian(xlim=c(config$xlim[1]-config$xlim_stride,rev(config$xlim)[1]+config$xlim_stride),expand=FALSE) +
#       annotate(x=config$xlim[1],xend=rev(config$xlim)[1],
#                y=config$ylim[1]-config$ylim_stride, yend=config$ylim[1]-config$ylim_stride,
#                colour="black", lwd=0.5, geom="segment")
#   }
#   
#   
#   return(gData)
# }
# 
# output2wayANOVA <- function(forDrawingSigANOVA) {
#   fVal <- c("A","B","A x B")
#   for (i in 1:3) {
#     
#     befSt <- paste0("F(",round(forDrawingSigANOVA[(i-1)*2+2,]$df.col, digits = 3),
#                     ", ",round(forDrawingSigANOVA[i*2+1,]$df.col, digits = 3)
#                     ,") = ",signif(forDrawingSigANOVA[i*2,]$f.col,digits = 4),",", sep = '')
#     if(forDrawingSigANOVA[i*2,]$p.col > 0.001){
#       if(forDrawingSigANOVA[i*2,]$p.col < 0.05){
#         pVal<- paste0("p = ",signif(forDrawingSigANOVA[i*2,]$p.col,digits = 3),"*,", sep = '')
#       }else{
#         pVal<- paste0("p = ",signif(forDrawingSigANOVA[i*2,]$p.col,digits = 3),",", sep = '')
#       }
#     }else{
#       pVal<- paste0("p < 0.001,", sep = '')
#     }
#     etaVal<- paste0("η2p =",signif(forDrawingSigANOVA[i*2,]$`p.eta^2`,digits = 3))
#     cat(forDrawingSigANOVA[i*2+1,]$source.col,": \n")
#     cat( befSt, pVal, etaVal,"\n")
#   }
# }
# 
# output1wayANOVA <- function(forDrawingSigANOVA) {
#   fVal <- c("A")
#   befSt <- paste0("F(",forDrawingSigANOVA[1,]$df.col,",",round(forDrawingSigANOVA[2,]$df.col, digits = 3),") = ", signif(forDrawingSigANOVA[2,]$f.col,digits = 4),",", sep = '')
#   if( forDrawingSigANOVA[2,]$p.col > 0.001){
#     if( forDrawingSigANOVA[2,]$p.col <0.05){
#       pVal<- paste0("p = ",signif(forDrawingSigANOVA[2,]$p.col,digits = 3),"*,", sep = '')
#     }else{
#       pVal<- paste0("p = ",signif(forDrawingSigANOVA[2,]$p.col,digits = 3),",", sep = '')
#     }  
#   }else{
#     pVal<- paste0("p < 0.001,", sep = '')
#   }
#   etaVal<- paste("η2p =",signif(forDrawingSigANOVA[2,]$`p.eta^2`,digits = 3))
#   
#   cat(forDrawingSigANOVA[2,]$source.col,": \n")
#   cat( befSt, pVal, etaVal,"\n")
# }
# 
# # 
# # combineGraphs <- function(graphNum,p,layout){
# #   
# #   titleStr = c("'A'", "'B'", "'C'", "'D'", "'E'", "'F'", "'G'")
# #   st = paste(p,graphNum, sep = "", collapse=",")
# #   labelSt = titleStr[seq(1,length(graphNum))]
# #   labelSt = paste(labelSt, collapse=",")
# #   
# #   ncolNum = round(length(graphNum) / 2 )
# #   
# #   if (is.numeric(layout)){
# #     eval(parse(text=paste0("p = grid.arrange(", st ,",layout_matrix = layout)")))
# #     
# #   }else{
# #     eval(parse(text=paste0("p = ggarrange(",
# #                            st ,",labels = c(",
# #                            labelSt,
# #                            "),font.label = list(size = 20),ncol = 2, nrow =", ncolNum, ")")))
# #   }
# #   return(p)
# # }
# 
# combineGraphs2 <- function(mmName,p_all,layout,aspects){
#   
#   titleStr = c("'A'", "'B'", "'C'", "'D'", "'E'", "'F'", "'G'")
#   
#   st = NULL
#   for(m in mmName){
#     st=paste0(st,paste0("p_all$", m, ","))
#   }
#   
#   labelSt = titleStr[seq(1,length(mmName))]
#   labelSt = paste(labelSt, collapse=",")
#   
#   aspects = paste(aspects, collapse=",")
#   
#   if(layout[1]==1){
#     eval(parse(text=paste0("p = ggarrange(",st ,"labels = c(",labelSt,
#                            "),font.label = list(size = 20),ncol =", layout[2], ",widths=c(",aspects,"))")))
#     
#   }else{
#     eval(parse(text=paste0("p = ggarrange(",st ,"labels = c(",labelSt,
#                            "),font.label = list(size = 20),nrow = ",layout[1], ",heights=c(",aspects,"))")))
#   } 
#   
#   return(p)
# }
# 
# 
# dispBarGraph <- function(ribbondata, config, data_y, factors,numOfSub = 0){
#   
#   eval(parse(text=paste0("ribbondata$data_y = ribbondata$", data_y)))
#   
#   if(numOfSub == 0) {numOfSub = length(unique(ribbondata$sub))}
#   
#   if(length(factors) == 1){
#     eval(parse(text=paste0("std_data = aggregate( data_y ~ ",factors[1], ", data = ribbondata, FUN = 'sd')")))
#     eval(parse(text=paste0("ribbondata = aggregate( data_y ~ ",factors[1], ", data = ribbondata, FUN = 'mean')")))
#   }
#   else if(length(factors) == 2){
#     eval(parse(text=paste0("std_data = aggregate( data_y ~ ",factors[1],"*",factors[2], ", data = ribbondata, FUN = 'sd')")))
#     eval(parse(text=paste0("ribbondata = aggregate( data_y ~ ",factors[1],"*",factors[2], ", data = ribbondata, FUN = 'mean')")))
#   }else{
#     eval(parse(text=paste0("std_data = aggregate( data_y ~ ",factors[1],"*",factors[2],"*",factors[3], ", data = ribbondata, FUN = 'sd')")))
#     eval(parse(text=paste0("ribbondata = aggregate( data_y ~ ",factors[1],"*",factors[2],"*",factors[3], ", data = ribbondata, FUN = 'mean')")))
#   }
#   
#   std_data$data_y = std_data$data_y / sqrt(numOfSub)
#   
#   ribbondata$SE_min <- ribbondata$data_y - std_data$data_y
#   ribbondata$SE_max <- ribbondata$data_y + std_data$data_y
#   
#   if(length(factors) == 1){ 
#     eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],
#                            ", y = data_y",
#                            # ", color =",factors[1] ,
#                            # ", fill =", factors[1] ,
#                            "))")))
#   } else if(length(factors) == 2){
#     # eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y, color=interaction(",factors[1],",",factors[2],"),fill = interaction(",factors[1],",",factors[2],")))")))
#     eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],
#                            ", y = data_y",
#                            # ",color = ",factors[2],
#                            ", fill = ",factors[2],
#                            # ",group = interaction(",factors[1],",",factors[2],")",
#                            "))")))
#   } else{
#     eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y, color = interaction(", factors[1],",",factors[2],",",factors[3],"),",
#                            "fill = interaction(", factors[1],",",factors[2],",",factors[3],")))")))
#   }
#   
#   p <- p + 
#     geom_bar(stat = "identity",
#              position = position_dodge(.4)
#              # position = "dodge"
#     )+
#     geom_errorbar(aes(ymin = SE_min, ymax = SE_max),
#                   width = 0.3, size=0.2, position = position_dodge(.4)) +
#     geom_hline(yintercept=0, colour="black", linetype="solid", size = 0.5) +
#     ggtitle(config$title) +
#     xlab(config$label_x) + ylab(config$label_y) +
#     theme(
#       axis.ticks.x = element_blank(),
#       # axis.text.x = element_text(angle = 30, hjust = 1),
#       axis.line.x = element_blank()
#     )
#   
#   if(!is.null(config$grCol)){
#     p=p+scale_fill_manual(values = config$grCol)+
#       scale_color_manual(values = config$gr_outline)
#   }
#   
#   return(p)
# }
# 
# 
# dispLineGraph <- function(ribbondata, config, data_y, factors, numOfSub = 0){
#   
#   eval(parse(text=paste0("ribbondata$data_y = ribbondata$", data_y)))
#   
#   if(numOfSub == 0) {numOfSub = length(unique(ribbondata$sub))}
#   
#   if(length(factors) == 1){ 
#     eval(parse(text=paste0("std_data = aggregate( data_y ~ ",factors[1], ", data = ribbondata, FUN = 'sd')")))
#     eval(parse(text=paste0("ribbondata = aggregate( data_y ~ ",factors[1], ", data = ribbondata, FUN = 'mean')")))
#   }
#   else if(length(factors) == 2){ 
#     eval(parse(text=paste0("std_data = aggregate( data_y ~ ",factors[1],"*",factors[2], ", data = ribbondata, FUN = 'sd')")))
#     eval(parse(text=paste0("ribbondata = aggregate( data_y ~ ",factors[1],"*",factors[2], ", data = ribbondata, FUN = 'mean')")))
#   }else{
#     eval(parse(text=paste0("std_data = aggregate( data_y ~ ",factors[1],"*",factors[2],"*",factors[3], ", data = ribbondata, FUN = 'sd')")))
#     eval(parse(text=paste0("ribbondata = aggregate( data_y ~ ",factors[1],"*",factors[2],"*",factors[3], ", data = ribbondata, FUN = 'mean')")))
#   }
#   
#   std_data$data_y = std_data$data_y / sqrt(numOfSub)
#   
#   ribbondata$SE_min <- ribbondata$data_y - std_data$data_y
#   ribbondata$SE_max <- ribbondata$data_y + std_data$data_y
#   # , color = ", factors[1],, color = ", factors[1],aes(shape = ", factors[2],"), 
#   if(length(factors) == 1){ 
#     eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],
#                            ", y = data_y,color = ", factors[1], ",group = ",factors[1], "))")))
#     eval(parse(text=paste0("p = p + geom_point(size = 3)")))
#   } else if(length(factors) == 2){
#     # , shape =  factors[2], 
#     if(!is.null(config$colFactor)){
#       eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y",
#                              ", group = ",factors[2], ",color = ",config$colFactor, "))")))
#     }else{
#       eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y",
#                              ", group = ",factors[2], ",color = ",factors[2], "))")))
#     }# ",group = interaction(",factors[1],",",factors[2],
#     
#     eval(parse(text=paste0("p = p + geom_point(size = 6, position = position_dodge(.1))")))
#     
#     if(config$line){
#       eval(parse(text=paste0("p = p + geom_line(position = position_dodge(.1))")))
#     }
#     
#   } else{
#     if(config$shape){
#       eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y, ",
#                              # "group = ", factors[2],
#                              "group = interaction(",factors[2],",",factors[3],')',
#                              # ",shape = interaction(",factors[2],",",factors[3],')',
#                              ", shape = ",factors[3],
#                              ", color = ",factors[2], "))")))
#       
#       eval(parse(text=paste0("p = p + scale_shape_manual(values = 0:length(unique(ribbondata$",factors[3],")))")))
#       
#     }else{
#       eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y, ",
#                              "group = ", factors[2],
#                              # ",color = interaction(",factors[2],",",factors[3],')',
#                              ", color = ",factors[2], "))")))
#     }
#     # eval(parse(text=paste0("p = p + geom_point(aes(shape = ", factors[2],"), size = 3)")))
#     eval(parse(text=paste0("p = p + geom_point(size = 4, position = position_dodge(.1))")))
#     
#     if(config$line){
#       eval(parse(text=paste0("p = p + geom_line(position = position_dodge(.1))")))
#     }
#     
#   }
#   
#   if(!is.null(config$grCol)){
#     p = p + scale_color_manual(values = config$grCol)
#   }
#   if(!is.null(config$title)){
#     p = p + ggtitle(config$title)
#   }
#   p = p +  
#     geom_errorbar(aes(ymin = SE_min, ymax = SE_max),size = 0.1, width = 0.1, position = position_dodge(.1),color="black")+ 
#     xlab(config$label_x) + ylab(config$label_y) +
#     theme(
#       axis.ticks.x = element_blank(),
#       axis.line.x = element_blank()
#     )
#   
#   return(p)
# }
# 
# makeSigPair <- function(forDrawingPost) {
#   
#   sigPairA = NULL
#   if(length(forDrawingPost[["A"]]) > 1) {
#     t = as.character(forDrawingPost[["A"]][["bontab"]][["significance"]])
#     p = forDrawingPost[["A"]][["bontab"]][["adj.p"]]
#     for(i in 1:length(t)){
#       t0 = strsplit(t[i], " ")
#       if(t0[[1]][4] == "*"){
#         sigPairA = rbind(sigPairA,t0[[1]][1:3])
#       }
#       
#       if(t0[[1]][2] == "="){
#         sigPairA = rbind(sigPairA,t0[[1]][1:3])
#       }
#     }
#     sigPairA = cbind(sigPairA,p)
#   }
#   
#   sigPairB = NULL
#   if(length(forDrawingPost[["B"]]) > 1) {
#     t = as.character(forDrawingPost[["B"]][["bontab"]][["significance"]])
#     for(i in 1:length(t)){
#       t0 = strsplit(t[i], " ")
#       if(t0[[1]][4] == "*"){
#         sigPairB = rbind(sigPairB,t0[[1]][1:3])
#       }
#     }
#   }
#   
#   # sigPairAB = NULL
#   # if(length(forDrawingPost[["A:B"]])>1){
#   #   t = as.character(forDrawingPost[["A:B"]][["simtab"]][["sig.col"]])
#   #   for(i in 1:length(t)){
#   #     t0 = strsplit(t[i], " ")
#   #     if(t0[[1]] == "*"){
#   #       sigPairAB = rbind(sigPairAB,t0[[1]][1:3])
#   #     }
#   #   }
#   # }
#   
#   return(rbind(sigPairA,sigPairB))
# }
# 
# makeSigPair2 <- function(forDrawingPost) {
#   
#   sigPairA = NULL
#   if(length(forDrawingPost[["A"]]) > 1) {
#     t = as.character(forDrawingPost[["A"]][["bontab"]][["significance"]])
#     p = forDrawingPost[["A"]][["bontab"]][["adj.p"]]
#     for(i in 1:length(t)){
#       t0 = strsplit(t[i], " ")
#       if(t0[[1]][4] == "*"){
#         sigPairA = rbind(sigPairA,t0[[1]][1:3])
#       }
#       
#       if(t0[[1]][2] == "="){
#         sigPairA = rbind(sigPairA,t0[[1]][1:3])
#       }
#     }
#     sigPairA = cbind(sigPairA,p)
#   }
#   
#   sigPairB = NULL
#   if(length(forDrawingPost[["B"]]) > 1) {
#     t = as.character(forDrawingPost[["B"]][["bontab"]][["significance"]])
#     for(i in 1:length(t)){
#       t0 = strsplit(t[i], " ")
#       if(t0[[1]][4] == "*"){
#         sigPairB = rbind(sigPairB,t0[[1]][1:3])
#       }
#     }
#   }
#   
#   # sigPairAB = NULL
#   # if(length(forDrawingPost[["A:B"]])>1){
#   #   t = as.character(forDrawingPost[["A:B"]][["simtab"]][["sig.col"]])
#   #   for(i in 1:length(t)){
#   #     t0 = strsplit(t[i], " ")
#   #     if(t0[[1]] == "*"){
#   #       sigPairAB = rbind(sigPairAB,t0[[1]][1:3])
#   #     }
#   #   }
#   # }
#   
#   return(rbind(sigPairA,sigPairB))
# }
# 
# drawSignificance <- function(p,sigPair,y_pos,range,nsFlag) {
#   if(!is.null(sigPair)){
#     for(i in 1:dim(sigPair)[1]){
#       if(sigPair[i,4] > 0.05){
#         if(nsFlag){
#           # p <- p + geom_signif(xmin=sigPair[i,1], xmax=sigPair[i,3],annotations="n.s.", y_position = y_pos+(i-1)*range,
#           #                      textsize = 8, size=0.2, tip_length = 0.00,family=fontFamily)
#           p <- p + geom_signif(xmin=sigPair[i,1], xmax=sigPair[i,3],annotations=paste("p=",round(as.numeric(sigPair[i,4]),4) ), y_position = y_pos+(i-1)*range,
#                                textsize = 5, size=0.2, tip_length = 0.00,family=fontFamily)
#           
#         }
#       }else{
#         if(sigPair[i,4] < 0.05 & sigPair[i,4] > 0.01){
#           p <- p + geom_signif(xmin=sigPair[i,1], xmax=sigPair[i,3],annotations="*", y_position = y_pos+(i-1)*range,
#                                textsize = 10, size=0.2, tip_length = 0.00,family=fontFamily)
#         }else if(sigPair[i,4] < 0.01 & sigPair[i,4] > 0.001){
#           p <- p + geom_signif(xmin=sigPair[i,1], xmax=sigPair[i,3],annotations="**", y_position = y_pos+(i-1)*range,
#                                textsize = 10, size=0.2, tip_length = 0.00,family=fontFamily)
#         }else{      
#           p <- p + geom_signif(xmin=sigPair[i,1], xmax=sigPair[i,3],annotations="***", y_position = y_pos+(i-1)*range,
#                                textsize = 10, size=0.2, tip_length = 0.00,family=fontFamily)
#         }
#       }
#     }
#   }
#   return(p)
# }
# 
# disp <- function(ribbondata, config, data_y, shadeFl, factors, numOfSub=0){
#   
#   eval(parse(text=paste0("ribbondata$data_y = ribbondata$", data_y)))
#   
#   if (shadeFl == 1) {
#     if(numOfSub == 0) {numOfSub = length(unique(ribbondata$sub))}
#     
#     if(length(factors) == 1){ 
#       eval(parse(text=paste0("data_std = aggregate( data_y ~ data_x * ",factors[1],", data = ribbondata, FUN = 'sd')"))) 
#       eval(parse(text=paste0("ribbondata = aggregate( data_y ~ data_x * ",factors[1],", data = ribbondata, FUN = 'mean')")))
#     }
#     else if(length(factors) == 2){ 
#       eval(parse(text=paste0(
#         "data_std = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],
#         ", data = ribbondata, FUN = 'sd')"))) 
#       eval(parse(text=paste0(
#         "ribbondata = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],
#         ", data = ribbondata, FUN = 'mean')")))
#     }else{
#       eval(parse(text=paste0(
#         "data_std = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],"*",factors[3],
#         ", data = ribbondata, FUN = 'sd')"))) 
#       eval(parse(text=paste0(
#         "ribbondata = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],"*",factors[3],
#         ", data = ribbondata, FUN = 'mean')")))
#     }
#     
#     data_std$data_y <- data_std$data_y / sqrt(numOfSub)
#     ribbondata$ymin <- ribbondata$data_y - data_std$data_y
#     ribbondata$ymax <- ribbondata$data_y + data_std$data_y
#     
#     if(length(factors) == 1){ 
#       eval(parse(text=paste0(
#         "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],", group = ",factors[1],")) +",
#         "geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = ",factors[1],", group = ",factors[1], "),
#       color = 'gray', fill = 'gray', alpha = config$alpha, size = 0.05)")))
#     }
#     else if(length(factors) == 2){ 
#       eval(parse(text=paste0(
#         "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],
#         # "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = interaction(",factors[1],",",factors[2],")",
#         ", group =  interaction(",factors[1],",",factors[2],")))+",
#         # ", group = ",factors[1],"))+",
#         # "annotation_raster(image, -Inf, Inf, -Inf, Inf) +",
#         "geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = ",factors[1],",),
#       color = 'gray', fill = 'gray', alpha = config$alpha, size = 0.05)")))
#       # "geom_line(aes(linetype=",factors[1],"))",
#       
#     }else{
#       eval(parse(text=paste0(
#         "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],
#         ", group = interaction(",factors[1],",",factors[2],",",factors[3],")))+",
#         "geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = ",factors[1],"),",
#         "color = 'gray', fill = 'gray', alpha = config$alpha, size = 0.05) +",
#         "geom_line(aes(linetype = ",factors[3],"))"
#       )))
#     }
#     
#     if(config$linetype){
#       eval(parse(text=paste0("p = p + ",
#                              "geom_line(aes(linetype = ",factors[1],"))"
#                              # "geom_line(aes(linetype = ",factors[2],"))"
#                              # "geom_line(aes(linetype= interaction(",factors[1],",",factors[2],")))"
#       )))
#     }else{
#       p = p + geom_line()
#     }
#     
#     if(!is.null(config$grCol)){
#       p <- p +
#         # scale_color_manual(values = config$grCol, name = factors[1])+
#         scale_fill_manual(values = config$grCol)+
#         scale_color_manual(values = config$grCol)
#     }
#     
#     p <- p +
#       ggtitle(config$title) +
#       xlab(config$label_x) + ylab(config$label_y) +
#       coord_cartesian(xlim = config$lim_x,ylim = config$lim_y) +
#       scale_x_continuous(expand = c(0, 0)) 
#     
#   }else{ 
#     
#     eval(parse(text=paste0(
#       "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],", group = ",factors[2],"))+", 
#       # "geom_line(aes(linetype = Type))",
#       "geom_line()")))
#     
#     if(!is.null(config$grCol)){
#       p <- p +
#         scale_color_manual(values = config$grCol)
#     }
#     p <- p + 
#       geom_vline(xintercept=0, colour='black', linetype='longdash', size = 0.1) +
#       ggtitle(config$title) +
#       xlab(config$label_x) + ylab(config$label_y) +
#       coord_cartesian(xlim=config$lim_x, ylim=config$lim_y) +
#       scale_x_continuous(expand = c(0, 0))
#     # scale_y_continuous(expand = c(0, 0.1))
#     # scale_y_continuous(breaks = seq(config$lim_y[1],config$lim_y[2],config$stride),expand = c(0, 0))
#     
#   }
#   
#   # p = setBarFigureStyle(p)
#   
#   return(p)
# }
# 
# # make2WayANOVAlines <-function(data_anova){
# #   anovakun(data_anova,"sAB",long=T, peta=T)
# #   
# #   fVal <- c("A","B","A x B")
# #   for (i in 1:3) {
# #     
# #     befSt <- paste0("F(",round(forDrawingSigANOVA[(i-1)*2+2,]$df.col, digits = 3),
# #                     ", ",round(forDrawingSigANOVA[i*2+1,]$df.col, digits = 3)
# #                     ,") = ",signif(forDrawingSigANOVA[i*2,]$f.col,digits = 4),",", sep = '')
# #     if(forDrawingSigANOVA[i*2,]$p.col > 0.001){
# #       if(forDrawingSigANOVA[i*2,]$p.col < 0.05){
# #         pVal<- paste0("p = ",signif(forDrawingSigANOVA[i*2,]$p.col,digits = 3),"*,", sep = '')
# #       }else{
# #         pVal<- paste0("p = ",signif(forDrawingSigANOVA[i*2,]$p.col,digits = 3),",", sep = '')
# #       }
# #     }else{
# #       pVal<- paste0("p < 0.001,", sep = '')
# #     }
# #     etaVal<- paste0("η2p =",signif(forDrawingSigANOVA[i*2,]$`p.eta^2`,digits = 3))
# #     cat(forDrawingSigANOVA[i*2+1,]$source.col,": \n")
# #     cat( befSt, pVal, etaVal,"\n")
# #   }
# # }
# 
# getTtestSummary <- function(x1,x2){
#   
#   datSummary = list()
#   datSummary$bf = ttestBF(x = x1, y = x2, paired=TRUE)
#   datSummary$ttest = t.test(x1, x2, var.equal=T, paired=TRUE)
#   datSummary$cohend = cohen.d(x1,x2,paired=TRUE, within=TRUE)
#   
#   if((datSummary$ttest$p.value > 0.05)&(datSummary$ttest$p.value <= 0.1)){
#     datSummary$sig = "+"
#   }else if((datSummary$ttest$p.value > 0.01)&(datSummary$ttest$p.value <= 0.05)){
#     datSummary$sig = "*"
#   }else if((datSummary$ttest$p.value > 0.005)&(datSummary$ttest$p.value <= 0.01)){
#     datSummary$sig = "**"
#   }else if(datSummary$ttest$p.value <= 0.005){
#     datSummary$sig = "**"
#   }else{
#     datSummary$sig = "n.s."
#   }
#   return(datSummary)
#   
# }
# 
# showAnovaSummary <- function(anovaData,bf_table,lineNum,factorNum){
#   
#   if(round(anovaData[lineNum,]['p.col'],3)==0){
#     pLines = "p < 0.001"
#   }else{
#     pLines = paste0("$p$ = ",round(anovaData[lineNum,]['p.col'],3))
#   }
#   
#   
#   if(factorNum==3){
#     
#     if(round(exp(bf_table@bayesFactor[["bf"]][4])/exp(bf_table@bayesFactor[["bf"]][3]),3)>100){
#       bfLines = "$BF_{10}$ > 100"
#     }else{
#       bfLines = paste0("$BF_{10}$ = ",round(exp(bf_table@bayesFactor[["bf"]][4])/exp(bf_table@bayesFactor[["bf"]][3]),3))
#     }
#     
#     paste0("$F$(", round(anovaData[lineNum,]['df.col'],3),",",
#            round(anovaData[lineNum+1,]['df.col'],3),") = ",
#            round(anovaData[lineNum,]['f.col'],3),
#            ", ", pLines,
#            ", $\\eta^2_p$ = ", round(anovaData[lineNum,]['p.eta^2'],3),
#            ", ",  bfLines)
#   }else{
#     
#     if(round(exp(bf_table@bayesFactor[["bf"]][factorNum]),3)>100){
#       bfLines = "$BF_{10}$ > 100"
#     }else{
#       bfLines = paste0("$BF_{10}$ = ",round(exp(bf_table@bayesFactor[["bf"]][factorNum]),3))
#     }
#     
#     paste0("$F$(", round(anovaData[lineNum,]['df.col'],3),",",
#            round(anovaData[lineNum+1,]['df.col'],3),") = ",
#            round(anovaData[lineNum,]['f.col'],3),
#            ", ", pLines,
#            ", $\\eta^2_p$ = ", round(anovaData[lineNum,]['p.eta^2'],3),
#            ", ", bfLines)
#   }
#   
# }
# 
# showTtestSummary <- function(ttest){
#   
#   if(round(ttest$ttest$p.value,3)==0){
#     pLines = "$p$ < 0.001"
#   }else{
#     pLines = paste0("$p$ = ",round(ttest$ttest$p.value,3))
#   }
#   
#   if(round(exp(ttest$bf@bayesFactor[["bf"]]),3)>100){
#     bfLines = "$BF_{10}$ > 100"
#   }else{
#     bfLines = paste0("$BF_{10}$ = ",round(exp(ttest$bf@bayesFactor[["bf"]]),3))
#   }
#   
#   
#   paste0("$t$(",ttest$ttest$parameter,") = ", round(ttest$ttest$statistic,3), 
#          ", ", pLines,
#          ", $d_z$ = ", round(ttest$cohend$estimate,3),
#          ", ", bfLines)
# }
# 
# showLMESummary <- function(model,mmName){
#   
#   model_summary = summary(model)
#   if(round(summary(model)$coefficients[mmName,"Pr(>|t|)"],3)==0){
#     pLines = "$p$ < 0.001"
#   }else{
#     pLines = paste0("$p$=",round(model_summary$coefficients[mmName,"Pr(>|t|)"],3))
#   }
#   
#   paste0("$t$(",length(unique(model@frame[["sub"]]))-1,") = ", 
#          round(model_summary$coefficients[mmName,"t value"],3),
#          ", ", pLines,
#          ", $d_z$ = ", abs(round(t_to_d(t = model_summary$coefficients[mmName,"t value"], df_error = model_summary$coefficients[mmName,"df"])$d,3))
#   )
# }
# 
# 
# get_cohend <- function(tval, df) {
#   abs(round(t_to_d(t = tval, df_error = df)$d, 3))
# }
# 
# extract_coef_info <- function(sm, terms) {
#   map_dfr(terms, function(term) {
#     est <- sm$coefficients[term, "Estimate"]
#     se <- sm$coefficients[term, "Std. Error"]
#     p <- sm$coefficients[term, "Pr(>|t|)"]
#     tval <- sm$coefficients[term, "t value"]
#     df <- sm$coefficients[term, "df"]
#     
#     data.frame(
#       Estimate = est,
#       factor = term,
#       pVal = p,
#       min = est - se,
#       max = est + se,
#       t = tval,
#       cohend = get_cohend(tval, df),
#       stringsAsFactors = FALSE
#     )
#   })
# }
