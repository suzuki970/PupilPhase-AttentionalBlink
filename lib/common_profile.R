library(easypackages)

liblist <- c("rjson","ggplot2","ggpubr","Cairo","gridExtra","effsize",
             "BayesFactor","reshape",
             "lavaan","semPlot","semptools",
             "RColorBrewer","ggbeeswarm",
             "permutes","quickpsy",
             "psycho","colorspace","knitr","kableExtra","openxlsx",
             "cowplot","fitdistrplus","reshape2","tidyr","tidyverse","ggsignif","ggnewscale",
             "lmerTest","modelsummary","effectsize","purrr","arrow","data.table",
             "R.matlab","patchwork")

libraries(liblist)

bayes_factor_seed <- 20260707

if(exists(".anovakun.env")){
  sys.source("./lib/anovakun_487.R", envir = .anovakun.env)
}else{
  .anovakun.env <- new.env()
  sys.source("./lib/anovakun_487.R", envir = .anovakun.env)
  attach(.anovakun.env)
}

fontFamily = "Times"
CairoFonts(regular = "Arial",bold="Arial")

figsize_w = 553.12

## JSON written by pandas DataFrame.to_json() (column-oriented) -> data.frame
load_json_dataframe <- function(folder, pattern) {
  file_path <- list.files(folder, pattern = pattern, full.names = TRUE)[1]
  txt <- readLines(file_path, warn = FALSE)
  txt <- gsub("\\bNaN\\b", "null", txt)
  df_list <- jsonlite::fromJSON(paste(txt, collapse = "\n"))
  
  all_names <- unique(unlist(lapply(df_list, names)))
  
  rows <- lapply(df_list, function(rec) {
    vals <- lapply(all_names, function(nm) {
      v <- rec[[nm]]
      if (is.null(v) || length(v) == 0) {NA} else {v}
    })
    names(vals) <- all_names
    t(as.data.frame(as.list(vals), stringsAsFactors = FALSE))
  })
  
  df = as.data.frame(rows)

}

setEmptyStyle <- function(gData,config,size_font=40){
  gData <- gData +theme(
    panel.border = element_blank(), 
    axis.ticks.length=unit(2, "cm"),
    axis.ticks.x = element_line(colour = "black",size = 0.5),
    axis.text.x = element_text(colour="black"),
    axis.text.y = element_text(colour="black"),
    panel.background = element_rect(fill = "transparent",size = 0.5),
    panel.grid.major = element_line(colour = NA),
    panel.grid.major.y = element_line(colour = "gray", size = 0.05),
    panel.grid.major.x = element_line(colour = NA),
    panel.grid.minor = element_line(colour = NA),
    axis.ticks = element_line(colour = "black",size = 0.5),
    text = element_text(size = size_font,family = fontFamily),
    legend.title = element_text(size=(size_font*0.6)),
    legend.text = element_text(size=(size_font*0.6)),
    legend.key=element_rect(colour="transparent", fill=NA),
    plot.background=element_rect(fill="transparent", colour=NA),
    legend.background=element_rect(fill="transparent", colour=NA),
    plot.title = element_text(size = size_font,hjust=-0.2)
  )
  
  config$xlim_stride = (config$xlim[2] - config$xlim[1]) / 2
  config$ylim_stride = (config$ylim[2] - config$ylim[1]) / 2
  
  if(length(config$ylim) > 1){
    gData = gData +
      scale_y_continuous(breaks=config$ylim)+
      coord_cartesian(ylim=c(config$ylim[1]-config$ylim_stride,rev(config$ylim)[1]+config$ylim_stride),expand=FALSE)+
      annotate(x=config$xlim[1]-config$xlim_stride, xend=config$xlim[1]-config$xlim_stride,
               y=config$ylim[1], yend=rev(config$ylim)[1],
               colour="black", lwd=0.5, geom="segment")+
      theme(
        axis.ticks = element_line(colour = "black",size = 0.5)
      )
  }
  
  if(length(config$xlim) > 1){
    gData = gData +
      coord_cartesian(xlim=c(config$xlim[1]-config$xlim_stride,rev(config$xlim)[1]+config$xlim_stride),expand=FALSE) +
      annotate(x=config$xlim[1],xend=rev(config$xlim)[1],
               y=config$ylim[1]-config$ylim_stride, yend=config$ylim[1]-config$ylim_stride,
               colour="black", lwd=0.5, geom="segment")
  }
  
  return(gData)
}

dispLineGraph <- function(ribbondata, config, data_y, factors, numOfSub = 0, markerSize=15){
  
  ribbondata$data_y = ribbondata[[data_y]]
  
  if(numOfSub == 0) {numOfSub = length(unique(ribbondata$sub))}
  
  fml <- as.formula(
    paste("data_y ~", paste(factors, collapse = "*"))
  )

  std_data   <- aggregate(fml, data = ribbondata, FUN = sd)
  ribbondata <- aggregate(fml, data = ribbondata, FUN = mean)
  
  std_data$data_y = std_data$data_y / sqrt(numOfSub)
  
  ribbondata$SE_min <- ribbondata$data_y - std_data$data_y
  ribbondata$SE_max <- ribbondata$data_y + std_data$data_y
  if(length(factors) == 1){ 
    eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],
                           ", y = data_y,color = ", factors[1], ",group = ",factors[1], "))")))
    eval(parse(text=paste0("p = p + geom_point(size = markerSize)")))
  } else if(length(factors) == 2){
    if(!is.null(config$colFactor)){
      eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y",
                             ", group = ",factors[2], ",color = ",config$colFactor, "))")))
    }else{
      eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y",
                             ", group = ",factors[2], ",color = ",factors[2], "))")))
    }# ",group = interaction(",factors[1],",",factors[2],
    
    eval(parse(text=paste0("p = p + geom_point(size = markerSize, position = position_dodge(.1))")))
    
    if(config$line){
      eval(parse(text=paste0("p = p + geom_line(position = position_dodge(.1))")))
    }
    
  } else{
    
    if(is.null(config$colFactor)){
      color_factor=factors[2]
    }else{
      color_factor=config$colFactor
    }
    
    if(config$shape){
      eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y, ",
                             "group = interaction(",factors[2],",",factors[3],')',
                             ", shape = ",factors[3],
                             ", color = ",color_factor, "))")))
      
      eval(parse(text=paste0("p = p + scale_shape_manual(values = 0:length(unique(ribbondata$",factors[3],")))")))
      
    }else{
      eval(parse(text=paste0("p <- ggplot(ribbondata,aes(x = ", factors[1],", y = data_y, ",
                             "group = ", factors[2],
                             ", color = ",color_factor, "))")))
    }
    eval(parse(text=paste0("p = p + geom_point(size = markerSize, position = position_dodge(.1))")))
    
    if(config$line){
      eval(parse(text=paste0("p = p + geom_line(position = position_dodge(.1))")))
    }
    
  }
  
  if(!is.null(config$grCol)){
    p = p + scale_color_manual(values = config$grCol)
  }
  if(!is.null(config$title)){
    p = p + ggtitle(config$title)
  }
  p = p +  
    geom_errorbar(aes(ymin = SE_min, ymax = SE_max),size = 0.1, width = 0.1, position = position_dodge(.1),color="black")+ 
    xlab(config$label_x) + ylab(config$label_y) +
    theme(
      axis.ticks.x = element_blank(),
      axis.line.x = element_blank()
    )
  
  return(p)
}

disp <- function(ribbondata, config, data_y, shadeFl, factors, numOfSub=0){
  
  eval(parse(text=paste0("ribbondata$data_y = ribbondata$", data_y)))
  
  if (shadeFl == 1) {
    if(numOfSub == 0) {numOfSub = length(unique(ribbondata$sub))}
    
    if(length(factors) == 1){ 
      eval(parse(text=paste0("data_std = aggregate( data_y ~ data_x * ",factors[1],", data = ribbondata, FUN = 'sd')"))) 
      eval(parse(text=paste0("ribbondata = aggregate( data_y ~ data_x * ",factors[1],", data = ribbondata, FUN = 'mean')")))
    }
    else if(length(factors) == 2){ 
      eval(parse(text=paste0(
        "data_std = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],
        ", data = ribbondata, FUN = 'sd')"))) 
      eval(parse(text=paste0(
        "ribbondata = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],
        ", data = ribbondata, FUN = 'mean')")))
    }else{
      eval(parse(text=paste0(
        "data_std = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],"*",factors[3],
        ", data = ribbondata, FUN = 'sd')"))) 
      eval(parse(text=paste0(
        "ribbondata = aggregate( data_y ~ data_x * ",factors[1],"*",factors[2],"*",factors[3],
        ", data = ribbondata, FUN = 'mean')")))
    }
    
    data_std$data_y <- data_std$data_y / sqrt(numOfSub)
    ribbondata$ymin <- ribbondata$data_y - data_std$data_y
    ribbondata$ymax <- ribbondata$data_y + data_std$data_y
    
    if(length(factors) == 1){ 
      eval(parse(text=paste0(
        "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],", group = ",factors[1],")) +",
        "geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = ",factors[1],", group = ",factors[1], "),
      color = 'gray', fill = 'gray', alpha = config$alpha, size = 0.05)")))
    }
    else if(length(factors) == 2){ 
      eval(parse(text=paste0(
        "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[2],
        ", group =  interaction(",factors[1],",",factors[2],")))+",
        "geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = ",factors[1],",),
      color = 'gray', fill = 'gray', alpha = config$alpha, size = 0.05)")))
      
    }else{
      eval(parse(text=paste0(
        "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],
        ", group = interaction(",factors[1],",",factors[2],",",factors[3],")))+",
        "geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = ",factors[1],"),",
        "color = 'gray', fill = 'gray', alpha = config$alpha, size = 0.05) +",
        "geom_line(aes(linetype = ",factors[3],"))"
      )))
    }
    
    if(config$linetype){
      eval(parse(text=paste0("p = p + ",
                             "geom_line(aes(linetype = ",factors[1],"))"
      )))
    }else{
      p = p + geom_line()
    }
    
    if(!is.null(config$grCol)){
      p <- p +
        scale_fill_manual(values = config$grCol)+
        scale_color_manual(values = config$grCol)
    }
    
    p <- p +
      ggtitle(config$title) +
      xlab(config$label_x) + ylab(config$label_y) +
      coord_cartesian(xlim = config$lim_x,ylim = config$lim_y) +
      scale_x_continuous(expand = c(0, 0)) 
    
  }else{ 
    
    eval(parse(text=paste0(
      "p <- ggplot(ribbondata,aes(x = data_x, y = data_y, colour = ", factors[1],", group = ",factors[2],"))+", 
      "geom_line()")))
    
    if(!is.null(config$grCol)){
      p <- p +
        scale_color_manual(values = config$grCol)
    }
    p <- p + 
      geom_vline(xintercept=0, colour='black', linetype='longdash', size = 0.1) +
      ggtitle(config$title) +
      xlab(config$label_x) + ylab(config$label_y) +
      coord_cartesian(xlim=config$lim_x, ylim=config$lim_y) +
      scale_x_continuous(expand = c(0, 0))
    
  }
  
  return(p)
}

getTtestSummary <- function(x1,x2){
  
  datSummary = list()
  set.seed(bayes_factor_seed)
  datSummary$bf = ttestBF(x = x1, y = x2, paired=TRUE)
  datSummary$ttest = t.test(x1, x2, var.equal=T, paired=TRUE)
  dz = as.numeric(datSummary$ttest$statistic) / sqrt(as.numeric(datSummary$ttest$parameter) + 1)
  datSummary$cohend = list(estimate = dz)
  
  if((datSummary$ttest$p.value > 0.05)&(datSummary$ttest$p.value <= 0.1)){
    datSummary$sig = "+"
  }else if((datSummary$ttest$p.value > 0.01)&(datSummary$ttest$p.value <= 0.05)){
    datSummary$sig = "*"
  }else if((datSummary$ttest$p.value > 0.005)&(datSummary$ttest$p.value <= 0.01)){
    datSummary$sig = "**"
  }else if(datSummary$ttest$p.value <= 0.005){
    datSummary$sig = "**"
  }else{
    datSummary$sig = "n.s."
  }
  return(datSummary)
  
}

showAnovaSummary <- function(anovaData,bf_table,lineNum,factorNum){
  
  if(round(anovaData[lineNum,]['p.col'],3)==0){
    pLines = "p < 0.001"
  }else{
    pLines = paste0("$p$ = ",round(anovaData[lineNum,]['p.col'],3))
  }
  
  bf_log = bf_table@bayesFactor[["bf"]]
  if(length(bf_log) >= 4 & factorNum %in% 1:3){
    if(factorNum == 1){
      # A main effect adjusted for B: BF(A + B + sub) / BF(B + sub)
      bf_val = exp(bf_log[3] - bf_log[2])
    }else if(factorNum == 2){
      # B main effect adjusted for A: BF(A + B + sub) / BF(A + sub)
      bf_val = exp(bf_log[3] - bf_log[1])
    }else{
      # Interaction adjusted for main effects.
      bf_val = exp(bf_log[4] - bf_log[3])
    }
  }else{
    bf_val = exp(bf_log[factorNum])
  }
  
  if(round(bf_val,3)>100){
    bfLines = "$BF_{10}$ > 100"
  }else{
    bfLines = paste0("$BF_{10}$ = ",round(bf_val,3))
  }
  
  paste0("$F$(", round(anovaData[lineNum,]['df.col'],3),",",
         round(anovaData[lineNum+1,]['df.col'],3),") = ",
         round(anovaData[lineNum,]['f.col'],3),
         ", ", pLines,
         ", $\\eta^2_p$ = ", round(anovaData[lineNum,]['p.eta^2'],3),
         ", ", bfLines)
  
}

showTtestSummary <- function(ttest){
  
  if(round(ttest$ttest$p.value,3)==0){
    pLines = "$p$ < 0.001"
  }else{
    pLines = paste0("$p$ = ",round(ttest$ttest$p.value,3))
  }
  
  if(round(exp(ttest$bf@bayesFactor[["bf"]]),3)>100){
    bfLines = "$BF_{10}$ > 100"
  }else{
    bfLines = paste0("$BF_{10}$ = ",round(exp(ttest$bf@bayesFactor[["bf"]]),3))
  }
  
  dz = as.numeric(ttest$ttest$statistic) / sqrt(as.numeric(ttest$ttest$parameter) + 1)
  
  paste0("$t$(",ttest$ttest$parameter,") = ", round(ttest$ttest$statistic,3), 
         ", ", pLines,
         ", $d_z$ = ", round(dz,3),
         ", ", bfLines)
}

showEstimateSummary <- function(estimate_row){
  
  if(is.data.frame(estimate_row)){
    estimate_row = estimate_row[1,]
  }
  
  if(round(estimate_row$pVal,3)==0){
    pLines = "$p$ < 0.001"
  }else{
    pLines = paste0("$p$ = ",round(estimate_row$pVal,3))
  }
  
  paste0("$\\beta$ = ", round(estimate_row$Estimate,3),
         ", 95% CI [", round(estimate_row$min,3), ", ", round(estimate_row$max,3), "]",
         ", $\\chi^2$(", estimate_row$chisq_df, ") = ", round(estimate_row$chisq,3),
         ", ", pLines,
         ", partial $r$ = ", round(estimate_row$partial_r,3))
}

get_cohend <- function(tval, df) {
  abs(round(t_to_d(t = tval, df_error = df)$d, 3))
}

extract_coef_info <- function(sm, terms) {
  map_dfr(terms, function(term) {
    est <- sm$coefficients[term, "Estimate"]
    se <- sm$coefficients[term, "Std. Error"]
    p <- sm$coefficients[term, "Pr(>|t|)"]
    tval <- sm$coefficients[term, "t value"]
    df <- sm$coefficients[term, "df"]
    
    data.frame(
      Estimate = est,
      factor = term,
      pVal = p,
      ## 95% CI with Satterthwaite df (lmerTest), not +/- 1 SE
      min = est - qt(0.975, df) * se,
      max = est + qt(0.975, df) * se,
      t = tval,
      df = df,
      partial_r = sign(tval) * sqrt(tval^2 / (tval^2 + df)),
      cohend = get_cohend(tval, df),
      stringsAsFactors = FALSE
    )
  })
}
