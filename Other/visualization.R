library(ggplot2)
library(cowplot)
library(scales)
library(tidyr)
library(dplyr)
library(RColorBrewer)

dir = getwd()
bases_4 = read.csv(paste0(dir,"/bases_4/training/history.csv"), head=F)
kmer_5 = read.csv(paste0(dir,"/kmer_5/training/history.csv"), head=F)
kmer_10 = read.csv(paste0(dir,"/kmer_10/training/history.csv"), head=F)
history = rbind(bases_4, kmer_5, kmer_10)
history = cbind(epoch=rep(1:500,3), history, model=rep(c("bases_4","kmer_5","kmer_10"), each=500))
colnames(history) = c("epoch","loss_training","accuracy_training","loss_testing","accuracy_testing","model")

history %>% 
    gather(key, value, -epoch, -model) %>% 
    extract(key, c("eval","set"), "(loss|accuracy)_(training|testing)") %>%
    spread(eval, value) -> df.history
df.history = cbind(df.history, dummy=paste(df.history$model, df.history$set))

loss = ggplot(df.history, aes(x=epoch, y=loss, color=dummy, group=dummy)) + 
	geom_point(show.legend=T) + 
	geom_line(show.legend=T) + 
	scale_y_continuous(breaks=c(0,0.5,1:8), limits=c(0,8), expand=c(0,0)) + 
        scale_x_continuous(expand=c(0,0)) +
	scale_color_brewer(palette="Paired", name="") + 
	background_grid(major="xy", colour.major="grey80", minor="none") + 
	theme(legend.position=c(0.5,0.8), 
		legend.title=element_blank(), 
		legend.background=element_rect(fill="grey95"))

accu = ggplot(df.history, aes(x=epoch, y=accuracy, color=dummy, group=dummy)) + 
	geom_point(show.legend=F) + 
	geom_line(show.legend=F) + 
	scale_y_continuous(limits=c(0,1), breaks=seq(0,1,0.1), expand=c(0,0)) +
        scale_x_continuous(expand=c(0,0)) + 
	scale_color_brewer(palette="Paired") + 
	background_grid(major="xy", colour.major="grey80", minor="none")

plot_grid(loss, accu)
