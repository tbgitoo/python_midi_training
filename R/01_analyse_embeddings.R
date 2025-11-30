library(umap)
load("data_sets/faiss/embeddings.rda")


for_umap=embeddings_data[,4:ncol(embeddings_data)]

umap=umap(for_umap)

plot(umap$layout[1:1000,], pch=20, 
col=as.integer(as.factor(embeddings_data$track_id[1:1000]))+1)