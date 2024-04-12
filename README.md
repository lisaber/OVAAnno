# OVAAnno
## introduction
Recent advances in single-cell technologies enable the rapid growth of multi-omics data. Cell type annotation is one common task in analyzing single-cell data. It is a challenge that some cell types in the testing set are not present in the training set (i.e. unknown cell types). Here, we present OVAAnno, an automatic cell types annotation method which utilizes open-set domain adaptation to detect unknown cell types in scATAC-seq data. OVAAnno consists of three parts: Variational AutoEncoder(VAE), a closed-set classifier and an open-set classifier including multiple binary classifiers(Figure \ref{fig1}). Through the Variational AutoEncoder, we can obtain an information-rich embedding. Mapping the embedding to known cell types, the closed-set classifier selects the nearest known cell type for one sample (i.e. one cell). With adversarial training strategy, the open-set classifier identifies whether the sample belongs to known or unknown cell types.

## Installation
    conda env create -f environment.yml

## Tutorial
### tutorial.ipynb
We have provided a quick-start notebook named tutorial.ipynb in script. Users can modify flexibly based on this tutorial to customize the functionality and results according to their needs.

### demo.py
We have provided an alternative demo program, demo.py, which users can run directly using the following command:

    python script/demo.py --train_data train_data_path --test_data test_data_path --sample_weight --class_t --binary

The training and test datasets should both be encapsulated in AnnData structures and saved as .h5ad files. The .obs attribute must contain CellType field, where the initial values of the CellType field in the test set can be any type present in the training set.

The other command-line parameters are as follows:

    --train_data 
        input train dataset path
    --test_data 
        input test dataset path
    --epoch 
        training epoch
    --binary 
        Flag to set whether to use binary cross-entropy loss for reconstruction
    --sample_weight 
        using sample weight
    --batch 
        batch size
    --lr 
        learning rate
    --device 
        GPU device
    --threshold 
        top inteval(0.95 as a better choice)
    --class_t 
        using hardest loss
    --flag 
        save path flag

## Contact
If you have any questions, you can contact me from the email: linyf63@mail2.sysu.edu.cn