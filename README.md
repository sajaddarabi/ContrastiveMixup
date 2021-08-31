# ContrastiveMixup Pytorch

This repository contains the code for [(arxiv link)](https://arxiv.org/abs/2108.12296) 

```
@misc{darabi2021contrastive,
      title={Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain}, 
      author={Sajad Darabi and Shayan Fazeli and Ali Pazoki and Sriram Sankararaman and Majid Sarrafzadeh},
      year={2021},
      eprint={2108.12296},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Dependencies

The project was run on a conda virtual environment on Ubuntu 18.04.5 LTS.

Checkout the `requirements.txt` file, if you have conda pre-installed `cd` into the directory where you have downloaded the source code and run the following

```
conda create -n contrastivemixup python==3.7
conda activate contrastivemixup

pip install -r requirements.txt
```

## Running Experiments

To run experiments they are launched from the `train.py` file.  For example, to run ContrastiveMixup on MNIST use the following command

`python train.py -c ./configs/mnist/contrastivemixup.json --pretrain`

The trainer, data loader, model, optimizer, settings are all specified in the `./configs/mnist/contrastivemixup.json` file. 
The `--pretrain` options specifies whether to run the pretraining phase (i.e. training the encoder).


## TODO

- [ ] Add instructions to run code
- [ ] File structure
- [ ] Config file instructions
- [ ] Comment/clean code
- [ ] Jupyter lab demo
