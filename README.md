To run training:

Change train_misc, train, wandb projectname
```
python train.py k=100 hparams.batch_size=128 model.alpha=5 > output_logs/train_diffae_nonlinear_nonlinear_alpha5_model719_BS128.log 2>&1
```



To generate:

```
bash run_gen.sh
```


### Models

## Model after 180 epochs (BS = 16)

- alpha = 1
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_1/2024-08-21
```

Results: 

(alpha upto +-5)
```
outputs/2024-08-22/11-17-23
```

(alpha upto +- 7)
```
outputs/2024-08-22/11-26-54
```



- alpha = 3
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_3/2024-08-21
```

Results:

(alpha upto +- 5)
```
outputs/2024-08-22/11-45-13
```

(alpha upto +-7)
```
outputs/2024-08-22/11-49-36
```




- alpha = 10
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_10/2024-08-21
```

Results:

(alpha upto +-5)
```
outputs/2024-08-22/12-28-46
```

(alpha upto +-7)
```
outputs/2024-08-22/12-40-44
```

(alpha upto +-15)
```
outputs/2024-08-22/13-12-52
```



- alpha = 50
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_50/2024-08-21
```


Results:

(alpha upto +-5)
```
outputs/2024-08-22/13-41-15
```

(alpha upto +-7)
```
outputs/2024-08-22/13-46-02
```

(alpha upto +-15)
```
outputs/2024-08-22/13-49-17
```



## Model after 960 epochs (BS = 16)

- alpha = 1
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_1/2024-08-22
```

Results: 

(alpha upto +-5)
```
outputs/2024-08-22/18-22-33
```

(alpha upto +- 7)
```
outputs/2024-08-22/18-26-27
```



- alpha = 3
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_3/2024-08-22
```

Results:

(alpha upto +- 5)
```
outputs/2024-08-22/18-30-04
```

(alpha upto +-7)
```
outputs/2024-08-22/18-33-37
```




- alpha = 10
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_10/2024-08-22
```

Results:

(alpha upto +-5)
```
outputs/2024-08-22/19-03-44
```

(alpha upto +-7)
```
outputs/2024-08-22/18-39-05
```

(alpha upto +-15)
```
outputs/2024-08-22/18-46-30
```





- alpha = 50
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_50/2024-08-22
```





## Model after 940 epochs (BS = 256, K = 10)

- alpha = 1
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_1__BS_256/2024-08-23
```


- alpha = 3
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_3__BS_256/2024-08-23
```



- alpha = 10
```
outputs/run/train/ffhq_10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_10__BS_256/2024-08-23
```




## Model after 720 epochs (BS = 256, K = 100)

- alpha = 1 
```
outputs/run/train/ffhq_100/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_1__BS_128/2024-08-23
```


- alpha = 3
```
outputs/run/train/ffhq_100/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_3__BS_128/2024-08-23
```


- alpha = 5
```
outputs/run/train/ffhq_100/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_5__BS_128/2024-08-23
```




diffaepath: /projects/deepdevpath2/Saranga/diffae/checkpoints/ffhq128_w_newclassifier2/epoch_checkpoints/epoch=939-step=256463.ckpt