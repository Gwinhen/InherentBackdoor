# InherentBackdoor
Repository for ACSAC 2024 paper Exploring Inherent Backdoors in Deep Learning Models

We provide 13 trigger generation methods, include two existing techniques (NC and DualTanh).

Here is the basic command to generate an inherent backdoor from a `source` class to a `target` class. For universal backdoors, use the number of classes as `source`.


```
python3 main.py --opt [generation_method] --pair [source-target]
```

Please select the generation methods from the following list.

|  Methods  |
|:---------:|
| nc |
| dualtanh |
| patch |
| dynamic |
| input_aware |
| composite |
| wanet |
| invisible |
| blend |
| reflection |
| sig |
| filter |
| dfst |

Note: for the composite backdoor, please first download [the StyleGAN model](https://drive.google.com/file/d/1F-SSibh2SCXW6_CYJYIkRVttBTHY-LnO/view?usp=sharing) and place it in the `ckpt/` folder.

## Reference

```
@inproceedings{tao2024exploring,
  title={Exploring Inherent Backdoors in Deep Learning Models},
  author={Tao, Guanhong and Cheng, Siyuan and Wang, Zhenting and Ma, Shiqing and An, Shengwei and Liu, Yingqi and Shen, Guangyu and Zhang, Zhuo and Mao, Yunshu and Zhang, Xiangyu},
  booktitle={2024 Annual Computer Security Applications Conference (ACSAC)},
  pages={923--939},
  year={2024},
  organization={IEEE}
}
```
