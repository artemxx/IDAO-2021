# IDAO-2021

### Useful links
* [IDAO First Task Presentation](https://youtu.be/VzH_58yYz5k)
* [Transfer learning for computer vision: PyTorch tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* [Editorial of the competition on Kaggle, competition also about pictures](https://www.youtube.com/watch?v=gdBVOIfeW98&t=1588s)

### Как запускать
* Обучать сетки и делать предсказания в _train_regression.ipynb_ и _train_classification.ipynb_, там всё сохраняется в файлы
* В _merge_results.ipynb_ объединить предсказания и засылать посылку

### Libraries versions
* _torchvision==0.8.2_
* _numpy==1.19.5_
* _torch==1.7.1_
* _pickle==4.0_

### TODO
* Вместо __resnet18__ юзать что-то побольше и получше
* У нас картинки размера _(576, 576, 3)_, надо посмотреть какие сетки обучались тоже на таком большом _shape_
* Нужно картинки читать параллельно и батчи на сэмплировать, написать __DataLoader__
* По условию распределение данных во train/public/private равномерные, нужно смотреть на них в _merge_results.ipynb_
* Нужно думать, как не переобучаться под _train_ классы, возможно __Data Augmentation__ или __Dropout__'ы
* Возможно, кроме угла нам даны ещё какие-то бесполезные (полезные) фичи
