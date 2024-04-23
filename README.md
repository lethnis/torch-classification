# Классификация изображений с нуля на PyTorch

Пример классификации изображений с моделями с нуля на PyTorch. [Датасет](https://www.kaggle.com/datasets/alessiocorrado99/animals10) взят с сайта Kaggle.

# Использование
### Обучение

1. Подготовить виртуальную среду. Использовался `python 3.11.8`
2. Установить зависимости. `pip install -r requirements.txt`
3. В терминале запустить `main.py`
   Пример: `python main.py path/to/dataset densenet`
   Если написать `python main.py` без аргументов, то выведется подробная инструкция по использованию.
4. Информация об обучении сохраняется в папку `runs`.

<details>
    <summary>Подробнее об аргументах</summary>
    <ul>
        <li>Обязательные аргументы:
            <ul>
                <li>путь до данных - данные могут быть в формате папок train, val, test, или сразу папки с классами. В этом случае для val будет отобрано 10% датасета.</li>
                <li>модель или путь до модели - есть подготовленные модели mobilenetv1 и densenet. Можно так же указать путь до готовой модели в формате .pt или .pth.</li>
            </ul>
        </li>
        <li>Необязательные аргументы:
            <ul>
                <li>alpha - множитель для модели mobilenetv1.</li>
                <li>reps - список повторений слоёв для densenet.</li>
                <li>bottleneck - для densenet использовать или нет стиль bottleneck</li>
                <li>batch - размер батча</li>
                <li>epochs - количество эпох</li>
                <li>img_size - размер изображений</li>
                <li>augs - какие аугментации применять к изображениям</li>
                <li>name - имя проекта, под которым будет сохраняться информация об обучении</li>
            </ul>
        </li>
    </ul>
    <a>Примеры:</a>
    <code>python main.py path/to/dataset mobilenetv1 alpha=0.5 batch=32 epochs=100 img_size=224 name=experiment1</code>
    <code>python main.py path/to/dataset/ densenet reps=[2,4,8] bottleneck=True augs='soft'</code>
</details>

### Предсказание
Выполнить `python predict.py path/to/best.pt path/to/images`

# Результаты обучения
Результаты обучения сохраняются в папку `runs/project_name` со следующей информацией:
<img src=https://github.com/lethnis/torch-classification/assets/88483002/071bf21a-6a6e-43d4-a6d0-df1d150638f1>

### Графики обучения:
<table>
   <tr>
      <td><img src=runs/animals10/loss.png width=300></td>
      <td><img src=runs/animals10/accuracy.png width=300></td>
      <td><img src=runs/animals10/conf_norm.png width=300></td>
   </tr>
</table>

Достигнута точность ~80%: Как можно её улучшить:

-   экспериментировать с аугментацией
-   экспериментировать с гиперпараметрами(количество эпох, размер батчей, размер изображений и т.д.)
-   собрать больше данных
-   Использовать другие архитектуры (например, ViT, ResNet, EfficientNet)
-   использовать fine-tuning и feature extraction предобученных моделей

### Примеры предсказания модели
<table>
   <tr>
      <td><img src=predictions/chims.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/chiro.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/crunchycat.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/maxwell.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/nelson.jpg width=800></td>
   </tr>
</table>

# TODO

-   [x] save best model
-   [x] add tensorboard support
-   [x] add args.yaml file to store all parameters
-   [ ] add resume training from saved model
